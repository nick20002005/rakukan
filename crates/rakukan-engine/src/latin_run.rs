//! 先頭ラテン語ランの復元
//!
//! ひらがなモードのまま英単語を打つと、読みはローマ字がかなに潰れた姿になる。
//! 「seedream」なら `せえdれあm`（トライで解決できなかった子音だけが素の ASCII
//! として残る）。この読みをそのまま [`crate::digits`] のリテラル保護レイヤーへ
//! 渡すと、素の `d` / `m` だけがアルファベット run とみなされ、
//! `せえ` / `れあ` / 後続の日本語が別々に LLM へ渡る。結果は
//! `せえdレアmのぺーす` のような読めない文字列になる（2026-09-01 に実害）。
//!
//! ここでは打鍵ログ（[`crate::Engine::romaji_log_str`]）と読みを突き合わせ、
//! **読みの先頭にあるラテン語ランを打鍵どおりのラテン文字へ戻した読み**を作る。
//! `せえdれあmのぺーすはどう` → `seedreamのぺーすはどう`。
//! これを変換器へ渡せば、既存のリテラル保護レイヤーが
//! `Alpha("seedream")` ＋ `Kana("のぺーすはどう")` に分割し、LLM は日本語部分
//! だけを見る（→ `seedreamのペースはどう`）。
//!
//! # 対象を「先頭」に限る理由
//! 読みの途中に現れる英単語（`これはseedreamです`）は、どこから英単語が
//! 始まるのかを読みから決められない。かなは全てローマ字由来なので、左境界を
//! 示す手掛かりが打鍵ログに存在しないため。誤った位置で切ると日本語側を
//! 壊すので、境界が自明な「読みの先頭」だけを扱う。
//!
//! # 全体がラテン文字の場合は対象外
//! `mac` のように読み全体が英単語なら [`crate::Engine::romaji_alnum_candidates`]
//! が候補を出す。二重に候補を作らないよう、後続にかなが無い場合は `None`。

use crate::romaji::RomajiConverter;

/// 打鍵ログを [`crate::Engine::push_char`] と同じ手順で流し直し、
/// 「ローマ字 n 文字まで ↔ かな m 文字まで」の境界一覧を返す。
///
/// 各要素は「未確定バッファを含まない」位置。`seedream` の `m` は次の `n` を
/// 打った時点で初めて素の `m` として確定するが、境界としては `m` まで＝
/// ローマ字 8 文字と記録する。ここで `n` まで含めてしまうと復元結果が
/// `seedreamn` になる。
fn boundaries(romaji: &str) -> Vec<(usize, usize)> {
    let mut conv = RomajiConverter::new();
    let mut pending_len = 0usize;
    let mut kana_len = 0usize;
    let mut marks = Vec::new();
    for (i, c) in romaji.chars().enumerate() {
        pending_len += 1;
        let prev_output_len = conv.output().len();
        let _ = conv.push(c);
        let added_kana = conv.output()[prev_output_len..].chars().count();
        let buffered = conv.buffer().chars().count();
        if pending_len < buffered {
            // 想定外（ログとコンバータがずれた）。以降の対応は信用できない。
            return Vec::new();
        }
        let consumed = pending_len - buffered;
        if consumed > 0 {
            pending_len = buffered;
            kana_len += added_kana;
            marks.push((i + 1 - buffered, kana_len));
        }
    }
    marks
}

/// 打鍵ログ全体をかなへ復元する（末尾の未確定バッファも flush する）。
fn replay(romaji: &str) -> String {
    let mut conv = RomajiConverter::new();
    let mut out = String::new();
    for c in romaji.chars() {
        let prev = conv.output().len();
        let _ = conv.push(c);
        out.push_str(&conv.output()[prev..]);
    }
    out.push_str(&conv.flush());
    out
}

/// 読みの先頭ラテン語ランを打鍵どおりのラテン文字へ戻した読みを返す。
///
/// `romaji` は [`crate::Engine::romaji_log_str`]（未確定バッファを含まない）、
/// `hiragana` は `hiragana_buf` を渡す。復元できない場合は `None`。
pub fn normalize_leading_latin(romaji: &str, hiragana: &str) -> Option<String> {
    if romaji.is_empty() || hiragana.is_empty() {
        return None;
    }
    let kana: Vec<char> = hiragana.chars().collect();
    // 読みに素の ASCII 英字が残っている＝ローマ字がかなに変換しきれていない。
    // 普通の日本語の読みはここで弾かれる。
    let last_ascii = kana.iter().rposition(|c| c.is_ascii_alphabetic())?;
    let split = last_ascii + 1;
    if split >= kana.len() {
        // 後続のかなが無い＝読み全体が英単語。romaji_alnum_candidates の担当。
        return None;
    }
    // F9/F10 の force_preedit 後などログと読みが対応しないケースを弾く。
    if replay(romaji) != hiragana {
        return None;
    }
    let romaji_chars: Vec<char> = romaji.chars().collect();
    let romaji_prefix_len = boundaries(romaji)
        .into_iter()
        .find(|&(_, kana_len)| kana_len == split)
        .map(|(romaji_len, _)| romaji_len)?;
    let head: String = romaji_chars.get(..romaji_prefix_len)?.iter().collect();
    // 記号・数字が混ざるものは digits.rs のリテラル保護レイヤーの担当。
    if !head.chars().all(|c| c.is_ascii_alphanumeric()) || !head.chars().any(|c| c.is_ascii_alphabetic())
    {
        return None;
    }
    let tail: String = kana[split..].iter().collect();
    Some(format!("{head}{tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_leading_latin_word_before_kana() {
        // 「seedreamのぺーすはどう」と打った状態
        assert_eq!(
            normalize_leading_latin("seedreamnope-suhadou", "せえdれあmのぺーすはどう").as_deref(),
            Some("seedreamのぺーすはどう")
        );
    }

    #[test]
    fn restores_leading_latin_word_mid_typing() {
        // 変換が追いつく前（`のぺー` まで打った時点）でも同じ位置で切れる
        assert_eq!(
            normalize_leading_latin("seedreamnope-", "せえdれあmのぺー").as_deref(),
            Some("seedreamのぺー")
        );
    }

    #[test]
    fn ignores_pure_kana_reading() {
        assert_eq!(normalize_leading_latin("kanntanna", "かんたんな"), None);
    }

    #[test]
    fn ignores_reading_that_is_entirely_latin() {
        // 読み全体が英単語 → romaji_alnum_candidates の担当
        assert_eq!(normalize_leading_latin("seedream", "せえdれあm"), None);
    }

    #[test]
    fn ignores_reading_out_of_sync_with_log() {
        // force_preedit 後を模した不一致
        assert_eq!(normalize_leading_latin("seedreamno", "SEEDREAMの"), None);
    }
}
