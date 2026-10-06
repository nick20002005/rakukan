//! 英単語リストに一致するラテン文字区間の復元
//!
//! 文中のログ境界から最長一致を取り、ローマ字として完結しない3文字以上の語を
//! 復元する。ユーザー指定語は日本語として読めても復元する。
//! 以下は辞書にない造語用に保持している先頭限定フォールバックの説明。
//!
//! ひらがなモードのまま英単語を打つと、読みはローマ字が部分的にかなへ潰れた姿に
//! なる。`seedream` なら `せえdれあm`（トライで解決できなかった子音だけが素の
//! ASCII として残る）。この読みをそのまま [`crate::digits`] のリテラル保護レイヤーへ
//! 渡すと、素の `d` / `m` だけがアルファベット run とみなされ、`せえ` / `れあ` /
//! 後続の日本語が別々に LLM へ渡る。結果は `せえdレアmのぺーす` のような読めない
//! 文字列になる。
//!
//! ここでは打鍵ログ（[`crate::InputEntry`]）と読みを突き合わせ、**読みの先頭にある
//! ラテン語ランを打鍵どおりのラテン文字へ戻した読み**を作る。
//! `せえdれあmのぺーすはどう` → `seedreamのぺーすはどう`。これを変換器へ渡せば、
//! 既存のリテラル保護レイヤーが `Alpha("seedream")` ＋ `Kana("のぺーすはどう")` に
//! 分割し、LLM は日本語部分だけを見る（→ `seedreamのペースはどう`）。
//!
//! # 境界をログから読む
//! 各エントリは「打鍵（`typed`）」と「そのとき `hiragana_buf` に足した文字列
//! （`output`）」を持ち、未確定ローマ字（`pending_romaji_buf`）はログに入らない。
//! したがって `output` を累積した位置が、そのままローマ字とかなの対応表になる。
//! 変換器に流し直して対応を作り直す必要はない。
//!
//! # 扱える形が限られる理由
//! 素の ASCII は「ローマ字がかなに潰れきらなかった位置」を示すだけで、英単語の
//! 左右の端そのものは指さない。かなは全てローマ字由来なので、境界の手掛かりは
//! 打鍵ログにも存在しない。誤った位置で切ると日本語側を壊すため、両端が決まる形
//! だけを扱う:
//!
//! - 右端は「素の ASCII の直後が助詞」の場合に限る。`seedream` は末尾が `m` で
//!   止まるので `のぺーすはどう` との境目が読める。`google`（読み `ごおgぇ`）は
//!   素の ASCII が途中の `g` で終わるため右端が決まらず、対象外
//! - 左端は「読みの先頭」に限る。しかも先頭であること自体は証明できないので、
//!   ラテン語ランの範囲に助詞のかなが現れたら日本語の前置きを疑って手を引く
//!   （`これはseedreamの…` の `これは` を巻き込まないため）
//!
//! # 既知の限界: 助詞を含まない日本語の前置き
//! 左端の判定は「ラテン語ランの範囲に助詞が無い」ことしか見ないので、助詞を含まない
//! 日本語の前置き（`きょう` `いま` `あした` など）に英単語が続く読みは前置きごと
//! ラテン語ランとみなされる（`kyouseedreamnope-suhadou` → `kyouseedreamのぺーすはどう`）。
//! 前置きが日本語か英単語の一部かは読みからもログからも決まらない。誤発動するのは
//! 読みに素の ASCII が残っている（＝英単語をひらがなモードで打った時点で既に壊れて
//! いる）場合に限られ、正常な日本語の変換を壊すことはないので、この方式の限界として
//! 受け入れている（PR #44 のレビュー、2026-09-12）。
//!
//! # 全体がラテン文字の場合は対象外
//! 後続にかなが無い読みは、この関数の目的である「英単語と日本語の境目を読める
//! ようにする」対象ではないので `None` を返す。

/// 英単語の直後に来る助詞。読みの右端を決める唯一の手掛かりであり、
/// 左端に日本語の前置きが無いことを疑うための手掛かりでもある。
const PARTICLES: [char; 10] = ['の', 'を', 'は', 'が', 'に', 'で', 'と', 'も', 'や', 'へ'];

use std::collections::HashSet;
use std::sync::OnceLock;

struct LatinWords {
    words: HashSet<String>,
    terms: HashSet<String>,
    /// ユーザー指定語のうち、ローマ字としても読み切れる語（`sushi` `make`）。
    /// 後ろの境界を緩めると `makeru` → `makeる` のように日本語を壊すので、
    /// 助詞・「する」の直前でしか復元しない。
    readable: HashSet<String>,
    max_len: usize,
}

impl LatinWords {
    fn from_lists(standard: &str, user: &str) -> Self {
        let mut words = HashSet::new();
        let mut terms = HashSet::new();
        let mut readable = HashSet::new();
        let mut conv = crate::romaji::RomajiConverter::new();
        for (text, explicit) in [(standard, false), (user, true)] {
            for line in text.lines() {
                let word = line.trim();
                if word.len() < 3 || !word.bytes().all(|c| c.is_ascii_lowercase()) {
                    continue;
                }
                conv.reset();
                for c in word.chars() {
                    conv.push(c);
                }
                // 単独 n は「ん」と読めるため日本語側へ倒す。
                let reads_as_romaji = !conv.output().chars().any(|c| c.is_ascii_alphabetic())
                    && (conv.buffer().is_empty() || conv.buffer() == "n");
                if reads_as_romaji && !explicit {
                    continue;
                }
                if explicit {
                    terms.insert(word.to_string());
                    if reads_as_romaji {
                        readable.insert(word.to_string());
                    }
                }
                words.insert(word.to_string());
            }
        }
        let max_len = words.iter().map(String::len).max().unwrap_or(0);
        Self { words, terms, readable, max_len }
    }
}

fn latin_words() -> &'static LatinWords {
    static WORDS: OnceLock<LatinWords> = OnceLock::new();
    WORDS.get_or_init(|| {
        let user = crate::user_dict_path()
            .map(|path| path.with_file_name("latin_words.txt"))
            .and_then(|path| match std::fs::read_to_string(&path) {
                Ok(text) => Some(text),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                Err(err) => {
                    tracing::warn!("latin words: {}: {err}", path.display());
                    None
                }
            })
            .unwrap_or_default();
        let standard = concat!(
            include_str!("../data/latin_scowl.txt"),
            include_str!("../data/latin_extra.txt")
        );
        let mut dictionary = LatinWords::from_lists(standard, &user);
        dictionary.terms = include_str!("../data/latin_extra.txt").lines()
            .chain(user.lines().map(str::trim))
            .filter(|word| dictionary.words.contains(*word))
            .map(str::to_string).collect();
        dictionary
    })
}

/// ログの境界で最長一致を探し、互いに重ならない区間を左から復元する。
/// 未確定末尾は仮想エントリとして扱い、実際の入力状態は変更しない。
pub(crate) fn normalize_latin_spans(
    log: &[crate::InputEntry],
    detached_at: usize,
    hiragana: &str,
    pending: &str,
) -> Option<String> {
    normalize_with_words(log, detached_at, hiragana, pending, latin_words())
}

fn normalize_with_words(
    log: &[crate::InputEntry],
    detached_at: usize,
    hiragana: &str,
    pending: &str,
    dictionary: &LatinWords,
) -> Option<String> {
    if log.is_empty() || detached_at != 0 {
        return None;
    }
    let logged: String = log.iter().map(|e| e.output.as_str()).collect();
    if logged != hiragana {
        return None;
    }
    let mut entries: Vec<(&str, &str, bool)> = log.iter()
        .map(|e| (e.typed.as_str(), e.output.as_str(), e.kind == crate::InputKind::Romaji))
        .collect();
    if !pending.is_empty() {
        entries.push((pending, "", true));
    }
    let mut offsets = vec![0];
    for (_, output, _) in &entries {
        offsets.push(offsets.last().unwrap() + output.len());
    }
    let mut restored = String::new();
    let mut cursor = 0;
    let mut start = 0;
    let mut changed = false;
    while start < entries.len() {
        let mut typed = String::new();
        let mut longest = None;
        for end in start..entries.len() {
            let (keys, _, romaji) = entries[end];
            if !romaji || !keys.bytes().all(|c| c.is_ascii_lowercase()) {
                break;
            }
            typed.push_str(keys);
            if typed.len() > dictionary.max_len {
                break;
            }
            if !dictionary.words.contains(&typed) {
                continue;
            }
            // 一般英単語の接尾辞だけを未知語から拾わない（seedream 内の dream 等）。
            // 固有名詞・ユーザー指定語には辞書で明示した強い手掛かりがある。
            if start > 0 && !dictionary.terms.contains(&typed)
                && !hiragana[..offsets[start]].chars().next_back().is_some_and(is_boundary_particle)
            {
                continue;
            }
            // 区間の最後が促音になった子音で、その手前がかなとして読み切れているなら
            // 日本語の途中（`ki` + `t`(っ) + `to` = きっと。`kit` + と ではない）。
            // 手前に素の ASCII が残る語（`discord` + `da` の `d`）は英単語の末尾。
            if entries[end].1 == "っ"
                && !hiragana[offsets[start]..offsets[end]].chars().any(|c| c.is_ascii_alphabetic())
            {
                continue;
            }
            let right = &hiragana[offsets[end + 1]..];
            // 助詞・「する」の活用、または日本語の前置きがある末尾だけを境界とする。
            let at_end = end + 1 == entries.len();
            let boundary = right.chars().next().is_some_and(is_boundary_particle)
                || ["する", "した", "して", "しない", "します", "すれば", "しよう"]
                    .iter().any(|suffix| right.starts_with(suffix));
            // 補助リスト・ユーザー指定語でローマ字として読めない語は、後ろが何でも
            // 英単語として切る（`googleから` `githubって` `discordだよ`）。
            let loose = dictionary.terms.contains(&typed) && !dictionary.readable.contains(&typed);
            if (at_end && start > 0) || (!right.is_empty() && (boundary || loose)) {
                longest = Some((end + 1, typed.clone()));
            }
        }
        if let Some((end, word)) = longest {
            restored.push_str(&hiragana[cursor..offsets[start]]);
            restored.push_str(&word);
            cursor = offsets[end];
            start = end;
            changed = true;
        } else {
            start += 1;
        }
    }
    if changed {
        restored.push_str(&hiragana[cursor..]);
        Some(restored)
    } else {
        None
    }
}

fn is_boundary_particle(c: char) -> bool {
    PARTICLES.contains(&c)
}

/// 読みの先頭ラテン語ランを打鍵どおりのラテン文字へ戻した読みを返す。
///
/// `log` は [`crate::RakunEngine`] の `input_log`、`detached_at` は
/// `log_detached_at`、`hiragana` は `hiragana_buf` を渡す。復元できない場合は
/// `None`（呼び出し側は読みをそのまま使う）。
pub(crate) fn normalize_leading_latin(
    log: &[crate::InputEntry],
    detached_at: usize,
    hiragana: &str,
) -> Option<String> {
    // F9/F10 の force_preedit 後はログと読みが対応しない。
    if hiragana.is_empty() || log.is_empty() || detached_at != 0 {
        return None;
    }
    // Backspace 再生などでログと読みがずれていたら手を出さない。
    let logged: String = log.iter().map(|entry| entry.output.as_str()).collect();
    if logged != hiragana {
        return None;
    }

    let chars: Vec<char> = hiragana.chars().collect();
    // 読みに素の ASCII 英字が残っている＝ローマ字がかなに変換しきれていない。
    // 普通の日本語の読みはここで弾かれる。
    let split = chars.iter().rposition(|c| c.is_ascii_alphabetic())? + 1;
    if split >= chars.len() {
        // 後続のかなが無い＝読み全体が英単語。分ける境目が無いので何もしない。
        return None;
    }
    // 素の ASCII の直後が助詞のときだけ「ここで英単語が終わった」と判断する。
    // 助詞以外が続くなら、英単語がまだ続いているのか日本語が始まったのかを読みから
    // 区別できない（`ごおgぇ` = google / `せえdれあmつかう` = 英単語＋助詞なしの続き）。
    if !is_boundary_particle(chars[split]) {
        return None;
    }
    // ラテン語ランの範囲に助詞が混ざっていたら、そこまでが日本語の前置きである
    // 可能性を否定できないので復元しない。`これはせえdれあmのぺーす` は最後の素
    // ASCII が `m`・直後が `の` なので上の判定は通ってしまうが、ここで `は` を見て
    // 手を引く。前置きを巻き込むと `korehaseedream` がリテラル化され、日本語側が
    // 丸ごと壊れる。英単語のローマ字が助詞と同じかなを生む語（`monitor` →
    // `もにとr`）も復元されなくなるが、その場合は復元しないだけで害が無い。
    if chars[..split].iter().copied().any(is_boundary_particle) {
        return None;
    }

    // `output` の累積がちょうど `split` になるエントリ境界を探す。境界がエントリの
    // 内側にある（跨いでしまう）場合は復元できない。
    let mut output_len = 0;
    let end = log.iter().position(|entry| {
        output_len += entry.output.chars().count();
        output_len == split
    })? + 1;
    // 記号・数字・Shift+英字が混ざる範囲は digits.rs のリテラル保護レイヤーの担当。
    if log[..end]
        .iter()
        .any(|entry| entry.kind != crate::InputKind::Romaji)
    {
        return None;
    }
    let mut head: String = log[..end]
        .iter()
        .map(|entry| entry.typed.as_str())
        .collect();
    if !head.chars().all(|c| c.is_ascii_alphanumeric())
        || !head.chars().any(|c| c.is_ascii_alphabetic())
    {
        return None;
    }
    head.extend(chars[split..].iter());
    Some(head)
}

#[cfg(test)]
mod tests {
    #[test]
    fn span_readings_match_observed_input() {
        for (typed, reading, pending) in [
            ("kyouhagoogledekensaku", "きょうはごおgぇでけんさく", ""),
            ("githubnipushsita", "ぎてゅbにぷshした", ""),
            ("zoomdekaigi", "ぞおmでかいぎ", ""),
            ("monitorwokau", "もにとrをかう", ""),
            ("korehagithubnohanasi", "これはぎてゅbのはなし", ""),
            ("pythondekaku", "pyてょんでかく", ""),
            ("pythonnohon", "pyてょんおほ", "n"),
            ("kyouhadezoom", "きょうはでぞお", "m"),
            ("sushiwotaberu", "すしをたべる", ""),
            ("tomatowokau", "とまとをかう", ""),
            ("nihongo", "にほんご", ""),
            ("itaiyo", "いたいよ", ""),
            ("makeinu", "まけいぬ", ""),
            ("animewomiru", "あにめをみる", ""),
            ("sorehasoreto", "それはそれと", ""),
        ] {
            let e = engine_after(typed);
            assert_eq!(e.hiragana_text(), reading, "{typed}");
            assert_eq!(e.pending_romaji_buf, pending, "{typed}");
        }
    }

    #[test]
    fn restores_dictionary_spans() {
        for (typed, expected) in [
            ("kyouhagoogledekensaku", "きょうはgoogleでけんさく"),
            ("githubnipushsita", "githubにpushした"),
            ("zoomdekaigi", "zoomでかいぎ"),
            ("monitorwokau", "monitorをかう"),
            ("korehagithubnohanasi", "これはgithubのはなし"),
            ("pythondekaku", "pythonでかく"),
            ("kyouhadezoom", "きょうはでzoom"),
        ] {
            assert_eq!(conv_reading_after(typed), expected, "{typed}");
        }
    }

    #[test]
    fn leaves_japanese_spans_unchanged() {
        for typed in ["kanntannna", "sushiwotaberu", "tomatowokau", "nihongo",
            "itaiyo", "makeinu", "animewomiru", "sorehasoreto"] {
            let e = engine_after(typed);
            assert_eq!(e.conv_reading(), e.hiragana_text(), "{typed}");
        }
    }

    #[test]
    fn explicit_words_override_romaji_filter_and_take_longest() {
        let dictionary = super::LatinWords::from_lists("sushi\nmake\ntomato\nanime\nsake\nno\nto\nit\n", "sushi\nno\nWithCase\nbad-word\n");
        assert!(!dictionary.words.contains("make"));
        assert!(!dictionary.words.contains("it"));
        assert_eq!(dictionary.words.len(), 1);
        let e = engine_after("sushiwotaberu");
        assert_eq!(super::normalize_with_words(&e.input_log, 0, &e.hiragana_buf,
            &e.pending_romaji_buf, &dictionary).as_deref(), Some("sushiをたべる"));
        // 同じ開始位置で二つの語が助詞境界に一致するときも最長を選ぶ。
        let dictionary = super::LatinWords::from_lists("monitor\nmonitorno\n", "");
        let e = engine_after("monitornoha");
        assert_eq!(super::normalize_with_words(&e.input_log, 0, &e.hiragana_buf,
            &e.pending_romaji_buf, &dictionary).as_deref(), Some("monitornoは"));
    }

    #[test]
    fn dictionary_spans_respect_log_guards() {
        let mut e = engine_after("zoomdekaigi");
        assert!(super::normalize_latin_spans(&e.input_log, 1, &e.hiragana_buf, "").is_none());
        assert!(super::normalize_latin_spans(&e.input_log, 0, "違う読み", "").is_none());
        e.input_log[0].kind = crate::InputKind::Raw;
        assert!(super::normalize_latin_spans(&e.input_log, 0, &e.hiragana_buf, "").is_none());
    }

    #[test]
    fn middle_alpha_runs_preserve_both_kana_sides() {
        use crate::digits::Run;
        let reading = conv_reading_after("kyouhagoogledekensaku");
        assert_eq!(crate::digits::split_by_digits(&reading), vec![
            Run::Kana("きょうは".into()), Run::Alpha("google".into()), Run::Kana("でけんさく".into())]);
    }

    #[test]
    fn pending_tail_restoration_does_not_mutate_input() {
        let e = engine_after("kyouhadezoom");
        assert_eq!(e.conv_reading(), "きょうはでzoom");
        assert_eq!(e.conv_reading(), "きょうはでzoom");
        assert_eq!(e.pending_romaji_buf, "m");
        assert_eq!(e.hiragana_text(), "きょうはでぞお");
    }

    /// 促音で終わる日本語（きっと・ネット・セット）を 英単語 + と に化かさない。
    #[test]
    fn sokuon_before_particle_stays_japanese() {
        for typed in ["kittokuru", "nettowa-ku", "settosuru", "hottosita", "korehanettode",
            "pettowokau", "hittosita", "sattokaeru", "guttokuru", "robottowotukuru",
            "merittoga", "yunittowo", "mottohosii", "kippuwokau"] {
            let e = engine_after(typed);
            assert_eq!(e.conv_reading(), e.hiragana_text(), "{typed}");
        }
    }

    /// 補助リストの語は、後ろが助詞・「する」以外でも切る。
    #[test]
    fn listed_terms_split_before_any_kana() {
        for (typed, expected) in [
            ("googlekara", "googleから"),
            ("githubtte", "githubって"),
            ("discorddayo", "discordだよ"),
            ("kyouhagithubmiru", "きょうはgithubみる"),
        ] {
            assert_eq!(conv_reading_after(typed), expected, "{typed}");
        }
        // ローマ字として読めるユーザー指定語は、助詞の前でしか切らない。
        let dictionary = super::LatinWords::from_lists("", "make
");
        let e = engine_after("makeru");
        assert!(super::normalize_with_words(&e.input_log, 0, &e.hiragana_buf,
            &e.pending_romaji_buf, &dictionary).is_none());
    }

    #[test]
    fn merged_nn_boundary_is_not_guessed() {
        // python + no は nn が一音に合流し、単語末尾のログ境界が存在しない。
        let e = engine_after("pythonnohon");
        assert!(e.input_log.iter().any(|entry| entry.typed == "nn"));
        assert_eq!(e.conv_reading(), "pyてょんおほ");
    }
    fn engine_after(typed: &str) -> crate::RakunEngine {
        let mut e = crate::RakunEngine::new(crate::EngineConfig::default());
        for c in typed.chars() {
            e.push_char(c);
        }
        e
    }

    fn conv_reading_after(typed: &str) -> String {
        engine_after(typed).conv_reading()
    }

    /// 各ケースの読みが想定どおりかを先に固定する。ここがずれていると、下の
    /// テストが「復元する / しない」の理由を取り違える。
    /// `seedream` 単体の末尾 `m` は次の打鍵まで未確定バッファに残るので読みに出ない。
    #[test]
    fn readings_match_expected() {
        let cases = [
            ("seedreamnope-suhadou", "せえdれあmのぺーすはどう"),
            ("seedreamnope-", "せえdれあmのぺー"),
            ("kanntannna", "かんたんな"),
            ("seedream", "せえdれあ"),
            ("google", "ごおgぇ"),
            ("seedreamtsukau", "せえdれあmつかう"),
            (
                "korehaseedreamnope-suhadou",
                "これはせえdれあmのぺーすはどう",
            ),
        ];
        for (typed, reading) in cases {
            assert_eq!(
                engine_after(typed).hiragana_text(),
                reading,
                "typed={typed}"
            );
        }
    }

    #[test]
    fn restores_leading_latin_word_before_kana() {
        // 「seedreamのぺーすはどう」と打った状態
        assert_eq!(
            conv_reading_after("seedreamnope-suhadou"),
            "seedreamのぺーすはどう"
        );
    }

    #[test]
    fn restores_leading_latin_word_mid_typing() {
        // 変換が追いつく前（`のぺー` まで打った時点）でも同じ位置で切れる
        assert_eq!(conv_reading_after("seedreamnope-"), "seedreamのぺー");
    }

    #[test]
    fn ignores_pure_kana_reading() {
        assert_eq!(conv_reading_after("kanntannna"), "かんたんな");
    }

    #[test]
    fn ignores_reading_that_is_entirely_latin() {
        // 読み全体が英単語 → 分ける境目が無いので対象外
        assert_eq!(conv_reading_after("seedream"), "せえdれあ");
    }

    #[test]
    fn ignores_latin_word_whose_right_edge_is_unknown() {
        // google は末尾の `ぇ` まで英単語だが、素の ASCII は途中の `g` が最後。
        // ここで切ると `googlれ` に化けるので切らない。
        assert_eq!(conv_reading_after("google"), "ごおgぇ");
    }

    #[test]
    fn ignores_latin_word_not_followed_by_a_particle() {
        // 助詞以外が続くと、英単語が終わったのか続いているのか読みから決まらない
        assert_eq!(conv_reading_after("seedreamtsukau"), "せえdれあmつかう");
    }

    #[test]
    fn ignores_latin_run_after_japanese_prefix() {
        // 前置き `これは` は日本語だが、読みの上では英単語の一部と区別できない。
        // 巻き込んで `korehaseedream` をリテラル化すると日本語側が壊れる。
        assert_eq!(
            conv_reading_after("korehaseedreamnope-suhadou"),
            "これはせえdれあmのぺーすはどう"
        );
    }

    #[test]
    fn folds_particle_free_japanese_prefix_known_limitation() {
        // 既知の限界（モジュール doc 参照）: 助詞を含まない前置き `きょう` / `いま` は
        // 英単語の一部と区別できず、ラテン語ランに巻き込まれる。挙動を固定しておき、
        // 判定を変えたときに気づけるようにする。
        assert_eq!(
            conv_reading_after("kyouseedreamnope-suhadou"),
            "kyouseedreamのぺーすはどう"
        );
        assert_eq!(
            conv_reading_after("imaseedreamnope-suhadou"),
            "imaseedreamのぺーすはどう"
        );
    }

    #[test]
    fn ignores_reading_out_of_sync_with_log() {
        // F9/F10 の force_preedit 後はログと読みが対応しない
        let mut e = engine_after("seedreamno");
        e.force_preedit("SEEDREAMの".to_string());
        assert_ne!(e.log_detached_at, 0);
        assert_eq!(e.conv_reading(), "SEEDREAMの");
    }
}
