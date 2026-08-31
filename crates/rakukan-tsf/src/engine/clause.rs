//! 変換結果（surface）を文節へ分割し、対応する読みを逆算する。
//!
//! ```text
//! 読み:    れーるがんいますぐにつくれない
//! surface: レールガン今すぐに作れない
//!          └カタカナ┘└漢字┘└ひら┘└漢字┘└ひら┘
//! → レールガン(れーるがん) / 今すぐに(いますぐに) / 作れない(つくれない)
//! ```
//!
//! # なぜ surface 側から割るのか
//!
//! [`docs/PHASE9_DESIGN.md`] の当初案は「読みを辞書で最長一致して割る」だったが、
//! それだと文節ごとに独立変換することになり、**LLM が文全体を見て変換していた
//! 文脈が失われて精度が落ちる**（`れーるがんいますぐにつくれない` が一発で
//! `レールガン今すぐに作れない` になっていたのは文全体を渡していたから）。
//!
//! surface から割れば、第 1 候補は文全体の LLM 出力そのままで、表示上だけ
//! 文節に区切られる。精度を落とさずに「文節移動」「部分確定」「語単位の
//! カタカナ判定」が手に入る。
//!
//! # 読みの逆算
//!
//! surface を文字種の run に分け、**ひらがな・カタカナ・英数記号の run を
//! アンカー**として読みの中から順に見つける。漢字 run の読みは「前のアンカーの
//! 終わりから次のアンカーの始まりまで」として確定する。
//!
//! アンカーが 1 つでも見つからなければ `None` を返す。呼び出し側は分割せず
//! 従来どおり 1 ブロックとして扱うこと（**推測で割るくらいなら割らない**）。

use super::text_util::to_hiragana;

/// 文節 1 つ。`reading` と `surface` は必ず対で持つ。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
    pub reading: String,
    pub surface: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunKind {
    /// 読みにそのまま現れる（アンカーになる）
    Hiragana,
    /// ひらがなへ落とせば読みにそのまま現れる（アンカーになる）
    Katakana,
    /// 英数字・記号。読みにそのまま現れる（アンカーになる）
    Literal,
    /// 読みの長さが不明。前後のアンカーから逆算する
    Kanji,
}

fn classify(c: char) -> RunKind {
    match c {
        'ぁ'..='ゖ' | 'ゝ' | 'ゞ' => RunKind::Hiragana,
        'ァ'..='ヶ' | 'ヽ' | 'ヾ' | '・' => RunKind::Katakana,
        // 長音符は直前の run に吸わせる（下の `runs()` で処理）。
        // 単独で現れたときはカタカナ扱い。
        'ー' => RunKind::Katakana,
        c if c.is_ascii_alphanumeric() => RunKind::Literal,
        // 全角英数
        'Ａ'..='Ｚ' | 'ａ'..='ｚ' | '０'..='９' => RunKind::Literal,
        c if c.is_ascii_punctuation() || c.is_ascii_whitespace() => RunKind::Literal,
        _ => RunKind::Kanji,
    }
}

/// surface を文字種の run に分ける。長音符は直前の run に吸わせる。
fn runs(surface: &str) -> Vec<(RunKind, String)> {
    let mut out: Vec<(RunKind, String)> = Vec::new();
    for c in surface.chars() {
        let kind = classify(c);
        match out.last_mut() {
            // 長音符は「ひらがな run のあと」でもその run に吸わせる（`らーめん` 等）
            Some((last_kind, text)) if *last_kind == kind || (c == 'ー' && *last_kind == RunKind::Hiragana) => {
                text.push(c);
            }
            _ => out.push((kind, c.to_string())),
        }
    }
    out
}

/// アンカー run の「読みに現れるはずの文字列」を返す。漢字 run は `None`。
fn anchor_text(kind: RunKind, text: &str) -> Option<String> {
    match kind {
        RunKind::Hiragana | RunKind::Literal => Some(text.to_string()),
        RunKind::Katakana => Some(to_hiragana(text)),
        RunKind::Kanji => None,
    }
}

/// `haystack` の `from` 文字目以降から `needle` を探し、見つかった **文字** 位置を返す。
fn find_chars(haystack: &[char], from: usize, needle: &[char]) -> Option<usize> {
    if needle.is_empty() || from > haystack.len() {
        return None;
    }
    (from..=haystack.len().saturating_sub(needle.len()))
        .find(|&i| haystack[i..i + needle.len()] == *needle)
}

/// 変換結果を文節へ分割する。分割できなければ `None`。
///
/// `None` を返すのは以下のとき。呼び出し側は 1 ブロックのまま扱うこと。
/// - アンカーが読みの中に見つからない（LLM が読みに無い文字を出した等）
/// - 読みを使い切れなかった / 足りなかった
/// - 文節が 1 つしかできなかった（割る意味が無い）
pub fn split_into_clauses(reading: &str, surface: &str) -> Option<Vec<Clause>> {
    if reading.is_empty() || surface.is_empty() {
        return None;
    }
    let r: Vec<char> = reading.chars().collect();
    let runs = runs(surface);

    // 各 run の読み（文字範囲）を確定する
    let mut run_readings: Vec<Option<(usize, usize)>> = vec![None; runs.len()];
    let mut pos = 0usize;
    let mut pending_kanji: Option<usize> = None;

    for (i, (kind, text)) in runs.iter().enumerate() {
        let Some(anchor) = anchor_text(*kind, text) else {
            // 漢字 run。次のアンカーが来るまで長さを決められない。
            // run は文字種で切っているので漢字 run が連続することはない。
            debug_assert!(pending_kanji.is_none(), "漢字 run が連続した: {surface}");
            pending_kanji = Some(i);
            continue;
        };
        let needle: Vec<char> = anchor.chars().collect();
        let hit = if pending_kanji.is_some() {
            // 漢字の読みは最低 1 文字。pos ちょうどで見つかっても採用しない。
            find_chars(&r, pos + 1, &needle)?
        } else {
            let at = find_chars(&r, pos, &needle)?;
            // 保留中の漢字が無いなら、アンカーは今の位置から始まっていなければ
            // ならない。ずれていたら読みと surface が対応していない。
            if at != pos {
                return None;
            }
            at
        };
        if let Some(k) = pending_kanji.take() {
            run_readings[k] = Some((pos, hit));
        }
        run_readings[i] = Some((hit, hit + needle.len()));
        pos = hit + needle.len();
    }

    if let Some(k) = pending_kanji.take() {
        // 末尾が漢字 run。残りの読みを全部あてる。
        if pos >= r.len() {
            return None;
        }
        run_readings[k] = Some((pos, r.len()));
        pos = r.len();
    }
    if pos != r.len() {
        return None;
    }

    // run を文節へまとめる。
    // 文節 = [漢字 / カタカナ の run] + 後続のひらがな run。
    // 文頭のひらがな run は単独で 1 文節。
    //
    // 英数記号 run は文節を切らない。`4枚の` `第3の` のように数字と語が
    // 一体の文節を作るため、英数記号は現在の文節に吸わせ、その直後の
    // 漢字・カタカナ run も同じ文節に入れる。
    let mut clauses: Vec<Clause> = Vec::new();
    let mut prev_kind: Option<RunKind> = None;
    for (i, (kind, text)) in runs.iter().enumerate() {
        let (rs, re) = run_readings[i]?;
        let reading_part: String = r[rs..re].iter().collect();
        let attach = !clauses.is_empty()
            && match kind {
                RunKind::Hiragana | RunKind::Literal => true,
                // `4` + `枚` のように英数記号の直後なら同じ文節
                RunKind::Kanji | RunKind::Katakana => prev_kind == Some(RunKind::Literal),
            };
        prev_kind = Some(*kind);
        if attach {
            let last = clauses.last_mut()?;
            last.reading.push_str(&reading_part);
            last.surface.push_str(text);
        } else {
            clauses.push(Clause {
                reading: reading_part,
                surface: text.clone(),
            });
        }
    }

    if clauses.len() < 2 {
        return None;
    }
    Some(clauses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clauses(reading: &str, surface: &str) -> Vec<(String, String)> {
        split_into_clauses(reading, surface)
            .unwrap_or_else(|| panic!("分割できなかった: {reading:?} / {surface:?}"))
            .into_iter()
            .map(|c| (c.reading, c.surface))
            .collect()
    }

    #[test]
    fn splits_katakana_kanji_hiragana_mix() {
        assert_eq!(
            clauses("れーるがんいますぐにつくれない", "レールガン今すぐに作れない"),
            vec![
                ("れーるがん".to_string(), "レールガン".to_string()),
                ("いますぐに".to_string(), "今すぐに".to_string()),
                ("つくれない".to_string(), "作れない".to_string()),
            ]
        );
    }

    #[test]
    fn splits_typical_sentence() {
        assert_eq!(
            clauses("わたしはがっこうへいきます", "私は学校へ行きます"),
            vec![
                ("わたしは".to_string(), "私は".to_string()),
                ("がっこうへ".to_string(), "学校へ".to_string()),
                ("いきます".to_string(), "行きます".to_string()),
            ]
        );
    }

    #[test]
    fn leading_hiragana_run_is_its_own_clause() {
        assert_eq!(
            clauses("そのほんをよむ", "その本を読む"),
            vec![
                ("その".to_string(), "その".to_string()),
                ("ほんを".to_string(), "本を".to_string()),
                ("よむ".to_string(), "読む".to_string()),
            ]
        );
    }

    #[test]
    fn digits_and_counters_stay_aligned() {
        assert_eq!(
            clauses("4まいのしゃしん", "4枚の写真"),
            vec![
                ("4まいの".to_string(), "4枚の".to_string()),
                ("しゃしん".to_string(), "写真".to_string()),
            ]
        );
    }

    #[test]
    fn literal_run_does_not_break_a_clause() {
        assert_eq!(
            clauses("だい3のおとこ", "第3の男"),
            vec![
                ("だい3の".to_string(), "第3の".to_string()),
                ("おとこ".to_string(), "男".to_string()),
            ]
        );
    }

    #[test]
    fn trailing_kanji_takes_the_rest() {
        assert_eq!(
            clauses("あかいはな", "赤い花"),
            vec![
                ("あかい".to_string(), "赤い".to_string()),
                ("はな".to_string(), "花".to_string()),
            ]
        );
    }

    #[test]
    fn long_vowel_mark_joins_the_hiragana_run() {
        // 「らーめん」のようにひらがな内に長音符がある形でもアンカーが崩れない
        assert_eq!(
            clauses("らーめんをたべる", "らーめんを食べる"),
            vec![
                ("らーめんを".to_string(), "らーめんを".to_string()),
                ("たべる".to_string(), "食べる".to_string()),
            ]
        );
    }

    #[test]
    fn single_clause_is_not_split() {
        // 割る意味が無いものは None（呼び出し側は 1 ブロックのまま）
        assert!(split_into_clauses("でもりっしゃー", "デモリッシャー").is_none());
        assert!(split_into_clauses("へんかん", "変換").is_none());
    }

    #[test]
    fn returns_none_when_surface_does_not_match_reading() {
        // LLM が読みに無い文字を出した / 読みと surface が対応していない
        assert!(split_into_clauses("わたしはがっこうへ", "私は会社へ行きます").is_none());
        assert!(split_into_clauses("", "私は").is_none());
        assert!(split_into_clauses("わたしは", "").is_none());
    }

    #[test]
    fn kanji_reading_is_at_least_one_char() {
        // 「は」が読みの先頭にもあるが、漢字の読みを 0 文字にはしない
        assert_eq!(
            clauses("はしをわたる", "橋を渡る"),
            vec![
                ("はしを".to_string(), "橋を".to_string()),
                ("わたる".to_string(), "渡る".to_string()),
            ]
        );
    }
}
