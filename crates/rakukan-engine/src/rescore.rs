//! 長文変換の n-best を辞書との整合で並べ替える。
//!
//! # なぜ必要か
//!
//! 辞書は **読み全体の完全一致** でしか引かれない。`しじぶん` 単独なら
//! ユーザー辞書 / MOZC 辞書が `指示文` を返すのに、
//! `しじぶんもそんなにこまかくはなさそうだし` と伸びた瞬間に辞書は一切
//! 寄与せず、区切りは LLM の勘だけで決まる。読点が無い文はブロック分割も
//! されないので、長文ほど「辞書に載っている語なのに出ない」が起きる。
//!
//! そこで **候補の集合は変えずに順序だけ** 触る。LLM が返した n-best の
//! それぞれについて「表層を読みへ割り戻し、各語が辞書に載っているか」を見て、
//! 辞書との一致が最も強い候補を先頭へ繰り上げる。n-best の中に正しい区切りが
//! 居るのに 2 番目以降で埋もれている、という取りこぼしを拾うのが目的。
//!
//! # 読みの割り戻し
//!
//! `tsf/engine/clause.rs` と同じ「surface 側から割る」方式。表層を文字種の
//! run に分け、ひらがな・カタカナ・英数記号の run をアンカーとして読みの中に
//! 順に見つけ、漢字 run の読みは前後のアンカーに挟まれた区間として確定する。
//! アンカーが 1 つでも見つからなければ割らない（推測で割らない）。
//!
//! 🔴 **候補を増やしも減らしもしない**。順序だけを入れ替える。候補リストへ
//! 何かを差し込む実装は「実候補 0 件の瞬間」に preview を壊してきた
//! （engine の短文予測で 3 回踏んだ）ので、ここでは集合を触らない。

use crate::kana::katakana_to_hiragana;
use rakukan_dict::DictStore;

/// 得点に数える run の最小読み長。単漢字（`は` → `葉` 等）は辞書に載って
/// いても偶然一致しやすく、雑音にしかならないので除く。
const MIN_RUN_READING_CHARS: usize = 2;

/// 辞書の出自ごとの重み。ユーザー辞書 > 学習履歴 > MOZC 辞書。
/// 「登録したのに長文で出ない」を潰すのが主目的なのでユーザー辞書を厚くする。
const WEIGHT_USER: f64 = 3.0;
const WEIGHT_LEARN: f64 = 2.0;
const WEIGHT_DICT: f64 = 1.0;

/// MOZC 辞書を引くときの候補上限。同音異義が多い読みでも表層一致を
/// 取りこぼさない程度に広く取る。
const DICT_LOOKUP_LIMIT: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunKind {
    /// 読みにそのまま現れる（アンカー）
    Hiragana,
    /// ひらがなへ落とせば読みにそのまま現れる（アンカー）
    Katakana,
    /// 英数字・記号。読みにそのまま現れる（アンカー）
    Literal,
    /// 読みの長さが不明。前後のアンカーから逆算する
    Kanji,
}

fn classify(c: char) -> RunKind {
    match c {
        'ぁ'..='ゖ' | 'ゝ' | 'ゞ' => RunKind::Hiragana,
        'ァ'..='ヶ' | 'ヽ' | 'ヾ' | '・' => RunKind::Katakana,
        // 長音符は直前の run に吸わせる（`runs()` 側で処理）。単独ならカタカナ。
        'ー' => RunKind::Katakana,
        c if c.is_ascii_alphanumeric() => RunKind::Literal,
        'Ａ'..='Ｚ' | 'ａ'..='ｚ' | '０'..='９' => RunKind::Literal,
        c if c.is_ascii_punctuation() || c.is_ascii_whitespace() => RunKind::Literal,
        _ => RunKind::Kanji,
    }
}

fn runs(surface: &str) -> Vec<(RunKind, String)> {
    let mut out: Vec<(RunKind, String)> = Vec::new();
    for c in surface.chars() {
        let kind = classify(c);
        match out.last_mut() {
            Some((last_kind, text))
                if *last_kind == kind || (c == 'ー' && *last_kind == RunKind::Hiragana) =>
            {
                text.push(c);
            }
            _ => out.push((kind, c.to_string())),
        }
    }
    out
}

/// アンカー run の「読みに現れるはずの文字列」。漢字 run は `None`。
fn anchor_text(kind: RunKind, text: &str) -> Option<String> {
    match kind {
        RunKind::Hiragana | RunKind::Literal => Some(text.to_string()),
        RunKind::Katakana => Some(katakana_to_hiragana(text)),
        RunKind::Kanji => None,
    }
}

fn find_chars(haystack: &[char], from: usize, needle: &[char]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() || from > haystack.len() - needle.len() {
        return None;
    }
    (from..=haystack.len() - needle.len())
        .find(|&i| haystack[i..i + needle.len()] == *needle)
}

/// 表層 run と、そこに割り当てた読み。
struct AlignedRun {
    kind: RunKind,
    surface: String,
    reading: String,
}

/// 表層を run に割り、各 run の読みを逆算する。割れなければ `None`。
fn align(reading: &str, surface: &str) -> Option<Vec<AlignedRun>> {
    if reading.is_empty() || surface.is_empty() {
        return None;
    }
    let r: Vec<char> = reading.chars().collect();
    let runs = runs(surface);

    let mut run_readings: Vec<Option<(usize, usize)>> = vec![None; runs.len()];
    let mut pos = 0usize;
    let mut pending_kanji: Option<usize> = None;

    for (i, (kind, text)) in runs.iter().enumerate() {
        let Some(anchor) = anchor_text(*kind, text) else {
            // 漢字 run。次のアンカーが来るまで長さを決められない。
            if pending_kanji.is_some() {
                return None;
            }
            pending_kanji = Some(i);
            continue;
        };
        let needle: Vec<char> = anchor.chars().collect();
        let hit = if pending_kanji.is_some() {
            // 漢字の読みは最低 1 文字。pos ちょうどで見つかっても採用しない。
            find_chars(&r, pos + 1, &needle)?
        } else {
            let at = find_chars(&r, pos, &needle)?;
            if at != pos {
                // 保留中の漢字が無いのにずれている＝読みと表層が対応していない。
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
        if pos >= r.len() {
            return None;
        }
        run_readings[k] = Some((pos, r.len()));
        pos = r.len();
    }
    if pos != r.len() {
        return None;
    }

    let mut out = Vec::with_capacity(runs.len());
    for (i, (kind, text)) in runs.into_iter().enumerate() {
        let (rs, re) = run_readings[i]?;
        out.push(AlignedRun {
            kind,
            surface: text,
            reading: r[rs..re].iter().collect(),
        });
    }
    Some(out)
}

/// run 1 つぶんの辞書一致の重み。載っていなければ `None`。
fn run_weight(store: &DictStore, reading: &str, surface: &str) -> Option<f64> {
    if store.lookup_user(reading).iter().any(|c| c == surface) {
        return Some(WEIGHT_USER);
    }
    if store.lookup_learn(reading).iter().any(|c| c == surface) {
        return Some(WEIGHT_LEARN);
    }
    if store
        .lookup_dict(reading, DICT_LOOKUP_LIMIT)
        .iter()
        .any(|c| c == surface)
    {
        return Some(WEIGHT_DICT);
    }
    None
}

/// 候補 1 件の辞書一致スコア。割り戻せなければ `None`。
///
/// 読み長の **二乗** で効かせるのは、同じ読みを短く刻んだ区切りに負けない
/// ようにするため。`しじぶん` は `指示文`(4²=16) と `指示`+`分`(2²+2²=8) が
/// 一致文字数では並ぶので、長い語を選ぶ力が要る（IME の最長一致に相当）。
///
/// 動詞・形容詞の語幹（`作`＝`つく`）は活用のせいで辞書に一致しないが、
/// それは全候補に等しく効くので相対比較は壊れない。
pub fn dict_agreement_score(store: &DictStore, reading: &str, surface: &str) -> Option<f64> {
    let runs = align(reading, surface)?;
    let mut total = 0.0;
    for run in runs {
        if !matches!(run.kind, RunKind::Kanji | RunKind::Katakana) {
            continue;
        }
        let n = run.reading.chars().count();
        if n < MIN_RUN_READING_CHARS {
            continue;
        }
        if let Some(w) = run_weight(store, &run.reading, &run.surface) {
            total += w * (n * n) as f64;
        }
    }
    Some(total)
}

/// n-best の中で最も辞書と整合する候補を先頭へ繰り上げる。
///
/// 繰り上げは **先頭候補より `min_gain` 以上勝っているときだけ**。LLM は
/// 文脈（直前の確定文字列）を見て並べているが辞書は見ていないので、僅差で
/// ひっくり返すと文脈で決まる曖昧さ（`きょうはいしゃに` の
/// 「今日は医者」/「今日歯医者」）まで辞書の都合で塗り替えてしまう。
///
/// 実辞書で測った差は、拾いたい取りこぼし（`指示分`→`指示文` が 32、
/// かな残り→`会議資料` が 40、`れーるがん`→`レールガン` が 50）と、
/// 触ってほしくない曖昧さ（`今日は医者`/`今日歯医者` が 18）の間に
/// 開きがある。既定 24.0 はその谷を取ったもの。
///
/// 候補の集合・件数は変えない。
pub fn promote_dict_agreeing(
    store: &DictStore,
    reading: &str,
    candidates: Vec<String>,
    min_reading_chars: usize,
    min_gain: f64,
) -> Vec<String> {
    if candidates.len() < 2 || reading.chars().count() < min_reading_chars {
        return candidates;
    }

    let scores: Vec<f64> = candidates
        .iter()
        .map(|c| dict_agreement_score(store, reading, c).unwrap_or(0.0))
        .collect();

    let top = scores[0];
    let mut best_idx = 0usize;
    let mut best = top;
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best {
            best = s;
            best_idx = i;
        }
    }

    if best_idx == 0 || best - top < min_gain {
        tracing::debug!(
            reading = %reading,
            scores = ?scores,
            "rescore: keep LLM order"
        );
        return candidates;
    }

    let mut out = candidates;
    let promoted = out.remove(best_idx);
    tracing::info!(
        reading = %reading,
        promoted = %promoted,
        from_rank = best_idx,
        score = best,
        top_score = top,
        "rescore: promoted dict-agreeing candidate"
    );
    out.insert(0, promoted);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn aligned(reading: &str, surface: &str) -> Option<Vec<(String, String)>> {
        align(reading, surface).map(|runs| {
            runs.into_iter()
                .map(|r| (r.reading, r.surface))
                .collect::<Vec<_>>()
        })
    }

    #[test]
    fn aligns_kanji_runs_between_anchors() {
        let got = aligned("しじぶんもそんなにこまかくない", "指示文もそんなに細かくない").unwrap();
        assert_eq!(
            got,
            vec![
                ("しじぶん".to_string(), "指示文".to_string()),
                ("もそんなに".to_string(), "もそんなに".to_string()),
                ("こま".to_string(), "細".to_string()),
                ("かくない".to_string(), "かくない".to_string()),
            ]
        );
    }

    #[test]
    fn aligns_katakana_run_by_hiragana_form() {
        let got = aligned("れーるがんいますぐ", "レールガン今すぐ").unwrap();
        assert_eq!(
            got,
            vec![
                ("れーるがん".to_string(), "レールガン".to_string()),
                ("いま".to_string(), "今".to_string()),
                ("すぐ".to_string(), "すぐ".to_string()),
            ]
        );
    }

    #[test]
    fn refuses_alignment_when_anchor_missing() {
        // 読みに無いひらがなを LLM が出した場合は割らない。
        assert!(aligned("とうきょうへ", "東京から").is_none());
    }

    #[test]
    fn refuses_alignment_when_surface_longer_than_reading() {
        // 予測候補は読みより長い表層を返す。スライス外を触らずに割れないと答える。
        assert!(aligned("あい", "あいうえおかきくけこ").is_none());
    }

    fn store_with_user_entries(entries: &str) -> (tempfile::TempDir, DictStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("user_dict.toml");
        fs::write(&path, entries).unwrap();
        let store = DictStore::load(Some(&path), None, None).unwrap();
        (dir, store)
    }

    #[test]
    fn promotes_candidate_matching_user_dict() {
        let (_dir, store) = store_with_user_entries(
            r#"
[[entries]]
reading = "しじぶん"
surfaces = ["指示文"]
"#,
        );
        let reading = "しじぶんもそんなにこまかくない";
        let cands = vec![
            "指示分もそんなに細かくない".to_string(),
            "指示文もそんなに細かくない".to_string(),
        ];
        let got = promote_dict_agreeing(&store, reading, cands, 12, 24.0);
        assert_eq!(got[0], "指示文もそんなに細かくない");
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn keeps_llm_order_for_short_reading() {
        let (_dir, store) = store_with_user_entries(
            r#"
[[entries]]
reading = "しじぶん"
surfaces = ["指示文"]
"#,
        );
        let cands = vec!["指示分".to_string(), "指示文".to_string()];
        // 短い読みは辞書が完全一致で効くので、この経路は触らない。
        let got = promote_dict_agreeing(&store, "しじぶん", cands.clone(), 12, 24.0);
        assert_eq!(got, cands);
    }

    #[test]
    fn keeps_llm_order_without_dict_evidence() {
        let (_dir, store) = store_with_user_entries("");
        let cands = vec![
            "あああああああああああああ".to_string(),
            "いいいいいいいいいいいい".to_string(),
        ];
        let got = promote_dict_agreeing(&store, "あああああああああああああ", cands.clone(), 12, 24.0);
        assert_eq!(got, cands);
    }

    #[test]
    fn prefers_longer_dictionary_word_over_partial_conversion() {
        let (_dir, store) = store_with_user_entries(
            r#"
[[entries]]
reading = "しじぶん"
surfaces = ["指示文"]

[[entries]]
reading = "しじ"
surfaces = ["指示"]
"#,
        );
        let reading = "しじぶんをかくにんする";
        // 後半をかなで残した候補も「指示」だけは辞書に一致するが、
        // 読み長の二乗で効かせているので語全体を当てたほうが高い。
        let partial = dict_agreement_score(&store, reading, "指示ぶんをかくにんする").unwrap();
        let whole = dict_agreement_score(&store, reading, "指示文をかくにんする").unwrap();
        assert!(whole > partial, "whole={whole} should beat partial={partial}");
    }
}
