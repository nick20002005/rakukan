//! ユーザー辞書語の前方一致候補
//!
//! ユーザー辞書は読みの**完全一致**でしか引かれないため、登録した語が文中に
//! 現れると候補にすら入らない。`とらぶる` 単独なら `To LOVEる` が出るのに、
//! `とらぶると` と打った瞬間に別の読みになって LLM の `トラブルと` しか残らない
//! （2026-09-01 に実害）。助詞が付くたびに登録するのは現実的ではない。
//!
//! ここでは読みの先頭がユーザー辞書の語に前方一致したとき、
//! **`語 + 残りを変換したもの`** を候補に足す（`To LOVEる` ＋ `と` →
//! `To LOVEると`）。
//!
//! # 誤爆を抑えるための制約
//! ユーザー辞書は数千件あるので、素直に前方一致させると短い語が大量に誤爆する
//! （`みき → 美樹` が登録済みだと `みきわめる` に `美樹わめる` が湧く）。
//!
//! - 一致する語は **[`MIN_PREFIX_CHARS`] 文字以上**に限る
//! - 一致は**最長のものだけ**を使う
//! - 候補は先頭に置かない（[`INSERT_AT`]）。先頭候補はライブ変換の preview に
//!   そのまま出るため、打鍵途中に誤爆した表記が見え続けることになる
//! - 完全一致は既存の経路（`merge_candidates_for_reading`）の担当なので除く

use crate::kanji::KanaKanjiConverter;
use crate::DigitCandidateKind;

/// 前方一致の対象にするユーザー辞書語の最小文字数。
///
/// 2 文字まで許すと `みき` `かな` のような短い登録語が一般語の先頭に噛んで
/// 誤爆が実用にならない量になる。
pub const MIN_PREFIX_CHARS: usize = 3;

/// 候補リストのどこに差し込むか（0 始まり）。
///
/// 0 にするとライブ変換の preview を奪う。末尾だと 1 ページ目に出ないことがある。
const INSERT_AT: usize = 2;

/// 残りの読みから何件まで組み合わせるか。
const MAX_REMAINDER_CANDIDATES: usize = 2;

/// `語 + 残りの変換` を作って `out` に差し込む。
///
/// `split` は (ユーザー辞書の表記, 残りの読み)。残りの変換に失敗した場合は
/// 何もしない（候補を減らさない）。
#[allow(clippy::too_many_arguments)]
pub fn insert_candidates(
    converter: &KanaKanjiConverter,
    split: &(String, String),
    context: &str,
    digit_candidates_order: &[DigitCandidateKind],
    alpha_fullwidth_first: bool,
    symbol_fullwidth_first: bool,
    out: &mut Vec<String>,
) {
    let (surface, remainder) = split;
    if surface.is_empty() || remainder.is_empty() {
        return;
    }
    // 残りの変換には「確定済み ＋ 辞書語」を文脈として渡す。
    // 「と」だけを裸で変換すると助詞が漢字（外）になりやすい。
    let local_context = format!("{context}{surface}");
    let rest = match crate::digits::convert_with_digit_protection(
        converter,
        remainder,
        &local_context,
        MAX_REMAINDER_CANDIDATES,
        digit_candidates_order,
        alpha_fullwidth_first,
        symbol_fullwidth_first,
    ) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("dict_prefix: 残りの変換に失敗 {remainder:?}: {e}");
            return;
        }
    };

    let mut built: Vec<String> = Vec::new();
    for r in rest.into_iter().take(MAX_REMAINDER_CANDIDATES) {
        let combined = format!("{surface}{r}");
        if !out.contains(&combined) && !built.contains(&combined) {
            built.push(combined);
        }
    }
    if built.is_empty() {
        return;
    }
    tracing::info!(
        "dict_prefix: {:?} + {:?} → {:?}",
        surface,
        remainder,
        built
    );
    let at = INSERT_AT.min(out.len());
    for (i, c) in built.into_iter().enumerate() {
        out.insert(at + i, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_at_keeps_top_candidate() {
        let mut out = vec!["トラブルと".to_string(), "とらぶると".to_string()];
        // convert を通さず差し込み位置だけを確かめる
        let at = INSERT_AT.min(out.len());
        out.insert(at, "To LOVEると".to_string());
        assert_eq!(out[0], "トラブルと");
        assert_eq!(out[2], "To LOVEると");
    }
}
