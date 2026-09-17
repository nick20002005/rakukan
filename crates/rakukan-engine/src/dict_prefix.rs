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
//! - 候補は原則先頭に置かない（[`INSERT_AT`]）。先頭候補はライブ変換の preview に
//!   そのまま出るため、打鍵途中に誤爆した表記が見え続けることになる。
//!   残りが助詞・接尾辞のとき（[`TOP_INSERT_REMAINDERS`]）だけは先頭に置く
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

/// この「残りの読み」なら候補を**先頭**に置く。
///
/// [`INSERT_AT`] は打鍵途中の誤爆が preview に居座るのを避けるための位置だが、
/// 副作用として「登録語＋助詞」が常に 3 番目に沈む。`なかたにいく → 中谷育` を
/// 登録していても `なかたにいくが` の preview は LLM の `な方にいくが` のままで、
/// 一度手で選んで学習させるまで直らない（2026-09-06 に実害）。
///
/// 登録語の直後が助詞・接尾辞なら「登録語＋それ」以外の読み方はまず無いので、
/// preview を渡してよい。ここに載っていない残りは従来どおり [`INSERT_AT`] に置く
/// ——`はんぷ → 頒布` を登録した状態の `はんぷく`（反復）の `く` のように、
/// 残りを足すと語の途中でしかない例があるため。
const TOP_INSERT_REMAINDERS: &[&str] = &[
    // 助詞
    "が", "を", "は", "に", "へ", "と", "も", "の", "や", "か", "ね", "よ", "で", "から", "まで",
    "より", "では", "には", "とは", "にも", "でも", "との", "への", "なら", "って",
    // 接尾辞（人名の後ろに付きやすいもの）
    "さん", "くん", "ちゃん", "さま", "たち",
];

/// 残りの読みから何件まで組み合わせるか。
const MAX_REMAINDER_CANDIDATES: usize = 2;

/// 差し込み位置を決める。残りが助詞・接尾辞なら先頭、それ以外は [`INSERT_AT`]。
fn insert_position(remainder: &str, out_len: usize) -> usize {
    if TOP_INSERT_REMAINDERS.contains(&remainder) {
        0
    } else {
        INSERT_AT.min(out_len)
    }
}

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
    let at = insert_position(remainder, out.len());
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
        let at = insert_position("わめる", out.len());
        out.insert(at, "美樹わめる".to_string());
        assert_eq!(out[0], "トラブルと");
        assert_eq!(out[2], "美樹わめる");
    }

    #[test]
    fn particle_remainder_takes_the_top_slot() {
        // 「なかたにいくが」の preview が LLM の誤変換のままにならないこと
        let mut out = vec!["な方にいくが".to_string(), "な方に行くが".to_string()];
        let at = insert_position("が", out.len());
        out.insert(at, "中谷育が".to_string());
        assert_eq!(out[0], "中谷育が");
    }

    #[test]
    fn non_particle_remainder_stays_below_the_top() {
        // 「はんぷ → 頒布」を登録した状態の「はんぷく」（反復）で preview を奪わない
        assert_eq!(insert_position("く", 5), INSERT_AT);
        assert_eq!(insert_position("なかで", 5), INSERT_AT);
    }

    #[test]
    fn insert_position_clamps_to_short_lists() {
        assert_eq!(insert_position("わめる", 1), 1);
        assert_eq!(insert_position("わめる", 0), 0);
    }
}
