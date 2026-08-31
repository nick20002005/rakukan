//! 入力中の予測ウィンドウ（Google 日本語入力の「予測候補」相当）。
//!
//! Space を押す前、打鍵している最中に、学習済みフレーズを読みの前方一致で
//! 候補ウィンドウに出す。ここでの表示は**あくまで表示だけ**で `SessionState` は
//! 変えない（打鍵はそのまま続けられる）。↓ / Tab が押された時点で初めて
//! `Selecting` に遷移する（`edit_ops::on_candidate_move` / `on_candidate_page`）。
//!
//! 予測は学習履歴の HashMap 前方一致走査だけなので、LLM も MOZC 辞書も引かない。
//! 打鍵ごとに 1 RPC 増えるが、ライブ変換の BG 起動より桁違いに軽い。

use std::sync::Mutex;

use crate::engine::state::{DynEngine, caret_rect_get};
use crate::tsf::candidate_window;

/// 現在表示中の予測。`reading` は表示時点の読みで、↓ を押した時に
/// 「今の読みと一致するか」を確かめるために持つ（打鍵とキー処理の間で
/// 読みがずれていたら開かない）。
struct Shown {
    reading: String,
    items: Vec<String>,
}

static SHOWN: Mutex<Option<Shown>> = Mutex::new(None);

/// 予測候補を取り出す（エンジンが要る側）。`show` と分けているのは、
/// キャレット位置が `update_composition` の後でないと確定しないため。
pub(crate) fn fetch(engine: &DynEngine, reading: &str) -> Vec<String> {
    let cfg = crate::engine::config::current_config();
    if !cfg.prediction.enabled || !cfg.prediction.suggest_while_typing {
        return vec![];
    }
    if reading.is_empty() || reading.chars().count() < cfg.prediction.min_reading_chars {
        return vec![];
    }
    let limit = cfg.prediction.suggest_max_candidates.clamp(1, 9);
    engine
        .predict(reading, limit)
        .into_iter()
        .filter(|s| !s.is_empty() && s != reading)
        .collect()
}

/// 予測ウィンドウを出す（候補が空なら閉じる）。`update_composition` の後に呼ぶこと。
///
/// 選択行は出さない（`page_selected` にリスト外の添字を渡す）。ハイライトが
/// 無いことで「まだ確定に関与していない」ことを示す。
pub(crate) fn show(reading: &str, items: Vec<String>) {
    if items.is_empty() {
        clear();
        return;
    }
    let caret = caret_rect_get();
    candidate_window::show_suggestion(&items, caret.left, caret.bottom, Some("Tab/↓ で予測候補"));
    if let Ok(mut g) = SHOWN.lock() {
        *g = Some(Shown {
            reading: reading.to_string(),
            items,
        });
    }
}

/// 予測ウィンドウを閉じる。表示していなかった場合は何もしない
/// （変換候補リストを出している最中に誤って閉じないため）。
pub(crate) fn clear() {
    let was_shown = match SHOWN.lock() {
        Ok(mut g) => g.take().is_some(),
        Err(_) => false,
    };
    if was_shown {
        // hide() 側でも forget_shown() を呼ぶが、既に take 済みなので no-op。
        candidate_window::hide();
    }
}

/// 表示状態だけ捨てる（ウィンドウは触らない）。確定・キャンセル経路のように
/// 呼び出し側が既にウィンドウを閉じている場合に使う。
pub(crate) fn forget_shown() {
    if let Ok(mut g) = SHOWN.lock() {
        *g = None;
    }
}

/// `reading` に対して表示中の予測候補を取り出して表示状態を捨てる。
/// ↓ / Tab で候補リストに入る時に使う。
pub(crate) fn take_for(reading: &str) -> Option<Vec<String>> {
    let mut g = SHOWN.lock().ok()?;
    match g.as_ref() {
        Some(s) if s.reading == reading && !s.items.is_empty() => g.take().map(|s| s.items),
        _ => None,
    }
}
