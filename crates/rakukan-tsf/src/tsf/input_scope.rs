//! 入力欄の種類（TSF の InputScope）に応じて IME をオフで始める。
//!
//! ブラウザのアドレスバー・URL 欄・メールアドレス欄・パスワード欄のように、
//! 英数で打つことが多い入力欄にフォーカスが入ったら IME を直接入力にする
//! （`input.ime_off_in_url_fields`）。
//!
//! 🔴 Chrome / Edge のアドレスバーは `IS_URL` ではなく `IS_SEARCH` を返す
//! （日本語などの IME 環境では検索語を打ちやすいよう、Chromium が意図的に
//! SEARCH にしている。2026-11-10 に Edge で実測）。ページ内の
//! `<input type=search>` も同じ `IS_SEARCH`（しかも同じ DocumentManager）なので、
//! 種類だけでは区別できない。違いは Win32 のフォーカス窓で、ページ内の入力欄は
//! 描画用の子窓 `Chrome_RenderWidgetHostHWND`、アドレスバーはブラウザ本体の
//! `Chrome_WidgetWin_1` にフォーカスがある。`IS_SEARCH` は後者のときだけオフにする。
//!
//! 入ったときに 1 回オフにするだけで、その後ユーザーがオンにすればそのまま続く。
//!
//! InputScope は `ITfContext::GetAppProperty(GUID_PROP_INPUTSCOPE)` の値で、
//! 読むには edit cookie が要る。フォーカス変化の遅延処理（msctf コールバック外）
//! から読み取り専用の EditSession を投げて読む。

use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::UI::TextServices::TF_CONTEXT_EDIT_CONTEXT_FLAGS;
use windows::Win32::UI::TextServices::{
    GUID_PROP_INPUTSCOPE, IS_EMAIL_SMTPEMAILADDRESS, IS_EMAIL_USERNAME, IS_PASSWORD, IS_SEARCH,
    IS_URL, ITfContext, ITfInputScope, ITfRange, ITfThreadMgr, InputScope, TF_DEFAULT_SELECTION,
    TF_ES_ASYNCDONTCARE, TF_ES_READ, TF_SELECTION,
};
use windows::core::{IUnknown, Interface};

use crate::engine::ime_mode::ImeMode;

/// IME をオフで始める入力欄の種類。
const OFF_SCOPES: [InputScope; 4] = [
    IS_URL,
    IS_EMAIL_SMTPEMAILADDRESS,
    IS_EMAIL_USERNAME,
    IS_PASSWORD,
];

/// Chromium 系ブラウザ本体（ページの外＝アドレスバー等の UI）の窓クラス。
const CHROMIUM_UI_CLASS: &str = "Chrome_WidgetWin_1";

/// 入力欄の種類の一覧に、オフで始めるべきものが含まれるか。
///
/// `in_browser_ui`: Win32 のフォーカス窓が Chromium 系ブラウザ本体の窓か
/// （ページ内の入力欄ではないか）。`IS_SEARCH` はこのときだけ対象にする。
pub fn should_force_off(scopes: &[InputScope], in_browser_ui: bool) -> bool {
    scopes
        .iter()
        .any(|s| OFF_SCOPES.contains(s) || (in_browser_ui && *s == IS_SEARCH))
}

/// このスレッドの Win32 フォーカス窓が Chromium 系ブラウザ本体の窓か。
fn focus_in_browser_ui() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetFocus;
    class_of(unsafe { GetFocus() }) == CHROMIUM_UI_CLASS
}

fn class_of(hwnd: windows::Win32::Foundation::HWND) -> String {
    use windows::Win32::UI::WindowsAndMessaging::GetClassNameW;
    if hwnd.0.is_null() {
        return String::new();
    }
    let mut buf = [0u16; 64];
    let n = unsafe { GetClassNameW(hwnd, &mut buf) };
    if n <= 0 {
        return String::new();
    }
    String::from_utf16_lossy(&buf[..n as usize])
}

fn view_class(ctx: &ITfContext) -> String {
    unsafe {
        ctx.GetActiveView()
            .and_then(|v| v.GetWnd())
            .map(class_of)
            .unwrap_or_default()
    }
}

/// フォーカスが `dm_ptr` の DocumentManager に移った直後に呼ぶ。
///
/// 入力欄の種類を読み、オフで始めるべき欄なら IME をオフにする。
/// 読めなかった場合（InputScope を出さないアプリ・編集権が取れない等）は何もしない。
pub fn apply_on_focus(tm: Option<&ITfThreadMgr>, tid: u32, dm_ptr: usize) {
    if !crate::engine::config::current_config()
        .input
        .ime_off_in_url_fields
    {
        return;
    }
    let Some(tm) = tm.cloned() else {
        return;
    };
    if tid == 0 || dm_ptr == 0 {
        return;
    }
    let Ok(dm) = (unsafe { tm.GetFocus() }) else {
        return;
    };
    if dm.as_raw() as usize != dm_ptr {
        // 遅延処理の間にフォーカスがさらに動いた。次のイベントで処理される。
        return;
    }
    let Ok(ctx) = (unsafe { dm.GetTop() }) else {
        return;
    };

    let ctx_in = ctx.clone();
    let session = crate::tsf::edit_session::EditSession::new(move |ec| {
        // 非同期で後から走った場合、フォーカスが別の入力欄へ移っていたら何もしない。
        let still_focused = unsafe { tm.GetFocus() }
            .map(|d| d.as_raw() as usize == dm_ptr)
            .unwrap_or(false);
        if !still_focused {
            return Ok(());
        }
        let scopes = read_input_scopes(&ctx_in, ec);
        tracing::debug!(
            "input_scope: dm={dm_ptr:#x} scopes={:?} focus={} view={}",
            scope_ids(&scopes),
            class_of(unsafe { windows::Win32::UI::Input::KeyboardAndMouse::GetFocus() }),
            view_class(&ctx_in)
        );
        if should_force_off(&scopes, false) {
            force_off(&tm, tid, dm_ptr, "input_scope");
        } else if scopes.contains(&IS_SEARCH) {
            // アドレスバーかページ内の検索欄かは Win32 のフォーカス窓で見分けるが、
            // アドレスバー → ページへ移った直後は TSF のフォーカス通知が先に来て
            // Win32 のフォーカスがまだブラウザ本体に残っている（2026-11-10 実測）。
            // 少し待って落ち着いてから判定する。
            schedule_search_check(tm.clone(), tid, dm_ptr);
        }
        Ok(())
    });

    let flags = TF_CONTEXT_EDIT_CONTEXT_FLAGS(TF_ES_READ.0 | TF_ES_ASYNCDONTCARE.0);
    match unsafe { ctx.RequestEditSession(tid, &session, flags) } {
        Ok(hr) if hr.is_err() => {
            tracing::debug!("input_scope: edit session refused hr={hr:?}");
        }
        Err(e) => tracing::debug!("input_scope: RequestEditSession failed: {e}"),
        _ => {}
    }
}

fn force_off(tm: &ITfThreadMgr, tid: u32, dm_ptr: usize, source: &'static str) {
    let from = crate::tsf::ime_sync::apply(Some(tm), tid, ImeMode::Off, true, source);
    if from.is_some_and(|f| f != ImeMode::Off) {
        tracing::info!("{source}: dm={dm_ptr:#x} → Off");
    }
}

/// `IS_SEARCH` の判定をこの時間だけ遅らせる。Win32 のフォーカス移動が追いつくのを待つ。
const SEARCH_CHECK_DELAY_MS: u32 = 60;

struct PendingSearchCheck {
    tm: ITfThreadMgr,
    tid: u32,
    dm_ptr: usize,
    timer_id: usize,
}

thread_local! {
    static TL_PENDING_SEARCH: std::cell::RefCell<Option<PendingSearchCheck>> =
        const { std::cell::RefCell::new(None) };
}

fn schedule_search_check(tm: ITfThreadMgr, tid: u32, dm_ptr: usize) {
    use windows::Win32::UI::WindowsAndMessaging::{KillTimer, SetTimer};
    // 前の予約は捨てる（フォーカスが動いたので古い判定は要らない）
    if let Some(old) = TL_PENDING_SEARCH.with(|c| c.borrow_mut().take()) {
        unsafe {
            let _ = KillTimer(None, old.timer_id);
        }
    }
    let timer_id = unsafe { SetTimer(None, 0, SEARCH_CHECK_DELAY_MS, Some(on_search_check_timer)) };
    if timer_id == 0 {
        tracing::debug!("input_scope: SetTimer failed, search check skipped");
        return;
    }
    TL_PENDING_SEARCH.with(|c| {
        *c.borrow_mut() = Some(PendingSearchCheck {
            tm,
            tid,
            dm_ptr,
            timer_id,
        })
    });
}

unsafe extern "system" fn on_search_check_timer(
    _hwnd: windows::Win32::Foundation::HWND,
    _msg: u32,
    id: usize,
    _time: u32,
) {
    use windows::Win32::UI::WindowsAndMessaging::KillTimer;
    unsafe {
        let _ = KillTimer(None, id);
    }
    let Some(p) = TL_PENDING_SEARCH.with(|c| {
        let mut g = c.borrow_mut();
        if g.as_ref().is_some_and(|p| p.timer_id == id) {
            g.take()
        } else {
            None
        }
    }) else {
        return;
    };
    let still_focused = unsafe { p.tm.GetFocus() }
        .map(|d| d.as_raw() as usize == p.dm_ptr)
        .unwrap_or(false);
    let in_ui = focus_in_browser_ui();
    tracing::debug!(
        "input_scope: search check dm={:#x} still_focused={still_focused} browser_ui={in_ui}",
        p.dm_ptr
    );
    if still_focused && in_ui {
        force_off(&p.tm, p.tid, p.dm_ptr, "input_scope(search)");
    }
}

fn scope_ids(scopes: &[InputScope]) -> Vec<i32> {
    scopes.iter().map(|s| s.0).collect()
}

/// 現在の選択位置（無ければ文書の先頭）にかかっている InputScope の一覧を読む。
fn read_input_scopes(ctx: &ITfContext, ec: u32) -> Vec<InputScope> {
    unsafe {
        let Ok(prop) = ctx.GetAppProperty(&GUID_PROP_INPUTSCOPE) else {
            return Vec::new();
        };
        let Some(range) = selection_range(ctx, ec).or_else(|| ctx.GetStart(ec).ok()) else {
            return Vec::new();
        };
        let Ok(var) = prop.GetValue(ec, &range) else {
            return Vec::new();
        };
        let Ok(unk) = IUnknown::try_from(&var) else {
            // VT_EMPTY（InputScope を設定していない入力欄）
            return Vec::new();
        };
        let Ok(is) = unk.cast::<ITfInputScope>() else {
            return Vec::new();
        };
        let mut ptr: *mut InputScope = std::ptr::null_mut();
        let mut count: u32 = 0;
        if is.GetInputScopes(&mut ptr, &mut count).is_err() || ptr.is_null() {
            return Vec::new();
        }
        let out = std::slice::from_raw_parts(ptr, count as usize).to_vec();
        CoTaskMemFree(Some(ptr as *const _));
        out
    }
}

fn selection_range(ctx: &ITfContext, ec: u32) -> Option<ITfRange> {
    let mut sel = [TF_SELECTION::default()];
    let mut fetched: u32 = 0;
    unsafe {
        ctx.GetSelection(ec, TF_DEFAULT_SELECTION, &mut sel, &mut fetched)
            .ok()?;
    }
    if fetched == 0 {
        return None;
    }
    // TF_SELECTION.range は ManuallyDrop<Option<ITfRange>>。取り出して所有権を移す。
    std::mem::ManuallyDrop::into_inner(std::mem::take(&mut sel[0].range))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::TextServices::{IS_DEFAULT, IS_HIRAGANA, IS_NUMBER, IS_TEXT};

    #[test]
    fn url_email_password_force_off() {
        for in_ui in [false, true] {
            assert!(should_force_off(&[IS_URL], in_ui));
            assert!(should_force_off(&[IS_EMAIL_SMTPEMAILADDRESS], in_ui));
            assert!(should_force_off(&[IS_EMAIL_USERNAME], in_ui));
            assert!(should_force_off(&[IS_PASSWORD], in_ui));
            // 複数指定のどれかに含まれていればオフ
            assert!(should_force_off(&[IS_DEFAULT, IS_URL], in_ui));
        }
    }

    #[test]
    fn search_is_off_only_in_browser_ui() {
        // アドレスバー（ブラウザ本体の窓）
        assert!(should_force_off(&[IS_SEARCH], true));
        // ページ内の <input type=search> は日本語で検索したいので触らない
        assert!(!should_force_off(&[IS_SEARCH], false));
    }

    #[test]
    fn ordinary_fields_keep_mode() {
        for in_ui in [false, true] {
            assert!(!should_force_off(&[], in_ui));
            assert!(!should_force_off(&[IS_DEFAULT], in_ui));
            assert!(!should_force_off(&[IS_TEXT], in_ui));
            assert!(!should_force_off(&[IS_HIRAGANA], in_ui));
            // 数字欄は対象外（全角数字を打ちたい人もいるため触らない）
            assert!(!should_force_off(&[IS_NUMBER], in_ui));
        }
    }
}
