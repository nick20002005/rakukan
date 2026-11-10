//! 入力欄の種類（TSF の InputScope）に応じて IME をオフで始める。
//!
//! ブラウザのアドレスバー（Chrome / Edge は `IS_URL` を返す）・メールアドレス欄・
//! パスワード欄のように、英数でしか打たない入力欄にフォーカスが入ったら
//! IME を直接入力にする（`input.ime_off_in_url_fields`）。Mozc・MS-IME と同じ挙動。
//!
//! 入ったときに 1 回オフにするだけで、その後ユーザーがオンにすればそのまま続く。
//!
//! InputScope は `ITfContext::GetAppProperty(GUID_PROP_INPUTSCOPE)` の値で、
//! 読むには edit cookie が要る。フォーカス変化の遅延処理（msctf コールバック外）
//! から読み取り専用の EditSession を投げて読む。

use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::UI::TextServices::TF_CONTEXT_EDIT_CONTEXT_FLAGS;
use windows::Win32::UI::TextServices::{
    GUID_PROP_INPUTSCOPE, IS_EMAIL_SMTPEMAILADDRESS, IS_EMAIL_USERNAME, IS_PASSWORD, IS_URL,
    ITfContext, ITfInputScope, ITfRange, ITfThreadMgr, InputScope, TF_DEFAULT_SELECTION,
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

/// 入力欄の種類の一覧に、オフで始めるべきものが含まれるか。
pub fn should_force_off(scopes: &[InputScope]) -> bool {
    scopes.iter().any(|s| OFF_SCOPES.contains(s))
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
            "input_scope: dm={dm_ptr:#x} scopes={:?}",
            scope_ids(&scopes)
        );
        if should_force_off(&scopes) {
            let from =
                crate::tsf::ime_sync::apply(Some(&tm), tid, ImeMode::Off, true, "input_scope");
            if from.is_some_and(|f| f != ImeMode::Off) {
                tracing::info!(
                    "input_scope: dm={dm_ptr:#x} scopes={:?} → Off",
                    scope_ids(&scopes)
                );
            }
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
        assert!(should_force_off(&[IS_URL]));
        assert!(should_force_off(&[IS_EMAIL_SMTPEMAILADDRESS]));
        assert!(should_force_off(&[IS_EMAIL_USERNAME]));
        assert!(should_force_off(&[IS_PASSWORD]));
        // 複数指定のどれかに含まれていればオフ
        assert!(should_force_off(&[IS_DEFAULT, IS_URL]));
    }

    #[test]
    fn ordinary_fields_keep_mode() {
        assert!(!should_force_off(&[]));
        assert!(!should_force_off(&[IS_DEFAULT]));
        assert!(!should_force_off(&[IS_TEXT]));
        assert!(!should_force_off(&[IS_HIRAGANA]));
        // 数字欄は対象外（全角数字を打ちたい人もいるため触らない）
        assert!(!should_force_off(&[IS_NUMBER]));
    }
}
