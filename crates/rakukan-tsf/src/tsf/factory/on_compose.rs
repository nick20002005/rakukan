//! composition 操作の EditSession ヘルパー集約。
//!
//! 旧 factory.rs から M3 (T1-A) で純粋切り出し。動作変更なし、関数本体は完全に
//! 同一。可視性は `pub(super)` に揃え、factory.rs から `use on_compose::*;` で
//! 引き込む。
//!
//! 含まれる関数:
//! - `update_composition` / `update_composition_candidate_parts` / `update_caret_rect`
//! - `commit_then_start_composition` / `end_composition` / `commit_text`
//! - キャレット / range 取得ヘルパー (`get_caret_pos_from_context` / `get_cursor_range` /
//!   `get_insert_range_or_end` / `get_document_end_range`)
//! - 表示属性ヘルパー (`set_display_attr_prop`)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use windows::Win32::UI::TextServices::{
    GUID_PROP_ATTRIBUTE, ITfCompositionSink, ITfContext, ITfEditSession,
    TF_CONTEXT_EDIT_CONTEXT_FLAGS, TF_ES_ASYNC, TF_ES_READWRITE,
};
use windows::core::Interface;

use crate::engine::state::{
    caret_rect_set, composition_clone, composition_set_with_dm, composition_take,
};
use crate::tsf::display_attr;
use crate::tsf::edit_session::EditSession;

// ─── 確定 EditSession の投入とリトライ ────────────────────────────────────────
//
// `RequestEditSession` の失敗は **out の hr** で返る（Rust 側の Result は呼び出し
// そのものの失敗しか表さない）。ここを見ないと DoEditSession が一度も走らなかった
// 場合でも成功扱いになり、確定テキストが 1 文字も書かれないまま消える。呼び元は
// end_composition より前に engine.commit() / reset_preedit() を済ませているため、
// 失われたテキストはどこにも残らない。
//
// 実測 (rakukan.log.1 2026-09-13T14:06:16): CommitRaw の直後に
// hr=TS_E_READONLY(0x80040209) で edit session が走らず、確定済みの
// 「同人パブリシャーを書き換える」が消えて打ち直しになった。同型の穴が
// commit_then_start_composition / commit_text にもあり、そちらは hr を捨てて
// いたためログにすら残っていなかった。

/// 確定テキストを書き直す最大試行回数（初回を含む）。
const COMMIT_RETRY_MAX: u32 = 3;

/// 確定 EditSession の投入結果。
enum CommitDispatch {
    /// DoEditSession が走った（中身の成否はセッション内でログ済み）。
    Executed,
    /// TSF は受け付けたが実行は後（TF_ES_ASYNC）。実行の有無は `ran` で確かめる。
    Deferred,
    /// sync / async とも拒否された。テキストはまだどこにも書かれていない。
    Rejected,
}

/// 書き込めなかった確定テキスト。次の打鍵とフォーカス喪失で書き直す。
struct PendingCommit {
    text: String,
    /// 失敗時の DocumentMgr。別の入力欄へ書き込まないための照合用。
    dm_ptr: usize,
    /// 書き込みの占有券。最初に走った DoEditSession がこれを立て、後から走った
    /// セッションは何もせずに戻る。TF_ES_ASYNC で積まれたセッションと書き直しの
    /// セッションが両方走っても二重に書かないための唯一の防壁。
    ran: Arc<AtomicBool>,
    attempt: u32,
}

static PENDING_COMMIT: Mutex<Option<PendingCommit>> = Mutex::new(None);

/// TSF DLL はアプリのプロセス内で動くので current_exe = 発生アプリ。
fn current_app_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default()
}

/// 拒否の理由切り分け用（TS_SD_READONLY=0x1 / TS_SD_LOADING=0x2）。
fn doc_status(ctx: &ITfContext) -> String {
    match unsafe { ctx.GetStatus() } {
        Ok(st) => format!(
            "dyn=0x{:x} static=0x{:x}",
            st.dwDynamicFlags, st.dwStaticFlags
        ),
        Err(e) => format!("GetStatus failed: {e}"),
    }
}

/// 確定系の EditSession を投入する。sync で拒否されたら TF_ES_ASYNC で投げ直す。
///
/// `EditSession` のクロージャは `DoEditSession` の中で `take()` されるため、
/// 拒否されたセッションはそのまま再投入してよい（二重実行にはならない）。
/// TF_ES_ASYNC で積んだセッションが後から走るケースは、`ran` を占有券として
/// 使う各セッション先頭の `swap` で弾く。
fn request_commit_session(
    ctx: &ITfContext,
    tid: u32,
    session: &ITfEditSession,
    ran: &Arc<AtomicBool>,
    site: &str,
    text: &str,
) -> CommitDispatch {
    match unsafe { ctx.RequestEditSession(tid, session, TF_ES_READWRITE) } {
        Ok(hr) if hr.is_ok() => return CommitDispatch::Executed,
        Ok(hr) => {
            if ran.load(Ordering::SeqCst) {
                // セッションは走った上でクロージャが Err を返した。再投入すると
                // 二重に書く可能性があるのでここで止める。
                tracing::warn!(
                    "{site}: edit session ran but returned an error hr={hr:?} text={text:?} app={} status={}",
                    current_app_name(),
                    doc_status(ctx)
                );
                return CommitDispatch::Executed;
            }
            tracing::warn!(
                "{site}: edit session not granted hr={hr:?} text={text:?} app={} status={} → retry async",
                current_app_name(),
                doc_status(ctx)
            );
        }
        Err(e) => {
            tracing::warn!(
                "{site}: RequestEditSession failed: {e} text={text:?} app={} status={} → retry async",
                current_app_name(),
                doc_status(ctx)
            );
        }
    }

    let async_flags = TF_CONTEXT_EDIT_CONTEXT_FLAGS(TF_ES_READWRITE.0 | TF_ES_ASYNC.0);
    match unsafe { ctx.RequestEditSession(tid, session, async_flags) } {
        Ok(hr) if hr.is_ok() => {
            if ran.load(Ordering::SeqCst) {
                CommitDispatch::Executed
            } else {
                // TS_S_ASYNC。実際に走るかは保証されないので pending にも積んでおく。
                tracing::info!("{site}: queued as async edit session text={text:?}");
                CommitDispatch::Deferred
            }
        }
        Ok(hr) => {
            tracing::warn!("{site}: async edit session rejected hr={hr:?} text={text:?}");
            CommitDispatch::Rejected
        }
        Err(e) => {
            tracing::warn!("{site}: async RequestEditSession failed: {e} text={text:?}");
            CommitDispatch::Rejected
        }
    }
}

/// 書き込めなかった確定テキストを控える。`retry_pending_commit` が書き直す。
fn pending_commit_push(
    ctx: &ITfContext,
    text: String,
    ran: Arc<AtomicBool>,
    attempt: u32,
    site: &str,
) {
    if text.is_empty() {
        return;
    }
    let dm_ptr = unsafe { ctx.GetDocumentMgr() }
        .ok()
        .map(|dm| dm.as_raw() as usize)
        .unwrap_or(0);
    tracing::warn!("pending_commit: queued site={site} attempt={attempt} text={text:?}");
    let mut slot = PENDING_COMMIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(prev) = slot.replace(PendingCommit {
        text,
        dm_ptr,
        ran,
        attempt,
    }) && !prev.ran.load(Ordering::SeqCst)
    {
        tracing::error!(
            "pending_commit: dropped an earlier pending commit text={:?}",
            prev.text
        );
    }
}

/// 取りこぼした確定テキストを書き直す。
///
/// 呼ぶのは「次の打鍵の入口」と「スレッドフォーカス喪失」の 2 箇所。打鍵の入口で
/// 先に流し切ることで、そのキーの `update_composition` が composition を書き換える
/// 前に確定が済む（composition は失敗時に take されていないのでまだ生きている）。
pub(super) fn retry_pending_commit(trigger: &str) {
    let entry = {
        let mut slot = PENDING_COMMIT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.take()
    };
    let Some(entry) = entry else {
        return;
    };
    if entry.ran.load(Ordering::SeqCst) {
        tracing::debug!(
            "pending_commit: already applied, dropping text={:?}",
            entry.text
        );
        return;
    }
    let attempt = entry.attempt + 1;
    if attempt >= COMMIT_RETRY_MAX {
        tracing::error!(
            "pending_commit: gave up after {attempt} attempts, text lost text={:?}",
            entry.text
        );
        return;
    }
    let (ctx, tid, dm_ptr) = crate::tsf::live_session::last_input_context();
    let Some(ctx) = ctx else {
        tracing::error!(
            "pending_commit: no context to retry into, text lost text={:?}",
            entry.text
        );
        return;
    };
    if entry.dm_ptr != 0 && dm_ptr != entry.dm_ptr {
        tracing::error!(
            "pending_commit: document changed (0x{:x} → 0x{:x}), not retrying text={:?}",
            entry.dm_ptr,
            dm_ptr,
            entry.text
        );
        return;
    }
    tracing::warn!(
        "pending_commit: retrying trigger={trigger} attempt={attempt} text={:?}",
        entry.text
    );
    let _ = end_composition_attempt(ctx, tid, entry.text, attempt, entry.ran);
}

/// TSF コンテキストからキャレットのスクリーン座標 (x, y_bottom) を取得する。
/// mozc の FillCharPosition と同じアプローチ: GetSelection → GetTextExt。
/// 取得できない場合は None を返す（インジケーターは表示しない）。
pub(super) unsafe fn get_caret_pos_from_context(
    ctx: &windows::Win32::UI::TextServices::ITfContext,
    ec: u32,
) -> Option<(i32, i32)> {
    let range = unsafe { get_cursor_range(ctx, ec) }?;
    let view = unsafe { ctx.GetActiveView() }.ok()?;
    let mut rect = windows::Win32::Foundation::RECT::default();
    let mut clipped = windows::Win32::Foundation::BOOL(0);
    unsafe {
        view.GetTextExt(ec, &range, &mut rect, &mut clipped).ok()?;
    }
    // rect はスクリーン座標。left = x, bottom = キャレット下端。
    Some((rect.left, rect.bottom))
}

/// 現在のキャレット位置を表す長さ0の ITfRange を返す。
/// GetSelection で現在選択範囲を取得し、終端アンカーに collapse する。
/// 失敗時は None（呼び元が GetEnd にフォールバックする）。
pub(super) unsafe fn get_cursor_range(
    ctx: &windows::Win32::UI::TextServices::ITfContext,
    ec: u32,
) -> Option<windows::Win32::UI::TextServices::ITfRange> {
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::UI::TextServices::{
        TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
    };

    // windows-rs 0.58: GetSelection(ec, ulIndex, pSelection: &mut [TF_SELECTION]) -> *mut u32
    // TF_DEFAULT_SELECTION = 0xFFFF_FFFF
    let mut sel_buf = [TF_SELECTION {
        range: std::mem::ManuallyDrop::new(None),
        style: TF_SELECTIONSTYLE {
            ase: TfActiveSelEnd(0),
            fInterimChar: BOOL(0),
        },
    }];
    let mut fetched: u32 = 0;
    unsafe {
        ctx.GetSelection(ec, 0xFFFF_FFFF_u32, &mut sel_buf, &mut fetched as *mut u32)
            .ok()?;
    }
    if fetched == 0 {
        return None;
    }
    let range_ref = (*sel_buf[0].range).as_ref()?;
    let cloned = unsafe { range_ref.Clone() }.ok()?;
    if let Err(e) = unsafe { cloned.Collapse(ec, TF_ANCHOR_END) } {
        tracing::warn!("get_cursor_range: Collapse failed: {e}, range may not be zero-length");
    }
    Some(cloned)
}

/// 現在カーソル位置の range を優先し、取得できなければ `GetEnd` にフォールバックする。
///
/// TSF/COM が不安定な瞬間でも panic でホストプロセスを巻き込まないよう、
/// 失敗は `E_FAIL` に変換して呼び元へ返す。
pub(super) unsafe fn get_insert_range_or_end(
    ctx: &windows::Win32::UI::TextServices::ITfContext,
    ec: u32,
    op: &str,
) -> windows::core::Result<windows::Win32::UI::TextServices::ITfRange> {
    use windows::Win32::Foundation::E_FAIL;

    if let Some(range) = unsafe { get_cursor_range(ctx, ec) } {
        return Ok(range);
    }

    tracing::debug!("{op}: cursor range unavailable, falling back to GetEnd");
    unsafe { ctx.GetEnd(ec) }
        .map_err(|e| windows::core::Error::new(E_FAIL, format!("{op}: GetEnd: {e}")))
}

/// `GetEnd` を安全に取得する。
///
/// `commit_then_start_composition` のように「現在選択位置を使うと意味が変わる」
/// 経路では、cursor range を見に行かず `GetEnd` を明示的に使う。
pub(super) unsafe fn get_document_end_range(
    ctx: &windows::Win32::UI::TextServices::ITfContext,
    ec: u32,
    op: &str,
) -> windows::core::Result<windows::Win32::UI::TextServices::ITfRange> {
    use windows::Win32::Foundation::E_FAIL;

    unsafe { ctx.GetEnd(ec) }
        .map_err(|e| windows::core::Error::new(E_FAIL, format!("{op}: GetEnd: {e}")))
}

pub(super) fn update_composition(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    preedit: String,
) -> Result<()> {
    update_composition_at(ctx, tid, sink, preedit, None)
}

/// [`update_composition`] のキャレット位置指定版。`caret` は composition 先頭からの
/// UTF-16 単位のオフセットで、`None` なら末尾（従来どおり）。未確定中の
/// キャレット編集（← で読みの途中へ戻る）で使う。
pub(super) fn update_composition_at(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    preedit: String,
    caret: Option<i32>,
) -> Result<()> {
    use windows::Win32::Foundation::E_FAIL;

    let existing = composition_clone()?;
    // M1.8 T-MID2: stale check 用に外側 snapshot のポインタを記録。
    // EditSession クロージャは TF_ES_READWRITE で遅延実行されるため、
    // ここで取った composition が DM 破棄や invalidate_composition_for_dm で
    // stale 化したまま SetText しないよう、クロージャ先頭で再検査する。
    let existing_ptr = existing.as_ref().map(|c| c.as_raw() as usize).unwrap_or(0);
    let ctx_req = ctx.clone();
    let session = EditSession::new(move |ec| unsafe {
        use windows::Win32::UI::TextServices::{
            ITfContextComposition, TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
        };

        // M1.8 T-MID2: クロージャ実行時点で composition が
        // 外側 snapshot と同一かを再確認。異なれば SetText せず no-op。
        // - existing=Some, current=None: invalidate_composition_for_dm で stale 化
        // - existing=Some(a), current=Some(b) で a != b: composition が置換された
        // - existing=None, current=Some: 別経路で新規 composition が立った
        // のいずれも安全側で abort する。
        let current = composition_clone()
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("comp re-check: {e}")))?;
        let current_ptr = current.as_ref().map(|c| c.as_raw() as usize).unwrap_or(0);
        if current_ptr != existing_ptr {
            tracing::debug!(
                "update_composition: stale snapshot, abort SetText (existing={:#x} current={:#x})",
                existing_ptr,
                current_ptr
            );
            return Ok(());
        }

        let preedit_w: Vec<u16> = preedit.encode_utf16().collect();
        tracing::debug!(
            "update_composition[EditSession]: preedit={:?} existing={}",
            preedit,
            existing.is_some()
        );

        let range = if let Some(comp) = &existing {
            comp.GetRange()
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("GetRange: {e}")))?
        } else {
            // Fix2: GetEnd(文書末尾)ではなく現在のカーソル位置を使う
            let insert_point = get_insert_range_or_end(&ctx, ec, "update_composition")?;
            let cc: ITfContextComposition = ctx.cast().map_err(|e| {
                windows::core::Error::new(E_FAIL, format!("cast ITfContextComposition: {e}"))
            })?;
            let new_comp = cc
                .StartComposition(ec, &insert_point, &sink)
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("StartComposition: {e}")))?;
            let r = new_comp
                .GetRange()
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("GetRange new: {e}")))?;
            let dm_ptr = ctx
                .GetDocumentMgr()
                .ok()
                .map(|dm| dm.as_raw() as usize)
                .unwrap_or(0);
            let _ = composition_set_with_dm(Some(new_comp), dm_ptr);
            r
        };

        // M1.8 T-MID3: SetText 排他化。Phase1A 経路の SetText と直列化する。
        // busy なら skip し、上位は no-op として処理する（次回 update_composition
        // が新しい preedit で再 SetText するので整合は保てる）。
        {
            let _apply_guard = match crate::engine::state::COMPOSITION_APPLY_LOCK.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    tracing::debug!(
                        "update_composition: COMPOSITION_APPLY_LOCK busy, skip SetText"
                    );
                    return Ok(());
                }
            };
            range
                .SetText(ec, 0, &preedit_w)
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("SetText: {e}")))?;
        }

        // アンダーライン属性をセット
        // SESSION_SELECTING アトミックで高速判定（クロージャ内なので Mutex は取れない）
        let atom = display_attr::atom_input();
        set_display_attr_prop(&ctx, ec, &range, atom);

        // プリエディット中はカーソルを末尾に置く（アプリのキャレット表示を正しくする）。
        // キャレット編集中は指定位置（読みの途中）に置く。
        if let Ok(cursor) = range.Clone() {
            match caret {
                Some(off) => {
                    let mut actual = 0i32;
                    let _ = cursor.ShiftStart(
                        ec,
                        off,
                        &mut actual,
                        std::ptr::null::<windows::Win32::UI::TextServices::TF_HALTCOND>(),
                    );
                    let _ = cursor.Collapse(
                        ec,
                        windows::Win32::UI::TextServices::TF_ANCHOR_START,
                    );
                }
                None => {
                    let _ = cursor.Collapse(ec, TF_ANCHOR_END);
                }
            }
            let sel = TF_SELECTION {
                range: std::mem::ManuallyDrop::new(Some(cursor)),
                style: TF_SELECTIONSTYLE {
                    ase: TfActiveSelEnd(0),
                    fInterimChar: windows::Win32::Foundation::BOOL(0),
                },
            };
            let _ = ctx.SetSelection(ec, &[sel]);
        }

        // 打鍵のたびにキャレット矩形を更新する。
        //
        // 候補ウィンドウの表示位置は `caret_rect_get()` を読むが、CARET_RECT を
        // 更新するのは Space 押下時の `update_caret_rect()` と
        // `commit_then_start_composition()` だけだった。`update_caret_rect()` は
        // 非同期の EditSession を投げるだけなので、同じ Space の処理内で
        // `caret_rect_get()` を読む側はまだ更新前の値を見る。プロセス内で最初の
        // 変換ではそれが初期値 (0,0,0,0) にあたり、候補ウィンドウがプライマリ
        // モニタの左上に出てしまう。
        //
        // ここで更新しておけば、Space が来る前に必ず正しい矩形が入っている。
        if let Ok(view) = ctx.GetActiveView() {
            use windows::Win32::Foundation::RECT;
            let mut rect = RECT::default();
            let mut clipped = windows::Win32::Foundation::BOOL(0);
            match view.GetTextExt(ec, &range, &mut rect, &mut clipped) {
                Ok(()) => {
                    // 全ゼロは「取れなかった」と同義（成功を返しつつ空矩形を
                    // 返すアプリがある）。古い正しい値を壊さないよう捨てる。
                    if rect.bottom != 0 || rect.left != 0 {
                        caret_rect_set(rect);
                        crate::tsf::candidate_window::reposition(rect.left, rect.bottom);
                    } else {
                        tracing::debug!("update_composition: GetTextExt returned empty rect");
                    }
                }
                Err(e) => {
                    tracing::debug!("update_composition: GetTextExt failed: {e}");
                }
            }
        }

        Ok(())
    });
    // preedit の更新は失敗しても次の打鍵で書き直されるためリトライしないが、
    // 「表示が古いまま」の調査にはログが要る。
    unsafe {
        match ctx_req.RequestEditSession(tid, &session, TF_ES_READWRITE) {
            Ok(hr) if hr.is_err() => {
                tracing::debug!("update_composition: edit session failed hr={hr:?}");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!("update_composition: RequestEditSession failed: {e}");
            }
        }
    }
    Ok(())
}

/// 確定テキストを commit し、即座に新しい composition を開始する（1 EditSession）。
///
/// end_composition + update_composition を別々に呼ぶと TSF が2セッションを
/// 別タイミングで実行し、"composition=None" の瞬間にアプリがテキストを
/// クリアすることがある。これを1セッションにまとめて防ぐ。
pub(super) fn commit_then_start_composition(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    commit_text: String,
    next_preedit: String,
) -> Result<()> {
    use windows::Win32::Foundation::E_FAIL;

    // M1.8 T-MID1 拡張: 確定はライブ変換の世代を進める。これをしないと、
    // 確定前に積まれた Phase1A/1B の遅延 SetText が gen 一致のまま
    // 確定後の新 composition に古い preview を書き込み、テキストが
    // 二重に見える競合が起こりうる。
    crate::tsf::live_session::conv_gen_bump();

    // composition_take() をセッション内に移動する（end_composition と同じ理由）。
    // セッション外で take すると COMPOSITION=None になった瞬間に update_composition が
    // 誤ったカーソル位置から新 composition を開始するリスクがある。
    let ctx_req = ctx.clone();
    let text_for_retry = commit_text.clone();
    let ran = Arc::new(AtomicBool::new(false));
    let ran_in_session = ran.clone();
    let session = EditSession::new(move |ec| unsafe {
        if ran_in_session.swap(true, Ordering::SeqCst) {
            tracing::debug!("commit_then_start[session]: already written, skipping");
            return Ok(());
        }
        use windows::Win32::UI::TextServices::{
            ITfContextComposition, TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
        };

        // SetText 排他化（M1.8 T-MID3 の commit 版）: Phase1A / update_composition
        // の SetText と直列化する。他サイトは try_lock + skip だが、確定はスキップ
        // するとユーザーテキストが失われるためブロッキングで取得する。
        // 遅延実行された Phase1A がこのセッションの後に走っても、composition_take
        // で旧 composition が外れているため古い preview の書き込みは失敗する。
        let _apply_guard = crate::engine::state::COMPOSITION_APPLY_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let comp = composition_take().unwrap_or(None);
        tracing::debug!(
            "commit_then_start[session]: commit={:?} next={:?} has_comp={}",
            commit_text,
            next_preedit,
            comp.is_some()
        );

        // ── Step1: 既存 composition を確定テキストで終了 ──
        // 文節分割後に候補表示している場合、composition のテキストは
        // "確定部分 + remainder" の全体になっている。
        // EndComposition だけだとその全体が確定されてしまうため、
        // 先に SetText で commit_text だけに縮めてから EndComposition する。
        let commit_w: Vec<u16> = commit_text.encode_utf16().collect();
        // EndComposition 後の挿入位置: composition range の末尾（確定テキストの直後）を保存する。
        // EndComposition 後は GetSelection が composition 開始位置を返すことがあるため
        // EndComposition 前に range の末尾を取得しておく。
        let mut insert_after_commit: Option<windows::Win32::UI::TextServices::ITfRange> = None;
        if let Some(comp) = comp {
            // composition テキストを commit_text だけに縮める
            if let Ok(range) = comp.GetRange() {
                let _ = range.SetText(ec, 0, &commit_w);
                // 確定テキストの末尾位置を保存
                if let Ok(end_range) = range.Clone() {
                    let _ = end_range.Collapse(ec, TF_ANCHOR_END);
                    insert_after_commit = Some(end_range);
                }
            } else {
                tracing::warn!("commit_then_start: comp.GetRange() failed");
            }
            comp.EndComposition(ec)
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("EndComposition: {e}")))?;
        } else if !commit_text.is_empty() {
            let insert_point =
                get_insert_range_or_end(&ctx, ec, "commit_then_start direct commit")?;
            insert_point.SetText(ec, 0, &commit_w).map_err(|e| {
                windows::core::Error::new(E_FAIL, format!("SetText direct commit: {e}"))
            })?;
            if let Ok(end_range) = insert_point.Clone() {
                let _ = end_range.Collapse(ec, TF_ANCHOR_END);
                insert_after_commit = Some(end_range);
            }
        }

        if next_preedit.is_empty() {
            return Ok(());
        }

        // ── Step2: 同セッション内で新 composition 開始 ──
        // EndComposition 前に保存した確定テキスト末尾位置から新 composition を開始する。
        // EndComposition 後の GetSelection はカーソルが composition 開始位置を示すことがあり
        // 使用できない。ctx.GetEnd(ec) はドキュメント末尾を返すため文章途中の編集で問題になる。
        let insert_point = if let Some(p) = insert_after_commit {
            p
        } else {
            tracing::warn!("commit_then_start: insert_after_commit=None, falling back to GetEnd");
            get_document_end_range(&ctx, ec, "commit_then_start new composition")?
        };
        let cc: ITfContextComposition = ctx.cast().map_err(|e| {
            windows::core::Error::new(E_FAIL, format!("cast ITfContextComposition: {e}"))
        })?;
        let new_comp = cc
            .StartComposition(ec, &insert_point, &sink)
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("StartComposition: {e}")))?;
        let new_range = new_comp
            .GetRange()
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("GetRange new: {e}")))?;
        let dm_ptr = ctx
            .GetDocumentMgr()
            .ok()
            .map(|dm| dm.as_raw() as usize)
            .unwrap_or(0);
        let _ = composition_set_with_dm(Some(new_comp), dm_ptr);

        let preedit_w: Vec<u16> = next_preedit.encode_utf16().collect();
        new_range
            .SetText(ec, 0, &preedit_w)
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("SetText new: {e}")))?;

        // 新 composition にもアンダーライン属性をセット
        set_display_attr_prop(&ctx, ec, &new_range, display_attr::atom_input());

        // カーソルを末尾に
        if let Ok(cursor) = new_range.Clone() {
            let _ = cursor.Collapse(ec, TF_ANCHOR_END);
            let sel = TF_SELECTION {
                range: std::mem::ManuallyDrop::new(Some(cursor)),
                style: TF_SELECTIONSTYLE {
                    ase: TfActiveSelEnd(0),
                    fInterimChar: windows::Win32::Foundation::BOOL(0),
                },
            };
            let _ = ctx.SetSelection(ec, &[sel]);
        }

        // BlockSelecting 位置追従:
        // 新 composition の先頭位置を CARET_RECT に記録し、候補ウィンドウを即時移動する。
        // RequestEditSession は非同期のため、次のキー入力より先にここで更新しないと
        // Space キーハンドラが caret_rect_get() を読む時点でまだ旧値になってしまう。
        if let Ok(view) = ctx.GetActiveView() {
            use windows::Win32::Foundation::RECT;
            let mut rect = RECT::default();
            let mut clipped = windows::Win32::Foundation::BOOL(0);
            if view
                .GetTextExt(ec, &new_range, &mut rect, &mut clipped)
                .is_ok()
            {
                caret_rect_set(rect);
                // 候補ウィンドウが表示中であれば位置を更新（非表示なら何もしない）
                crate::tsf::candidate_window::reposition(rect.left, rect.bottom);
            }
        }

        Ok(())
    });
    // セッションが走らなかった場合、確定テキストと次の preedit の両方が消える。
    // 控えに積めるのは確定テキストだけなので、書き直しは end_composition 相当に
    // なる（次の preedit は engine 側に残っており、次の打鍵で composition が
    // 作り直される）。
    match request_commit_session(
        &ctx_req,
        tid,
        &session,
        &ran,
        "commit_then_start",
        &text_for_retry,
    ) {
        CommitDispatch::Executed => {}
        CommitDispatch::Deferred | CommitDispatch::Rejected => {
            pending_commit_push(&ctx_req, text_for_retry, ran, 0, "commit_then_start");
        }
    }
    Ok(())
}

/// GUID_PROP_ATTRIBUTE プロパティを range にセットしてアンダーラインを要求する
///
/// atom が 0（未登録）の場合は何もしない。
/// アプリが属性を無視する場合もあるが、メモ帳・Word 等の標準アプリでは表示される。
unsafe fn set_display_attr_prop(
    ctx: &ITfContext,
    ec: u32,
    range: &windows::Win32::UI::TextServices::ITfRange,
    atom: u32,
) {
    if atom == 0 {
        return;
    }
    let Ok(prop) = (unsafe { ctx.GetProperty(&GUID_PROP_ATTRIBUTE) }) else {
        return;
    };
    // 既存の属性を先にクリアして TSF に変更を通知させる
    let _ = unsafe { prop.Clear(ec, range) };
    // windows_core::VARIANT で VT_I4 (atom) を設定
    let var = windows_core::VARIANT::from(atom as i32);
    let _ = unsafe { prop.SetValue(ec, range, &var) };
}

/// 変換候補（`converted`）と未変換残り（`remainder`）を1つの composition に表示する。
///
/// `converted` + `remainder` を結合して composition にセットし、属性は
/// converted 部分を atom_converted（太実線）、remainder 部分を atom_input（点線）で付与する。
/// TSF の `ShiftEnd`/`ShiftStart` は実装によって挙動が異なるため使用しない。
/// `GetProperty → EnumerateRanges` ではなく 1 property に 2 値を書く安全な方法として
/// 先に全体を atom_converted で塗り、その後 remainder 部分のみ atom_input で上書きする。
///
/// `remainder` が空の場合は通常の `update_composition` と同じ動作になる。
/// キャレットは composition 全体の末尾に置く。
pub(super) fn update_composition_candidate_parts(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    prefix: String,
    converted: String,
    suffix: String,
) -> Result<()> {
    update_composition_parts_impl(
        ctx,
        tid,
        sink,
        prefix,
        converted,
        suffix,
        CaretPlacement::CompositionEnd,
        display_attr::atom_input(),
    )
}

/// 範囲指定変換（RangeSelect）用: 先頭から `selected` までを選択範囲として表示する。
///
/// 表示属性は `update_composition_candidate_parts` と同じ（選択範囲＝実線、残り＝点線）だが、
/// キャレットを選択範囲の末尾に置く。下線を描画しないアプリでも、Shift+Right / Left で
/// キャレットが動くことで範囲の変化が分かるようにするため。
pub(super) fn update_composition_range_select(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    selected: String,
    unselected: String,
) -> Result<()> {
    update_composition_parts_impl(
        ctx,
        tid,
        sink,
        String::new(),
        selected,
        unselected,
        CaretPlacement::ConvertedEnd,
        display_attr::atom_input(),
    )
}

/// composition 内でのキャレットの置き場所。
#[derive(Clone, Copy, PartialEq, Eq)]
enum CaretPlacement {
    /// composition 全体の末尾（通常の候補表示）
    CompositionEnd,
    /// `converted` 部分の末尾（範囲指定変換）
    ConvertedEnd,
}

pub(super) fn update_composition_block_parts(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    prefix: String,
    converted: String,
    suffix: String,
) -> Result<()> {
    update_composition_parts_impl(
        ctx,
        tid,
        sink,
        prefix,
        converted,
        suffix,
        CaretPlacement::CompositionEnd,
        display_attr::atom_done(),
    )
}

fn update_composition_parts_impl(
    ctx: ITfContext,
    tid: u32,
    sink: ITfCompositionSink,
    prefix: String,
    converted: String,
    suffix: String,
    caret: CaretPlacement,
    suffix_atom: u32,
) -> Result<()> {
    use windows::Win32::Foundation::E_FAIL;

    if prefix.is_empty() && suffix.is_empty() {
        return update_composition(ctx, tid, sink, converted);
    }

    let existing = composition_clone()?;
    // M1.8 T-MID2: update_composition と同じ stale check を入れる
    let existing_ptr = existing.as_ref().map(|c| c.as_raw() as usize).unwrap_or(0);
    let ctx_req = ctx.clone();
    let full = format!("{prefix}{converted}{suffix}");
    let prefix_utf16: i32 = prefix.encode_utf16().count() as i32;
    let converted_utf16: i32 = converted.encode_utf16().count() as i32;
    let suffix_utf16_all: i32 = suffix.encode_utf16().count() as i32;

    let session = EditSession::new(move |ec| unsafe {
        use windows::Win32::UI::TextServices::{
            ITfContextComposition, TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
        };

        // M1.8 T-MID2: クロージャ実行時点の stale check
        let current = composition_clone()
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("comp re-check: {e}")))?;
        let current_ptr = current.as_ref().map(|c| c.as_raw() as usize).unwrap_or(0);
        if current_ptr != existing_ptr {
            tracing::debug!(
                "update_composition_candidate_parts: stale snapshot, abort SetText (existing={:#x} current={:#x})",
                existing_ptr,
                current_ptr
            );
            return Ok(());
        }

        let full_w: Vec<u16> = full.encode_utf16().collect();

        // ── Step1: テキストをセット ──
        let range = if let Some(comp) = &existing {
            comp.GetRange()
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("GetRange: {e}")))?
        } else {
            let insert_point =
                get_insert_range_or_end(&ctx, ec, "update_composition_candidate_parts")?;
            let cc: ITfContextComposition = ctx
                .cast()
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("cast: {e}")))?;
            let new_comp = cc
                .StartComposition(ec, &insert_point, &sink)
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("StartComposition: {e}")))?;
            let r = new_comp
                .GetRange()
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("GetRange new: {e}")))?;
            let dm_ptr = ctx
                .GetDocumentMgr()
                .ok()
                .map(|dm| dm.as_raw() as usize)
                .unwrap_or(0);
            let _ = composition_set_with_dm(Some(new_comp), dm_ptr);
            r
        };

        // M1.8 T-MID3: SetText 排他化（candidate_parts 経路）。
        {
            let _apply_guard = match crate::engine::state::COMPOSITION_APPLY_LOCK.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    tracing::debug!(
                        "update_composition_candidate_parts: COMPOSITION_APPLY_LOCK busy, skip SetText"
                    );
                    return Ok(());
                }
            };
            range
                .SetText(ec, 0, &full_w)
                .map_err(|e| windows::core::Error::new(E_FAIL, format!("SetText: {e}")))?;
        }

        // ── Step2: 属性セット ──
        // 全体を atom_input（点線）で塗り、選択中ブロックのみ atom_converted（太実線）で上書きする
        // prefix（変換済み文節）は細実線。suffix は呼び出し側の指定
        // （未変換の読みなら点線、後続の変換済み文節なら細実線）。
        set_display_attr_prop(&ctx, ec, &range, display_attr::atom_done());
        if suffix_utf16_all > 0 && suffix_atom != display_attr::atom_done() {
            if let Ok(suf_range) = range.Clone() {
                let mut actual = 0i32;
                let _ = suf_range.ShiftStart(
                    ec,
                    prefix_utf16 + converted_utf16,
                    &mut actual,
                    std::ptr::null::<windows::Win32::UI::TextServices::TF_HALTCOND>(),
                );
                set_display_attr_prop(&ctx, ec, &suf_range, suffix_atom);
            }
        }
        let mut converted_range = None;
        if let Ok(sel_range) = range.Clone() {
            let mut actual = 0i32;
            let suffix_utf16: i32 = suffix.encode_utf16().count() as i32;
            let _ = sel_range.ShiftStart(
                ec,
                prefix_utf16,
                &mut actual,
                std::ptr::null::<windows::Win32::UI::TextServices::TF_HALTCOND>(),
            );
            if suffix_utf16 > 0 {
                let _ = sel_range.ShiftEnd(
                    ec,
                    -suffix_utf16,
                    &mut actual,
                    std::ptr::null::<windows::Win32::UI::TextServices::TF_HALTCOND>(),
                );
            }
            set_display_attr_prop(&ctx, ec, &sel_range, display_attr::atom_converted());
            converted_range = Some(sel_range);
        }

        // ── Step3: キャレットを置く ──
        // CompositionEnd: composition 全体の末尾（従来どおり）
        // ConvertedEnd  : converted 部分の末尾（範囲指定変換。converted 範囲が取れなければ末尾）
        let cursor_base = match (caret, converted_range) {
            (CaretPlacement::ConvertedEnd, Some(r)) => Ok(r),
            _ => range.Clone(),
        };
        if let Ok(cursor) = cursor_base {
            let _ = cursor.Collapse(ec, TF_ANCHOR_END);
            let sel = TF_SELECTION {
                range: std::mem::ManuallyDrop::new(Some(cursor)),
                style: TF_SELECTIONSTYLE {
                    ase: TfActiveSelEnd(0),
                    fInterimChar: windows::Win32::Foundation::BOOL(0),
                },
            };
            let _ = ctx.SetSelection(ec, &[sel]);
        }
        Ok(())
    });
    unsafe {
        match ctx_req.RequestEditSession(tid, &session, TF_ES_READWRITE) {
            Ok(hr) if hr.is_err() => {
                tracing::debug!("candidate_split: edit session failed hr={hr:?}");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!("candidate_split: RequestEditSession failed: {e}");
            }
        }
    }
    Ok(())
}

/// スペース押下時のみ呼ぶ: caret_rect をキャレット位置で更新する
pub(super) fn update_caret_rect(ctx: ITfContext, tid: u32) {
    let comp = match composition_clone() {
        Ok(Some(c)) => c,
        _ => return,
    };
    let ctx_req = ctx.clone();
    let session = EditSession::new(move |ec| unsafe {
        if let Ok(range) = comp.GetRange()
            && let Ok(view) = ctx.GetActiveView()
        {
            use windows::Win32::Foundation::RECT;
            let mut rect = RECT::default();
            let mut clipped = windows::Win32::Foundation::BOOL(0);
            if view.GetTextExt(ec, &range, &mut rect, &mut clipped).is_ok() {
                caret_rect_set(rect);
            }
        }
        Ok(())
    });
    unsafe {
        let _ = ctx_req.RequestEditSession(tid, &session, TF_ES_READWRITE);
    }
}

pub(super) fn end_composition(ctx: ITfContext, tid: u32, text: String) -> Result<()> {
    end_composition_attempt(ctx, tid, text, 0, Arc::new(AtomicBool::new(false)))
}

/// `end_composition` の本体。
///
/// - `attempt`: `retry_pending_commit` からの書き直し回数。
/// - `claim`: 書き込みの占有券。書き直しでは元のセッションと同じものを渡し、
///   先に走った側だけが実際に書くようにする。
fn end_composition_attempt(
    ctx: ITfContext,
    tid: u32,
    text: String,
    attempt: u32,
    claim: Arc<AtomicBool>,
) -> Result<()> {
    use windows::Win32::Foundation::E_FAIL;
    use windows::Win32::UI::TextServices::{
        TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
    };
    // 確定はライブ変換の世代を進める（commit_then_start_composition と同じ理由）。
    // 確定前に積まれた Phase1A/1B の遅延 SetText を gen 不一致で棄却させる。
    crate::tsf::live_session::conv_gen_bump();
    // composition_take() をセッション内に移動する。
    // セッション外で take すると COMPOSITION=None になった直後に次のキー入力が来たとき、
    // update_composition が existing=None を見て誤った位置から新 composition を開始してしまう。
    let ctx2 = ctx.clone();
    let text_for_retry = text.clone();
    let ran = claim;
    let ran_in_session = ran.clone();
    let session = EditSession::new(move |ec| unsafe {
        if ran_in_session.swap(true, Ordering::SeqCst) {
            tracing::debug!("end_composition[session]: already written, skipping");
            return Ok(());
        }
        // SetText 排他化（commit_then_start_composition と同様、確定なので
        // try_lock + skip ではなくブロッキングで取得する）
        let _apply_guard = crate::engine::state::COMPOSITION_APPLY_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let comp = match composition_take().unwrap_or(None) {
            Some(c) => c,
            None => {
                tracing::debug!("end_composition: no composition, inserting text directly");
                // composition がない場合はカーソル位置に直接挿入
                if !text.is_empty() {
                    let text_w: Vec<u16> = text.encode_utf16().collect();
                    let insert =
                        get_insert_range_or_end(&ctx2, ec, "end_composition direct insert")?;
                    if let Err(e) = insert.SetText(ec, 0, &text_w) {
                        tracing::warn!(
                            "end_composition: direct insert SetText failed text={:?}: {e}",
                            text
                        );
                    }
                }
                return Ok(());
            }
        };

        let text_w: Vec<u16> = text.encode_utf16().collect();
        tracing::debug!("end_composition[session]: text={:?}", text);
        let range = comp.GetRange().map_err(|e| {
            tracing::warn!("end_composition: GetRange failed text={:?}: {e}", text);
            windows::core::Error::new(E_FAIL, format!("GetRange: {e}"))
        })?;
        if let Err(e) = range.SetText(ec, 0, &text_w) {
            // 0x80040209 は TSF では TS_E_READONLY（ドキュメントが一時的に読み取り専用。
            // FormatMessage は同値の OLE エラー文字列を出すため紛らわしい）。
            // 一時的なロックなら同一セッション内の再試行で通ることがある。
            if let Err(e2) = range.SetText(ec, 0, &text_w) {
                // TSF DLL はアプリのプロセス内で動くため current_exe = 発生アプリ
                let app = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_default();
                tracing::warn!(
                    "end_composition: SetText failed text={:?} app={:?}: {e} / retry: {e2}",
                    text,
                    app
                );
                // SetText できなくても EndComposition までは進め、表示中のテキスト
                // （preedit）をそのまま確定させる。中断すると composition が宙吊りに
                // なり確定テキストが丸ごと消える（7月に 2 件実測）。表示済みテキストの
                // 確定は WYSIWYG 不変条件の範囲内。
            }
        }

        // Fix3: EndComposition の前に SetSelection する
        // （EndComposition 後に SetSelection するとアプリがカーソルをリセットしてしまうため）
        if let Ok(cursor) = range.Clone() {
            let _ = cursor.Collapse(ec, TF_ANCHOR_END);
            let sel = TF_SELECTION {
                range: std::mem::ManuallyDrop::new(Some(cursor)),
                style: TF_SELECTIONSTYLE {
                    ase: TfActiveSelEnd(0),
                    fInterimChar: windows::Win32::Foundation::BOOL(0),
                },
            };
            let _ = ctx2.SetSelection(ec, &[sel]);
        }

        comp.EndComposition(ec).map_err(|e| {
            tracing::warn!(
                "end_composition: EndComposition failed text={:?}: {e}",
                text
            );
            windows::core::Error::new(E_FAIL, format!("EndComposition: {e}"))
        })?;
        Ok(())
    });
    // 確定はユーザーテキストを失うと致命的なので、edit session の結果
    // (phrSession) まで確認し、走らなかったなら書き直しの控えに積む。
    match request_commit_session(
        &ctx,
        tid,
        &session,
        &ran,
        "end_composition",
        &text_for_retry,
    ) {
        CommitDispatch::Executed => {}
        CommitDispatch::Deferred | CommitDispatch::Rejected => {
            pending_commit_push(&ctx, text_for_retry, ran, attempt, "end_composition");
        }
    }
    Ok(())
}

pub(super) fn commit_text(ctx: ITfContext, tid: u32, text: String) -> Result<()> {
    use windows::Win32::Foundation::E_FAIL;

    let ctx_req = ctx.clone();
    let text_for_retry = text.clone();
    let ran = Arc::new(AtomicBool::new(false));
    let ran_in_session = ran.clone();
    let session = EditSession::new(move |ec| unsafe {
        if ran_in_session.swap(true, Ordering::SeqCst) {
            tracing::debug!("commit_text[session]: already written, skipping");
            return Ok(());
        }
        use windows::Win32::UI::TextServices::{
            TF_ANCHOR_END, TF_SELECTION, TF_SELECTIONSTYLE, TfActiveSelEnd,
        };
        let text_w: Vec<u16> = text.encode_utf16().collect();
        // 現在のカーソル位置に挿入（GetEnd=文書末尾ではなくカーソル位置）
        let insert = get_insert_range_or_end(&ctx, ec, "commit_text")?;
        insert
            .SetText(ec, 0, &text_w)
            .map_err(|e| windows::core::Error::new(E_FAIL, format!("SetText commit: {e}")))?;
        // 挿入したテキストの末尾にカーソルを移動
        if let Ok(cursor) = insert.Clone() {
            let _ = cursor.Collapse(ec, TF_ANCHOR_END);
            let sel = TF_SELECTION {
                range: std::mem::ManuallyDrop::new(Some(cursor)),
                style: TF_SELECTIONSTYLE {
                    ase: TfActiveSelEnd(0),
                    fInterimChar: windows::Win32::Foundation::BOOL(0),
                },
            };
            let _ = ctx.SetSelection(ec, &[sel]);
        }
        Ok(())
    });
    match request_commit_session(
        &ctx_req,
        tid,
        &session,
        &ran,
        "commit_text",
        &text_for_retry,
    ) {
        CommitDispatch::Executed => {}
        CommitDispatch::Deferred | CommitDispatch::Rejected => {
            pending_commit_push(&ctx_req, text_for_retry, ran, 0, "commit_text");
        }
    }
    Ok(())
}
