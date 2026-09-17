//! 使い捨て: 予測確定で焼き付いた「読みより長い表記」の学習エントリを洗い出す。
use rakukan_dict::DictStore;

#[test]
fn probe_prediction_shadow_entries() {
    let user = rakukan_dict::user_dict_path();
    let mozc = rakukan_dict::find_mozc_dict();
    let learn = rakukan_dict::learn_history_path();
    let store = DictStore::load(user.as_deref(), mozc.as_deref(), learn.as_deref()).unwrap();
    let all = store.learn_entries_snapshot();
    println!("learn_entries={}", all.len());

    // 同じ surface が「短いキー」と「それを prefix に持つ長いキー」の両方にある場合、
    // 短い方が予測候補の確定で焼き付いた汚染。
    let mut shadows: Vec<(String, String, String)> = Vec::new();
    for (short_key, surface) in &all {
        for (long_key, other_surface) in &all {
            if other_surface == surface
                && long_key.len() > short_key.len()
                && long_key.starts_with(short_key.as_str())
            {
                shadows.push((short_key.clone(), long_key.clone(), surface.clone()));
                break;
            }
        }
    }
    println!("shadow_entries={}", shadows.len());
    for (short_key, long_key, surface) in &shadows {
        println!("  {short_key:?} → {surface:?}   (正: {long_key:?})");
    }
}

/// `RAKUKAN_PURGE=1` の時だけ、上で列挙した汚染エントリを実際に削除する。
/// エンジンホストを止めてから実行すること（動いていると上書きされる）。
#[test]
fn purge_prediction_shadow_entries() {
    if std::env::var("RAKUKAN_PURGE").ok().as_deref() != Some("1") {
        println!("skipped (set RAKUKAN_PURGE=1 to run)");
        return;
    }
    let user = rakukan_dict::user_dict_path();
    let mozc = rakukan_dict::find_mozc_dict();
    let learn = rakukan_dict::learn_history_path();
    let store = DictStore::load(user.as_deref(), mozc.as_deref(), learn.as_deref()).unwrap();
    let all = store.learn_entries_snapshot();
    let mut targets: Vec<(String, String)> = Vec::new();
    for (short_key, surface) in &all {
        if all.iter().any(|(long_key, other)| {
            other == surface
                && long_key.len() > short_key.len()
                && long_key.starts_with(short_key.as_str())
        }) {
            targets.push((short_key.clone(), surface.clone()));
        }
    }
    for (reading, surface) in &targets {
        let ok = store.forget(reading, surface);
        println!("forget({reading:?}, {surface:?}) = {ok}");
    }
    println!("purged={}", targets.len());
}
