//! 使い捨て: ライブ変換が追いつかないまま確定された「後半が生かな」の学習
//! エントリを洗い出す。
//!
//! 症状（2026-09-01）: 速く打つと on_live_timer が preview を更新できず、
//! 「変換済みの前半 ＋ 打ったままのかな」がそのまま確定され、その壊れた表記が
//! `learn_force` で学習履歴に載る。短文予測の材料になるので再生産される。
//!
//! 判定: 読みと表記の**末尾がかなで一致している長さ**を見る。正常な変換でも
//! 語尾のかなは一致するので、文節をまたぐ長さ（既定 8 文字以上）だけを候補に出す。
use rakukan_dict::DictStore;

const MIN_TAIL: usize = 8;

fn is_kana(c: char) -> bool {
    matches!(c, 'ぁ'..='ゖ' | 'ー')
}

/// 読みと表記の共通するかな末尾の長さ（文字数）。
fn common_kana_tail(reading: &str, surface: &str) -> usize {
    let r: Vec<char> = reading.chars().collect();
    let s: Vec<char> = surface.chars().collect();
    let mut n = 0;
    while n < r.len() && n < s.len() {
        let rc = r[r.len() - 1 - n];
        let sc = s[s.len() - 1 - n];
        if rc != sc || !is_kana(rc) {
            break;
        }
        n += 1;
    }
    n
}

fn load() -> DictStore {
    let user = rakukan_dict::user_dict_path();
    let mozc = rakukan_dict::find_mozc_dict();
    let learn = rakukan_dict::learn_history_path();
    DictStore::load(user.as_deref(), mozc.as_deref(), learn.as_deref()).unwrap()
}

fn suspects(store: &DictStore) -> Vec<(String, String, usize)> {
    let mut out: Vec<(String, String, usize)> = store
        .learn_entries_snapshot()
        .into_iter()
        .filter_map(|(reading, surface)| {
            let tail = common_kana_tail(&reading, &surface);
            (tail >= MIN_TAIL).then_some((reading, surface, tail))
        })
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

#[test]
fn probe_tail_kana_entries() {
    let store = load();
    let all = store.learn_entries_snapshot();
    let found = suspects(&store);
    println!("learn_entries={} suspects={}", all.len(), found.len());
    for (reading, surface, tail) in &found {
        println!("  tail={tail:>2}  {reading:?} → {surface:?}");
    }
}

/// `RAKUKAN_FORGET_READING` に読みを渡した時だけ、その読みの学習エントリを削除する。
///
/// 上の一覧は「読みと表記の末尾が長くかなで一致する」だけの機械判定なので、正常な
/// 変換（語尾がかなで長いだけ）が大量に混ざる。一括削除はしない。消すものは目視で
/// 選んで、この入口から 1 件ずつ落とす。
///
/// エンジンホストを止めてから実行すること（動いているとメモリ上の履歴で上書きされる）。
#[test]
fn forget_one_reading() {
    let Ok(reading) = std::env::var("RAKUKAN_FORGET_READING") else {
        println!("skipped (set RAKUKAN_FORGET_READING=<読み>)");
        return;
    };
    let store = load();
    let targets: Vec<(String, String)> = store
        .learn_entries_snapshot()
        .into_iter()
        .filter(|(r, _)| *r == reading)
        .collect();
    if targets.is_empty() {
        println!("no entry for {reading:?}");
        return;
    }
    for (reading, surface) in &targets {
        let ok = store.forget(reading, surface);
        println!("forget({reading:?}, {surface:?}) = {ok}");
    }
}
