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
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

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
        } else if run.kind == RunKind::Kanji {
            total += kanji_compound_score(store, &run.reading, &run.surface);
        }
    }
    Some(total)
}

/// 漢字 1 文字あたりの読みの最大長。辞書に無い漢字を読み飛ばす幅に使う。
const MAX_READING_PER_KANJI: usize = 4;

/// 辞書を引く読み片の最大長（`りんしゃんかいほう` = 9 が入る程度）。
const MAX_PIECE_READING_CHARS: usize = 10;

/// 読み片 1 つに対する辞書表記と重み。出自が重なる表記は重い方を採る。
fn piece_surfaces(store: &DictStore, reading: &str) -> Vec<(Vec<char>, f64)> {
    let mut out: Vec<(Vec<char>, f64)> = Vec::new();
    let sources = [
        (store.lookup_user(reading), WEIGHT_USER),
        (store.lookup_learn(reading), WEIGHT_LEARN),
        (store.lookup_dict(reading, DICT_LOOKUP_LIMIT), WEIGHT_DICT),
    ];
    for (surfaces, w) in sources {
        for surface in surfaces {
            let chars: Vec<char> = surface.chars().collect();
            // 重みの大きい出自から順に積むので、既出の表記は捨ててよい
            if !out.iter().any(|(s, _)| *s == chars) {
                out.push((chars, w));
            }
        }
    }
    out
}

/// 漢字 run が丸ごとは辞書に無いとき、辞書の語の連なりとして採点する。
///
/// 漢字が続くと `麻雀用語` のように 1 つの run になり、`まーじゃんようご` では
/// 辞書に当たらず 0 点になる。一方 `マージャン用語` はカタカナで run が切れるので
/// `マージャン` と `用語` の両方が加点され、LLM が第 1 候補に出した漢字表記を
/// カタカナ表記が追い越していた（麻雀用語・リーチ/立直・ゼッタイ/絶対 など）。
/// そこで読みを区切って辞書を引き、表層の続きがその表記で始まっていれば
/// 片ごとに読み長の二乗で加点する。辞書に無い漢字は 1 文字ずつ 0 点で読み飛ばす。
///
/// 辞書は読み片ごとに 1 回だけ引き、表層側の区切りは引いた表記との前方一致で
/// 決める。打鍵ごとに n-best 全件へ走るので、表層×読みの総当たりにはしない。
fn kanji_compound_score(store: &DictStore, reading: &str, surface: &str) -> f64 {
    let r: Vec<char> = reading.chars().collect();
    let s: Vec<char> = surface.chars().collect();
    if s.len() < 2 || r.len() < MIN_RUN_READING_CHARS {
        return 0.0;
    }
    let mut cache: HashMap<(usize, usize), Vec<(Vec<char>, f64)>> = HashMap::new();
    // best[i][j] = 読み r[..i] と表層 s[..j] を対応させ切ったときの最高点
    let mut best = vec![vec![f64::NEG_INFINITY; s.len() + 1]; r.len() + 1];
    best[0][0] = 0.0;
    for j in 0..s.len() {
        for i in 0..r.len() {
            let base = best[i][j];
            if base == f64::NEG_INFINITY {
                continue;
            }
            // 辞書に無い漢字 1 文字を読み飛ばす（0 点）
            for i2 in i + 1..=(i + MAX_READING_PER_KANJI).min(r.len()) {
                if base > best[i2][j + 1] {
                    best[i2][j + 1] = base;
                }
            }
            // 辞書の語として進む
            for i2 in i + MIN_RUN_READING_CHARS..=(i + MAX_PIECE_READING_CHARS).min(r.len()) {
                // 丸ごとの run は呼び出し側で引き済み
                if i == 0 && i2 == r.len() {
                    continue;
                }
                let n = i2 - i;
                let entries = cache.entry((i, i2)).or_insert_with(|| {
                    let piece: String = r[i..i2].iter().collect();
                    piece_surfaces(store, &piece)
                });
                for (surf, w) in entries.iter() {
                    let j2 = j + surf.len();
                    if j2 > s.len() || s[j..j2] != surf[..] {
                        continue;
                    }
                    let score = base + w * (n * n) as f64;
                    if score > best[i2][j2] {
                        best[i2][j2] = score;
                    }
                }
            }
        }
    }
    best[r.len()][s.len()].max(0.0)
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

/// 送り仮名として学習キーに含める、漢字 run 直後のひらがなの最大文字数。
/// `重い`・`帰る` は 1、`新しい`・`変わる` は 2。
const INFLECTED_MAX_OKURI_CHARS: usize = 2;

/// 繰り上げを許す、第 1 候補との自信度（平均 log-prob）の差の上限。
///
/// 実ログ 4,330 変換で再生すると、直したい取り違え（`動作が思い`→`重い` が 0.17、
/// `プレビューが思い`→`重い` が 0.06、`もっと起き`→`置き` が 0.13）と、LLM が
/// 文脈で正しく選んでいるもの（`づくりを進めて`→`勧めて` が 1.0）の間に開きがある。
const INFLECTED_MAX_CONFIDENCE_GAP: f32 = 0.4;

/// 送り仮名ではなく助詞として付く文字。`あしの → 足の` のような「語＋助詞」の
/// 学習を、文中の `脚の` の選び直しに使わない。
const INFLECTED_PARTICLE_TAILS: &[char] = &['の', 'を', 'へ'];

/// 直近の変換の自信度を覚えておく件数。打鍵ごとの BG 変換と Space 変換が
/// 入れ替わりで入るので、数件あれば足りる。
const CONFIDENCE_MEMO_CAPACITY: usize = 16;

/// 直近の変換の `(読み, [(候補, 平均 log-prob)])`。
///
/// n-best は文字列だけで TSF との間を往復するので、自信度はここへ別に控えておき、
/// 読みと候補の文字列で引き直す。数字・英字で読みが割られた変換は読みが一致せず
/// 引けないが、そのときは繰り上げないだけで済む。
static CONFIDENCE_MEMO: Mutex<VecDeque<(String, Vec<(String, f32)>)>> =
    Mutex::new(VecDeque::new());

/// 変換器が出した候補の自信度を控える。
pub fn note_confidence(reading: &str, scored: &[(String, f32)]) {
    let Ok(mut memo) = CONFIDENCE_MEMO.lock() else {
        return;
    };
    memo.retain(|(r, _)| r != reading);
    if memo.len() >= CONFIDENCE_MEMO_CAPACITY {
        memo.pop_front();
    }
    memo.push_back((reading.to_string(), scored.to_vec()));
}

/// `top` と `other` の自信度の差（`top - other`）。どちらかが控えに無ければ `None`。
pub fn noted_confidence_gap(reading: &str, top: &str, other: &str) -> Option<f32> {
    let memo = CONFIDENCE_MEMO.lock().ok()?;
    let (_, scored) = memo.iter().find(|(r, _)| r == reading)?;
    let of = |c: &str| scored.iter().find(|(s, _)| s == c).map(|(_, lp)| *lp);
    Some(of(top)? - of(other)?)
}

fn is_kanji(c: char) -> bool {
    matches!(c, '一'..='鿿' | '㐀'..='䶿' | '々')
}

/// `promote_learned_inflection` が選び直した 1 か所。
struct InflectionPick {
    /// 繰り上げる候補の位置（1 以上）
    index: usize,
    /// 学習キー（漢字 run の読み＋送り仮名）
    key: String,
    /// 第 1 候補がその区間に出していた表記
    llm_surface: String,
}

fn learned_inflection_pick(
    store: &DictStore,
    reading: &str,
    candidates: &[String],
    confidence_gap: &dyn Fn(&str, &str) -> Option<f32>,
) -> Option<InflectionPick> {
    if candidates.len() < 2 {
        return None;
    }
    let first = &candidates[0];
    let runs = align(reading, first)?;
    let first_chars: Vec<char> = first.chars().collect();
    let mut spos = 0usize;
    for (i, run) in runs.iter().enumerate() {
        let start = spos;
        let run_len = run.surface.chars().count();
        spos += run_len;
        if run.kind != RunKind::Kanji || !run.surface.chars().all(is_kanji) {
            continue;
        }
        let Some(next) = runs.get(i + 1).filter(|n| n.kind == RunKind::Hiragana) else {
            continue;
        };
        let following: Vec<char> = next.surface.chars().collect();
        for k in 1..=INFLECTED_MAX_OKURI_CHARS.min(following.len()) {
            // 読み全体がその語なら merge 側の完全一致に任せる。
            if start == 0 && run_len + k == first_chars.len() {
                break;
            }
            if INFLECTED_PARTICLE_TAILS.contains(&following[k - 1]) {
                continue;
            }
            let okuri: String = following[..k].iter().collect();
            let key = format!("{}{}", run.reading, okuri);
            let llm_surface = format!("{}{}", run.surface, okuri);
            // 並びは「最後に確定した表記が先頭」。LLM と同じ表記を最後に選んで
            // いるなら、その語は今は LLM の表記で使っている。
            let learned = store.lookup_learn(&key);
            let Some(top) = learned.first() else {
                continue;
            };
            if *top == llm_surface {
                continue;
            }
            // 同じ送り仮名で終わる「漢字＋送り仮名」の形だけを対象にする。
            let Some(stem) = top.strip_suffix(okuri.as_str()) else {
                continue;
            };
            if stem.is_empty() || !stem.chars().all(is_kanji) {
                continue;
            }
            if store.learn_freq(&key, top) < LEARNED_SOFT_MIN_FREQ {
                continue;
            }
            let mut replaced: String = first_chars[..start].iter().collect();
            replaced.push_str(top);
            replaced.extend(&first_chars[start + run_len + k..]);
            if let Some(index) = candidates.iter().position(|c| *c == replaced)
                && confidence_gap(first, &replaced)
                    .is_some_and(|g| g <= INFLECTED_MAX_CONFIDENCE_GAP)
            {
                return Some(InflectionPick {
                    index,
                    key,
                    llm_surface,
                });
            }
        }
    }
    None
}

/// 送り仮名つきの同音語（`思い`/`重い`、`帰る`/`変える`）を、学習で選び直す。
///
/// `apply_learned_runs` は漢字 run の読みだけで学習を引くので、`動作が思い` の
/// `思` は `おも` で引かれ、`おもい → 重い` の学習に当たらない。送り仮名のある
/// 形容詞・動詞はすべてこの形で、単独では何度 `重い` を選んでいても文中では
/// LLM の `思い` がそのまま出ていた。
///
/// そこで漢字 run に直後のひらがなを 1〜2 文字足した読みで学習を引き、最後に
/// 選んだ表記が LLM と違っていて、かつ **その表記へ差し替えた文が n-best の中に
/// あり、LLM の自信度が第 1 候補と僅差の** ときだけ、そちらを先頭へ繰り上げる。
/// LLM 自身が迷っている候補からしか選ばないので、文として成り立たない差し替え
/// （`と思います`→`と重います`）や、文脈ではっきり決まっている語の塗り替えは
/// 起きない。`confidence_gap` は `(第 1 候補, 繰り上げる候補)` の自信度の差を返す。
///
/// 候補の集合・件数は変えない。断られたら（元の第 1 候補が確定されたら）
/// `LearnedRewrite` 経由でその読みに LLM の表記を学習し、次に学習表記を単独で
/// 選び直すまでは繰り上げない。
pub fn promote_learned_inflection(
    store: &DictStore,
    reading: &str,
    candidates: Vec<String>,
    confidence_gap: &dyn Fn(&str, &str) -> Option<f32>,
) -> (Vec<String>, Option<LearnedRewrite>) {
    let Some(pick) = learned_inflection_pick(store, reading, &candidates, confidence_gap) else {
        return (candidates, None);
    };
    let mut out = candidates;
    let promoted = out.remove(pick.index);
    tracing::info!(
        reading = %reading,
        promoted = %promoted,
        from_rank = pick.index,
        key = %pick.key,
        "rescore: promoted learned inflection"
    );
    let rewrite = LearnedRewrite {
        original: out[0].clone(),
        runs: vec![(pick.key, pick.llm_surface)],
    };
    out.insert(0, promoted);
    (out, Some(rewrite))
}

/// 語幹の一族として数える、語幹の読みより後ろ（送り仮名・活用語尾）の最大文字数。
/// `描く`=1、`描いた`=2、`描かない`=3、`描きました`=4。
const STEM_MAX_TAIL_CHARS: usize = 4;

/// 1 回の変換で 2 番目以降に足す、語幹違いの候補の最大数。
const STEM_MAX_OFFERS: usize = 2;

/// 送り仮名つきの形が辞書に載っているかを見るときの、辞書の引き数。
const STEM_DICT_LOOKUP_LIMIT: usize = 64;

/// 語幹の読み `stem_reading` で、`stem_kanji` と**同じ読みキーの下で選び分けた
/// ことのある**漢字を、最後に選んだ順で返す。
///
/// 戻り値は `(stem_kanji を最後に選んだ時刻, [(別の漢字, 最後に選んだ時刻)])`。
/// `かく → 書く / 描く` の両方を選んだことがあれば、`か`・`書` に対して `描` が返る。
/// 活用形はどれでもよく（`かいた → 描いた` も `描` の時刻に数える）、ここで活用形を
/// またいだ共有が起きる。読みが同じだけの別の動詞（`買う`）は、同じキーで
/// 選び分けたことが無いかぎり混ざらない。
fn stem_rivals(
    store: &DictStore,
    stem_reading: &str,
    stem_kanji: &str,
) -> Option<(u64, Vec<(String, u64)>)> {
    let mut by_key: HashMap<String, Vec<String>> = HashMap::new();
    let mut last: HashMap<String, u64> = HashMap::new();
    for (key, surface, at) in store.learn_entries_near(stem_reading, STEM_MAX_TAIL_CHARS) {
        let tail = &key[stem_reading.len()..];
        if !tail.chars().all(|c| classify(c) == RunKind::Hiragana) {
            continue;
        }
        let Some(kanji) = surface.strip_suffix(tail) else {
            continue;
        };
        if kanji.is_empty() || !kanji.chars().all(is_kanji) {
            continue;
        }
        let slot = last.entry(kanji.to_string()).or_insert(0);
        *slot = (*slot).max(at);
        by_key.entry(key).or_default().push(kanji.to_string());
    }
    let own = *last.get(stem_kanji)?;
    let mut rivals: Vec<(String, u64)> = Vec::new();
    for kanjis in by_key.values() {
        if !kanjis.iter().any(|k| k == stem_kanji) {
            continue;
        }
        for k in kanjis {
            if k != stem_kanji && !rivals.iter().any(|(r, _)| r == k) {
                rivals.push((k.clone(), last[k]));
            }
        }
    }
    if rivals.is_empty() {
        return None;
    }
    rivals.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Some((own, rivals))
}

/// `offer_learned_stems` が出した語幹違いの候補 1 件。
#[derive(Clone, Debug, PartialEq)]
pub struct StemOffer {
    /// 語幹を入れ替えた文全体
    pub sentence: String,
    /// 入れ替えた区間の読み（語幹＋送り仮名 1 文字。`かき`）
    pub key: String,
    /// 入れ替え後の区間の表記（`描き`）
    pub surface: String,
}

/// 活用形をまたいで学習を共有し、同音の語幹（`書`/`描`）を選びやすくする。
///
/// 学習は読みの完全一致でしか引かれないので、`かく → 描く` を何度選んでも
/// `かき`・`かいた` には効かず、`描き直そう` は n-best の下位か、そもそも出ない。
/// そこで第 1 候補の「漢字＋送り仮名」ごとに、同じ語幹の読みで選び分けたことの
/// ある別の漢字（`stem_rivals`）を引き、入れ替えた文を作る。
///
/// - 入れ替えた文は **必ず 2 番目**に置く（n-best にあれば移し、無ければ足す）。
///   先頭は動かさないので、ライブ変換の preview は変わらない。
/// - `allow_promote` で、別の漢字のほうを**後に**選んでいて、入れ替えた文が n-best に
///   あり、LLM の自信度が僅差のときだけ先頭へ繰り上げる
///   （`promote_learned_inflection` と同じ条件。断られたら同じ経路で止まる）。
///
/// 送り仮名つきの形（`描き`）が辞書の `かき` に載っているものしか作らないので、
/// 活用の合わない入れ替え（`買き`）や、読みの割り戻しがずれた区間では何もしない。
pub fn offer_learned_stems(
    store: &DictStore,
    reading: &str,
    candidates: Vec<String>,
    confidence_gap: &dyn Fn(&str, &str) -> Option<f32>,
    allow_promote: bool,
) -> (Vec<String>, Vec<StemOffer>, Option<LearnedRewrite>) {
    let Some(first) = candidates.first().filter(|c| c.as_str() != reading).cloned() else {
        return (candidates, Vec::new(), None);
    };
    let Some(runs) = align(reading, &first) else {
        return (candidates, Vec::new(), None);
    };
    let first_chars: Vec<char> = first.chars().collect();
    // (入れ替えた文, 区間の読み, 入れ替え後の表記, 第 1 候補の表記, 別の漢字のほうが後か)
    let mut found: Vec<(StemOffer, String, bool)> = Vec::new();
    let mut spos = 0usize;
    for (i, run) in runs.iter().enumerate() {
        let start = spos;
        let run_len = run.surface.chars().count();
        spos += run_len;
        if found.len() >= STEM_MAX_OFFERS {
            break;
        }
        if run.kind != RunKind::Kanji || !run.surface.chars().all(is_kanji) {
            continue;
        }
        let Some(okuri) = runs
            .get(i + 1)
            .filter(|n| n.kind == RunKind::Hiragana)
            .and_then(|n| n.surface.chars().next())
        else {
            continue;
        };
        // 読み全体がその語なら merge 側の完全一致（辞書・学習）に任せる。
        if (start == 0 && run_len + 1 == first_chars.len())
            || INFLECTED_PARTICLE_TAILS.contains(&okuri)
        {
            continue;
        }
        let Some((own_at, rivals)) = stem_rivals(store, &run.reading, &run.surface) else {
            continue;
        };
        let key = format!("{}{}", run.reading, okuri);
        let llm_surface = format!("{}{}", run.surface, okuri);
        let mut known = store.lookup_dict(&key, STEM_DICT_LOOKUP_LIMIT);
        known.extend(store.lookup_user(&key));
        known.extend(store.lookup_user_low(&key));
        if !known.contains(&llm_surface) {
            continue;
        }
        for (kanji, at) in rivals {
            let surface = format!("{kanji}{okuri}");
            if !known.contains(&surface) {
                continue;
            }
            let mut sentence: String = first_chars[..start].iter().collect();
            sentence.push_str(&surface);
            sentence.extend(&first_chars[start + run_len + 1..]);
            found.push((
                StemOffer {
                    sentence,
                    key: key.clone(),
                    surface,
                },
                llm_surface.clone(),
                at > own_at,
            ));
            break;
        }
    }
    if found.is_empty() {
        return (candidates, Vec::new(), None);
    }

    let mut out = candidates;
    let mut rewrite = None;
    if allow_promote
        && let Some(n) = found.iter().position(|(offer, _, newer)| {
            *newer
                && out.contains(&offer.sentence)
                && confidence_gap(&first, &offer.sentence)
                    .is_some_and(|g| g <= INFLECTED_MAX_CONFIDENCE_GAP)
        })
    {
        let (offer, llm_surface, _) = found.remove(n);
        out.retain(|c| *c != offer.sentence);
        tracing::info!(
            reading = %reading,
            promoted = %offer.sentence,
            key = %offer.key,
            "rescore: promoted learned stem"
        );
        rewrite = Some(LearnedRewrite {
            original: first.clone(),
            runs: vec![(offer.key, llm_surface)],
        });
        out.insert(0, offer.sentence);
    }
    // 先頭（繰り上げたならその次の、元の第 1 候補）の直後に並べる。
    let base = out.iter().position(|c| *c == first).unwrap_or(0) + 1;
    let mut offers: Vec<StemOffer> = Vec::new();
    for (offer, _, _) in found {
        out.retain(|c| *c != offer.sentence);
        out.insert((base + offers.len()).min(out.len()), offer.sentence.clone());
        offers.push(offer);
    }
    tracing::debug!(reading = %reading, offers = ?offers, "rescore: offered learned stems");
    (out, offers, rewrite)
}

/// 長文の第 1 候補を学習表記で**書き換える**ための、読みの最小文字数。2 文字の読み
/// （`いま`・`にち`・`えん`）は同音異義が多く、文脈を無視して塗り替えると壊す。
const LEARNED_RUN_MIN_READING_CHARS: usize = 3;

/// 長文の第 1 候補を学習表記で**書き換える**ための、減衰済み確定回数の下限。
/// 「1 回選んだだけ」の語を文中の同音語すべてに波及させないための閾値。
const LEARNED_RUN_MIN_FREQ: f64 = 3.0;

/// 第 1 候補は変えずに、学習表記へ差し替えた候補を **2 番目に足す** ための下限。
///
/// 文中で出ないのはむしろ 2 文字の短い語（`ほお`→`頬`、`かわ`→`河`、`はい`→`牌`、
/// `まい`→`枚`）で、そのたびに単独で変換し直す必要があった。学習履歴は読み全体の
/// 完全一致で記録されるので、`ほお` の学習がある＝その読みを単独で変換して選んだ、
/// という明示の意思表示になる。そこで 1 回（減衰込みで直近 1 か月に 1 回＝半減期
/// 30 日）選んだ語から、Space をもう 1 回押せば届く位置に出す。
///
/// 第 1 候補を書き換えないのは、実ログ 1,624 文で試すと 2 文字の読みは誤爆が
/// 多かったため（`週と同じ`→`集と…`、`以上`/`異常` の取り違え等）。ライブ変換の
/// 表示と Space 1 回目の結果は変わらない。1 文字（`は`→`葉`）は偶然一致が多すぎる
/// ので除く。
const LEARNED_SOFT_MIN_READING_CHARS: usize = 2;
const LEARNED_SOFT_MIN_FREQ: f64 = 0.5;

/// 2 番目に足す差し替えを許す、直後のひらがなの先頭文字（助詞・助動詞の頭）。
///
/// 漢字 run の直後が活用語尾だと、その run は語幹であって単独の語ではない。
/// 実ログでは `入って`→`牌って`、`含んで`→`服んで`、`同じ`→`オナじ`、
/// `話しかけ`→`花しかけ`、`可愛い`→`河い` がこれで出た。助詞で切れている
/// ときだけ差し替える。
const SOFT_FOLLOWING_HEADS: &[char] = &[
    'が', 'を', 'に', 'は', 'の', 'で', 'と', 'も', 'へ', 'や', 'か', 'ね', 'よ', 'だ', 'ご',
];

/// 漢字かカタカナを含むか。学習表記が記号（`かっこ` → `「」`、`みぎ` → `→`）や
/// ひらがな・英字だけのときは、文中の語を置き換えない。
fn has_kanji_or_katakana(s: &str) -> bool {
    s.chars()
        .any(|c| matches!(c, 'ァ'..='ヶ' | '一'..='鿿' | '㐀'..='䶿' | '々'))
}

/// 学習表記への差し替えの強さ。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strength {
    /// 第 1 候補を書き換える。
    Strong,
    /// 第 1 候補は残し、差し替えた候補を 2 番目に足す。
    Soft,
}

/// run の前後が、語の切れ目として差し替えてよい形か（Soft 用）。
fn soft_boundary_ok(prev: Option<&AlignedRun>, next: Option<&AlignedRun>) -> bool {
    // 直前が接頭辞 `お`・`ご`（`お腹`・`ご飯`）なら語の途中。
    if let Some(p) = prev
        && p.kind == RunKind::Hiragana
        && p.surface.ends_with(['お', 'ご'])
    {
        return false;
    }
    match next {
        None => true,
        Some(n) if n.kind == RunKind::Hiragana => n
            .surface
            .chars()
            .next()
            .is_some_and(|c| SOFT_FOLLOWING_HEADS.contains(&c)),
        Some(_) => true,
    }
}

/// 表層 run 1 つを学習表記へ置き換えるべきなら、その表記と強さを返す。
fn learned_replacement(
    store: &DictStore,
    run: &AlignedRun,
    prev: Option<&AlignedRun>,
    next: Option<&AlignedRun>,
) -> Option<(String, Strength)> {
    if !matches!(run.kind, RunKind::Kanji | RunKind::Katakana) {
        return None;
    }
    let reading_chars = run.reading.chars().count();
    if reading_chars < LEARNED_SOFT_MIN_READING_CHARS {
        return None;
    }
    let learned = store.lookup_learn(&run.reading);
    let top = learned.first()?;
    // LLM と同じ表記を一度でも選んでいるなら、文脈で使い分けている語。触らない。
    if learned.iter().any(|s| s == &run.surface) {
        return None;
    }
    if store
        .lookup_user(&run.reading)
        .iter()
        .any(|s| s == &run.surface)
    {
        return None;
    }
    if !has_kanji_or_katakana(top) {
        return None;
    }
    let freq = store.learn_freq(&run.reading, top);
    if reading_chars >= LEARNED_RUN_MIN_READING_CHARS && freq >= LEARNED_RUN_MIN_FREQ {
        return Some((top.clone(), Strength::Strong));
    }
    if freq >= LEARNED_SOFT_MIN_FREQ && soft_boundary_ok(prev, next) {
        return Some((top.clone(), Strength::Soft));
    }
    None
}

/// 学習語を、長文の候補の途中にも効かせる。
///
/// 学習履歴も辞書と同じく読み全体の完全一致でしか引かれないので、
/// `みかん → 美柑` を何十回確定していても、`みかんのへやのはいけい` と伸びた
/// 瞬間に効かなくなる。`promote_dict_agreeing` は n-best の中から選び直すだけ
/// なので、LLM が `美柑` を一度も出さなければ救えない。
///
/// そこで第 1 候補を読みへ割り戻し、漢字・カタカナ run の読みが学習キーと一致し、
/// かつ LLM とは別の表記を選んでいるなら、その run を学習表記へ差し替える。
///
/// - 何度も選んでいる長い語（`Strength::Strong`）は、差し替えた候補を
///   **先頭に足す**。元の第 1 候補は 2 番目に残るので 1 打鍵で戻せる。
/// - 短い語や選んだ回数が少ない語（`Strength::Soft`）は、第 1 候補は変えず、
///   差し替えた候補を **2 番目に足す**。
///
/// 先頭を書き換えたのに断られたら（元の候補が確定されたら）、その読みでは LLM の
/// 表記も使うと学習させる。次からは `learned_replacement` の「LLM と同じ表記を
/// 選んだことがある」で止まる（`しんちょう → 身長` を覚えていても、文中の `慎重`
/// を塗り替え続けないため）。2 番目に足しただけの差し替えは、元の候補の確定を
/// 「断った」とは見なさない（見ずに Space 1 回で確定しただけかもしれない）ので、
/// `LearnedRewrite` には入れない。
///
/// 候補が 0 件のときは何もしない（件数が増えるのは実候補がある時だけ）。
pub fn apply_learned_runs(
    store: &DictStore,
    reading: &str,
    candidates: Vec<String>,
) -> (Vec<String>, Option<LearnedRewrite>) {
    let Some(first) = candidates.first().cloned() else {
        return (candidates, None);
    };
    let Some(runs) = align(reading, &first) else {
        return (candidates, None);
    };
    if runs.len() < 2 {
        // 読み全体が 1 run なら完全一致の学習が merge 側で効く。
        return (candidates, None);
    }

    let mut strong_runs: Vec<(String, String)> = Vec::new();
    let mut any_soft = false;
    // Strong だけを差し替えた表記と、Soft も含めて差し替えた表記。
    let mut strong_text = String::new();
    let mut soft_text = String::new();
    for (i, run) in runs.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| &runs[j]);
        let next = runs.get(i + 1);
        match learned_replacement(store, run, prev, next) {
            Some((s, Strength::Strong)) => {
                strong_text.push_str(&s);
                soft_text.push_str(&s);
                strong_runs.push((run.reading.clone(), run.surface.clone()));
            }
            Some((s, Strength::Soft)) => {
                strong_text.push_str(&run.surface);
                soft_text.push_str(&s);
                any_soft = true;
            }
            None => {
                strong_text.push_str(&run.surface);
                soft_text.push_str(&run.surface);
            }
        }
    }
    if strong_runs.is_empty() && !any_soft {
        return (candidates, None);
    }

    tracing::info!(
        reading = %reading,
        from = %first,
        strong = %strong_text,
        soft = %soft_text,
        "rescore: applied learned runs"
    );
    let rewrite = (!strong_runs.is_empty()).then(|| LearnedRewrite {
        original: first,
        runs: strong_runs,
    });

    let mut out: Vec<String> = Vec::with_capacity(candidates.len() + 2);
    let mut push = |c: String| {
        if !out.contains(&c) {
            out.push(c);
        }
    };
    let mut rest = candidates.into_iter();
    if rewrite.is_some() {
        push(strong_text);
    } else if let Some(c) = rest.next() {
        push(c);
    }
    if any_soft {
        push(soft_text);
    }
    for c in rest {
        push(c);
    }
    (out, rewrite)
}

/// 文中の登録語を差し替える対象にする、読みの最小・最大文字数。
///
/// 短い読みは別の語の並びに偶然現れやすい（`きょうか`・`はいてい`）。5 文字以上の
/// 登録語は固有名詞がほとんどで、文中に偶然現れることはまず無い。
const USER_WORD_MIN_READING_CHARS: usize = 5;
const USER_WORD_MAX_READING_CHARS: usize = 24;

/// LLM の表記が一般語かどうかを MOZC 辞書で確かめるときの候補上限。
const USER_WORD_DICT_LOOKUP_LIMIT: usize = 256;

/// 読みの文字位置 → 表層の文字位置。その位置で表層を切れないときは `None`。
///
/// かな・英数の run は 1 文字ずつ対応するので途中でも切れる。漢字 run は読みの
/// 内訳が分からないので両端でしか切れない。
fn reading_cut_points(runs: &[AlignedRun], reading_chars: usize) -> Vec<Option<usize>> {
    let mut cuts: Vec<Option<usize>> = vec![None; reading_chars + 1];
    let (mut rpos, mut spos) = (0usize, 0usize);
    for run in runs {
        let rn = run.reading.chars().count();
        let sn = run.surface.chars().count();
        cuts[rpos] = Some(spos);
        if run.kind != RunKind::Kanji && rn == sn {
            for k in 1..rn {
                cuts[rpos + k] = Some(spos + k);
            }
        }
        rpos += rn;
        spos += sn;
    }
    cuts[rpos] = Some(spos);
    cuts
}

/// ユーザー辞書の登録語を、長文の候補の途中にも効かせる。
///
/// `apply_learned_runs` は表層の run 1 つぶんしか見ないので、LLM が登録語を
/// 複数の run に割って出すと拾えない。`はいていじゃんそう → 海底雀荘` を登録して
/// いても、文中では `ハイテイ雀荘`（カタカナ＋漢字）や `履いていじゃんそう`
/// （漢字＋かな）になり、どの run の読みも登録語と一致しない。
///
/// そこで読みの側から登録語を探し、その区間に当たる表層を登録表記へ差し替えた
/// 候補を **先頭に足す**。元の第 1 候補は 2 番目に残る。断られたら
/// （元の候補が確定されたら）`LearnedRewrite` 経由でその読みに LLM の表記を
/// 学習し、次からは差し替えない。
///
/// 候補が 0 件のときは何もしない。読み全体が登録語なら merge 側の完全一致に任せる。
pub fn apply_user_words(
    store: &DictStore,
    reading: &str,
    candidates: Vec<String>,
) -> (Vec<String>, Option<LearnedRewrite>) {
    let Some(first) = candidates.first().cloned() else {
        return (candidates, None);
    };
    let reading_chars: Vec<char> = reading.chars().collect();
    let n = reading_chars.len();
    let mut hits = store.user_words_within(
        reading,
        USER_WORD_MIN_READING_CHARS,
        USER_WORD_MAX_READING_CHARS,
    );
    hits.retain(|(start, end, _)| end - start < n);
    if hits.is_empty() {
        return (candidates, None);
    }
    let Some(runs) = align(reading, &first) else {
        return (candidates, None);
    };
    let cuts = reading_cut_points(&runs, n);
    let surface_chars: Vec<char> = first.chars().collect();

    // 長い登録語を優先し、重なる区間は捨てる。
    hits.sort_by_key(|(start, end, _)| (std::cmp::Reverse(end - start), *start));
    let mut picked: Vec<(usize, usize, usize, usize, String)> = Vec::new();
    for (start, end, surfaces) in hits {
        if picked.iter().any(|p| start < p.1 && p.0 < end) {
            continue;
        }
        let (Some(s), Some(e)) = (cuts[start], cuts[end]) else {
            continue;
        };
        let span_reading: String = reading_chars[start..end].iter().collect();
        let span_surface: String = surface_chars[s..e].iter().collect();
        let learned = store.lookup_learn(&span_reading);
        // 登録表記のどれか、または自分で選んだことのある表記なら触らない。
        if surfaces.contains(&span_surface) || learned.contains(&span_surface) {
            continue;
        }
        // LLM がその読みの一般語を 1 語で出しているなら同音異義。登録語で塗り替えない
        // （`とうじょう → 東條` を登録していても、文中の `登場` はそのまま）。
        // 差し替えるのは、登録語がばらばらに割れて出たときだけ。
        if store
            .lookup_dict(&span_reading, USER_WORD_DICT_LOOKUP_LIMIT)
            .contains(&span_surface)
        {
            continue;
        }
        // 登録表記が複数あるときは、よく選んでいるものを使う。
        let word = learned
            .iter()
            .find(|l| surfaces.contains(l))
            .unwrap_or(&surfaces[0]);
        if !has_kanji_or_katakana(word) {
            continue;
        }
        picked.push((start, end, s, e, word.clone()));
    }
    if picked.is_empty() {
        return (candidates, None);
    }

    picked.sort_by_key(|p| p.2);
    let mut rewritten = String::new();
    let mut declined: Vec<(String, String)> = Vec::new();
    let mut pos = 0usize;
    for (start, end, s, e, word) in &picked {
        rewritten.extend(&surface_chars[pos..*s]);
        rewritten.push_str(word);
        declined.push((
            reading_chars[*start..*end].iter().collect(),
            surface_chars[*s..*e].iter().collect(),
        ));
        pos = *e;
    }
    rewritten.extend(&surface_chars[pos..]);

    tracing::info!(
        reading = %reading,
        from = %first,
        to = %rewritten,
        "rescore: applied user words"
    );
    let mut out: Vec<String> = Vec::with_capacity(candidates.len() + 1);
    out.push(rewritten);
    for c in candidates {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    (
        out,
        Some(LearnedRewrite {
            original: first,
            runs: declined,
        }),
    )
}

/// `apply_learned_runs` が差し替えた内容。`original` が確定されたら、
/// `runs` の `(読み, LLM の表記)` を学習して次から差し替えないようにする。
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedRewrite {
    pub original: String,
    pub runs: Vec<(String, String)>,
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

    #[test]
    fn scores_kanji_compound_by_its_dictionary_words() {
        let (_dir, store) = store_with_user_entries(
            r#"
[[entries]]
reading = "まーじゃん"
surfaces = ["マージャン", "麻雀"]

[[entries]]
reading = "ようご"
surfaces = ["用語"]
"#,
        );
        let reading = "まーじゃんようごがぜんぜんへんかんできない";
        // `麻雀用語` は 1 つの漢字 run になるが、`麻雀`＋`用語` として採点されるので
        // カタカナ表記に追い越されない。
        let cands = vec![
            "麻雀用語が全然変換できない".to_string(),
            "マージャン用語が全然変換できない".to_string(),
        ];
        let got = promote_dict_agreeing(&store, reading, cands.clone(), 12, 24.0);
        assert_eq!(got, cands);
    }

    #[test]
    fn applies_learned_word_inside_long_reading() {
        let (_dir, store) = store_with_user_entries("");
        let reading = "みかんのへやのはいけい";
        let cands = vec![
            "蜜柑の部屋の背景".to_string(),
            "みかんの部屋の背景".to_string(),
        ];

        // 学習が無ければ触らない。
        let (got, rewrite) = apply_learned_runs(&store, reading, cands.clone());
        assert_eq!(got, cands);
        assert!(rewrite.is_none());

        // 1 回選んだだけなら先頭は変えず、2 番目に足す（断りの学習もしない）。
        store.learn_force("みかん", "美柑");
        let (got, rewrite) = apply_learned_runs(&store, reading, cands.clone());
        assert!(rewrite.is_none());
        assert_eq!(
            got,
            vec![
                "蜜柑の部屋の背景".to_string(),
                "美柑の部屋の背景".to_string(),
                "みかんの部屋の背景".to_string(),
            ]
        );

        // 何度も選んでいれば先頭を書き換える。
        for _ in 0..3 {
            store.learn_force("みかん", "美柑");
        }
        let (got, rewrite) = apply_learned_runs(&store, reading, cands);
        assert_eq!(
            rewrite.unwrap().runs,
            vec![("みかん".to_string(), "蜜柑".to_string())]
        );
        assert_eq!(
            got,
            vec![
                "美柑の部屋の背景".to_string(),
                "蜜柑の部屋の背景".to_string(),
                "みかんの部屋の背景".to_string(),
            ]
        );
    }

    #[test]
    fn keeps_llm_surface_the_user_has_also_chosen() {
        let (_dir, store) = store_with_user_entries("");
        for _ in 0..4 {
            store.learn_force("さくら", "サクラ");
        }
        store.learn_force("さくら", "桜");
        store.learn_force("さくら", "サクラ");
        let cands = vec!["桜が咲いた".to_string()];
        let (got, _) = apply_learned_runs(&store, "さくらがさいた", cands.clone());
        assert_eq!(got, cands);
    }

    #[test]
    fn offers_two_char_learned_word_as_second_candidate() {
        let (_dir, store) = store_with_user_entries("");
        // 何度選んでいても、2 文字の読みは先頭を書き換えない。
        for _ in 0..5 {
            store.learn_force("かわ", "河");
        }
        let cands = vec!["皮に捨てる".to_string(), "かわに捨てる".to_string()];
        let (got, rewrite) = apply_learned_runs(&store, "かわにすてる", cands);
        assert!(rewrite.is_none());
        assert_eq!(
            got,
            vec![
                "皮に捨てる".to_string(),
                "河に捨てる".to_string(),
                "かわに捨てる".to_string(),
            ]
        );
    }

    #[test]
    fn offers_learned_word_for_katakana_run_before_kanji() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("まい", "枚");
        let cands = vec!["マイ程度のフォルダ".to_string()];
        let (got, _) = apply_learned_runs(&store, "まいていどのふぉるだ", cands);
        assert_eq!(
            got,
            vec![
                "マイ程度のフォルダ".to_string(),
                "枚程度のフォルダ".to_string(),
            ]
        );
    }

    #[test]
    fn skips_verb_stem_followed_by_inflection() {
        // 実ログで出た誤爆: `入って`→`牌って`、`含んで`→`服んで`、`同じ`→`オナじ`。
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("はい", "牌");
        store.learn_force("ふく", "服");
        store.learn_force("おな", "オナ");
        for (reading, surface) in [
            ("ちからははいっていない", "力は入っていない"),
            ("くちでふくんで", "口で含んで"),
            ("おなじように", "同じように"),
        ] {
            let cands = vec![surface.to_string()];
            let (got, rewrite) = apply_learned_runs(&store, reading, cands.clone());
            assert!(rewrite.is_none(), "{reading}");
            assert_eq!(got, cands, "{reading}");
        }
    }

    #[test]
    fn skips_run_after_honorific_prefix() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("なか", "中");
        let cands = vec!["お腹が空いた".to_string()];
        let (got, _) = apply_learned_runs(&store, "おなかがすいた", cands.clone());
        assert_eq!(got, cands);
    }

    #[test]
    fn offers_learned_word_before_particle_or_at_end() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("ふく", "服");
        let cands = vec!["その腹ごと".to_string()];
        let (got, _) = apply_learned_runs(&store, "そのふくごと", cands);
        assert_eq!(got[1], "その服ごと");

        let cands = vec!["根元まで口で腹".to_string()];
        let (got, _) = apply_learned_runs(&store, "ねもとまでくちでふく", cands);
        assert_eq!(got[1], "根元まで口で服");
    }

    #[test]
    fn applies_user_word_split_across_runs() {
        let (_dir, store) = store_with_user_entries(
            r#"
[[entries]]
reading = "はいていじゃんそう"
surfaces = ["海底雀荘"]

[[entries]]
reading = "はいてい"
surfaces = ["海底", "ハイテイ"]
"#,
        );
        let reading = "もちべはあきらかにはいていじゃんそうにかたむいている";
        // 実ログで出た 2 つの割れ方: カタカナ＋漢字、漢字＋かな。
        for llm in [
            "モチベは明らかにハイテイ雀荘に傾いている",
            "モチベは明らかに履いていじゃんそうに傾いている",
        ] {
            let (got, rewrite) = apply_user_words(&store, reading, vec![llm.to_string()]);
            assert_eq!(
                got,
                vec![
                    "モチベは明らかに海底雀荘に傾いている".to_string(),
                    llm.to_string(),
                ]
            );
            assert_eq!(rewrite.unwrap().original, llm);
        }

        // LLM が登録表記を出していれば触らない。
        let cands = vec!["モチベは明らかに海底雀荘に傾いている".to_string()];
        let (got, rewrite) = apply_user_words(&store, reading, cands.clone());
        assert_eq!(got, cands);
        assert!(rewrite.is_none());

        // 読み全体が登録語なら merge 側の完全一致に任せる。
        let cands = vec!["ハイテイ雀荘".to_string()];
        let (got, _) = apply_user_words(&store, "はいていじゃんそう", cands.clone());
        assert_eq!(got, cands);

        // 断られた表記を覚えたら、次からは差し替えない。
        store.learn_force("はいていじゃんそう", "ハイテイ雀荘");
        let cands = vec!["モチベは明らかにハイテイ雀荘に傾いている".to_string()];
        let (got, _) = apply_user_words(&store, reading, cands.clone());
        assert_eq!(got, cands);
    }

    #[test]
    fn promotes_learned_inflection_from_nbest() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("おもい", "重い");
        let reading = "どうさがおもい";
        let cands = vec![
            "動作が思い".to_string(),
            "動作が重い".to_string(),
            "動作がおもい".to_string(),
        ];
        let close = |_: &str, _: &str| Some(0.17f32);

        let (got, rewrite) = promote_learned_inflection(&store, reading, cands.clone(), &close);
        assert_eq!(got[0], "動作が重い");
        assert_eq!(got[1], "動作が思い");
        assert_eq!(got.len(), 3);
        let rewrite = rewrite.unwrap();
        assert_eq!(rewrite.original, "動作が思い");
        assert_eq!(rewrite.runs, vec![("おもい".to_string(), "思い".to_string())]);

        // LLM がはっきり選んでいる（差が大きい・自信度が不明）なら触らない。
        let far = |_: &str, _: &str| Some(1.0f32);
        let (got, _) = promote_learned_inflection(&store, reading, cands.clone(), &far);
        assert_eq!(got, cands);
        let unknown = |_: &str, _: &str| None;
        let (got, _) = promote_learned_inflection(&store, reading, cands.clone(), &unknown);
        assert_eq!(got, cands);

        // n-best に無い表記は作らない。
        let only = vec!["動作が思い".to_string(), "動作がおもい".to_string()];
        let (got, _) = promote_learned_inflection(&store, reading, only.clone(), &close);
        assert_eq!(got, only);

        // 最後に LLM と同じ表記を選んでいれば触らない（断られた後の状態）。
        // 同じ秒の確定は回数で並ぶので 2 回選ぶ。
        store.learn_force("おもい", "思い");
        store.learn_force("おもい", "思い");
        let (got, rewrite) = promote_learned_inflection(&store, reading, cands.clone(), &close);
        assert_eq!(got, cands);
        assert!(rewrite.is_none());
    }

    #[test]
    fn skips_learned_word_with_particle_as_inflection() {
        // 実ログで出た誤爆: `あしの → 足の` の学習で `脚のある家具` を塗り替えた。
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("あしの", "足の");
        let cands = vec!["脚のある家具".to_string(), "足のある家具".to_string()];
        let close = |_: &str, _: &str| Some(0.09f32);
        let (got, _) = promote_learned_inflection(&store, "あしのあるかぐ", cands.clone(), &close);
        assert_eq!(got, cands);
    }

    const KAKI_DICT: &str = r#"
[[entries]]
reading = "かき"
surfaces = ["書き", "描き", "下記", "買き"]
"#;

    #[test]
    fn offers_stem_rival_as_second_candidate_across_inflections() {
        let (_dir, store) = store_with_user_entries(KAKI_DICT);
        // 選び分けたのは `かく` だけ。`かき` は一度も学習していない。
        store.learn_force("かく", "描く");
        store.learn_force("かく", "書く");
        let reading = "くらげがおおすぎるからかきなおそう";
        let cands = vec![
            "クラゲが多すぎるから書き直そう".to_string(),
            "くらげが多すぎるから書き直そう".to_string(),
        ];
        let close = |_: &str, _: &str| Some(0.1f32);

        // n-best に無ければ 2 番目に足す。先頭は動かさない。
        let (got, offers, rewrite) =
            offer_learned_stems(&store, reading, cands.clone(), &close, true);
        assert_eq!(
            got,
            [
                "クラゲが多すぎるから書き直そう",
                "クラゲが多すぎるから描き直そう",
                "くらげが多すぎるから書き直そう",
            ]
        );
        assert!(rewrite.is_none());
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].key, "かき");
        assert_eq!(offers[0].surface, "描き");

        // n-best の下位にあれば 2 番目へ移す（件数は増やさない）。もう一度通しても同じ。
        let mut with_alt = cands.clone();
        with_alt.push("クラゲが多すぎるから描き直そう".to_string());
        let (moved, _, _) = offer_learned_stems(&store, reading, with_alt, &close, false);
        assert_eq!(moved, got);
        let (again, _, _) = offer_learned_stems(&store, reading, moved, &close, true);
        assert_eq!(again, got);
    }

    #[test]
    fn promotes_stem_rival_chosen_later_when_llm_is_unsure() {
        let (_dir, store) = store_with_user_entries(KAKI_DICT);
        store.learn_force("かく", "書く");
        // 最終確定時刻は秒単位。`描` のほうを後に選んだ状態にする。
        std::thread::sleep(std::time::Duration::from_millis(1100));
        store.learn_force("かいた", "描いた");
        store.learn_force("かく", "描く");
        let reading = "えをかきなおそう";
        let cands = vec![
            "絵を書き直そう".to_string(),
            "えを書き直そう".to_string(),
            "絵を描き直そう".to_string(),
        ];
        let close = |_: &str, _: &str| Some(0.2f32);
        let (got, offers, rewrite) =
            offer_learned_stems(&store, reading, cands.clone(), &close, true);
        assert_eq!(got, ["絵を描き直そう", "絵を書き直そう", "えを書き直そう"]);
        assert!(offers.is_empty());
        let rewrite = rewrite.unwrap();
        assert_eq!(rewrite.original, "絵を書き直そう");
        assert_eq!(rewrite.runs, vec![("かき".to_string(), "書き".to_string())]);

        // LLM がはっきり選んでいる・繰り上げ不可のときは 2 番目に置くだけ。
        let far = |_: &str, _: &str| Some(1.0f32);
        let expect = ["絵を書き直そう", "絵を描き直そう", "えを書き直そう"];
        let (got, _, rewrite) = offer_learned_stems(&store, reading, cands.clone(), &far, true);
        assert_eq!(got, expect);
        assert!(rewrite.is_none());
        let (got, _, rewrite) = offer_learned_stems(&store, reading, cands.clone(), &close, false);
        assert_eq!(got, expect);
        assert!(rewrite.is_none());
    }

    #[test]
    fn skips_stem_never_chosen_under_the_same_reading() {
        let (_dir, store) = store_with_user_entries(KAKI_DICT);
        // `買` は語幹の読みが同じでも、`書` と同じ読みキーで選び分けたことが無い。
        store.learn_force("かく", "書く");
        store.learn_force("かう", "買う");
        let cands = vec!["絵を書き直そう".to_string()];
        let close = |_: &str, _: &str| Some(0.1f32);
        let (got, offers, _) =
            offer_learned_stems(&store, "えをかきなおそう", cands.clone(), &close, true);
        assert_eq!(got, cands);
        assert!(offers.is_empty());

        // 選び分けていても、その活用形が辞書に無ければ作らない。
        store.learn_force("かく", "描く");
        let cands = vec!["絵を書こう".to_string()];
        let (got, _, _) = offer_learned_stems(&store, "えをかこう", cands.clone(), &close, true);
        assert_eq!(got, cands);
    }

    #[test]
    fn skips_one_char_reading() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("は", "葉");
        let cands = vec!["歯が痛い".to_string()];
        let (got, rewrite) = apply_learned_runs(&store, "はがいたい", cands.clone());
        assert!(rewrite.is_none());
        assert_eq!(got, cands);
    }
}
