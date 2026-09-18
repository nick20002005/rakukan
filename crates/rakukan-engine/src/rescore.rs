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
use std::collections::HashMap;

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
    fn skips_one_char_reading() {
        let (_dir, store) = store_with_user_entries("");
        store.learn_force("は", "葉");
        let cands = vec!["歯が痛い".to_string()];
        let (got, rewrite) = apply_learned_runs(&store, "はがいたい", cands.clone());
        assert!(rewrite.is_none());
        assert_eq!(got, cands);
    }
}
