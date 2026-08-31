//! rakukan 変換エンジン
//!
//! karukan-engine のコードを直接統合したクレート。
//! 外部 git 依存なし。
//!
//! ```text
//! ローマ字 → RomajiConverter → ひらがな → (1) 辞書引き（同期）
//!                                          (2) KanaKanjiConverter（LLM, 非同期）
//!                                          → 候補マージ → 返却
//! ```

// ── 統合した karukan-engine モジュール ────────────────────────────────────────
pub mod kana;
pub mod kanji;
pub mod romaji;

pub use kana::{
    hiragana_to_halfwidth_katakana, hiragana_to_katakana, katakana_to_hiragana, normalize_nfkc,
};
pub use kanji::{Backend, KanaKanjiConverter};
pub use romaji::{BackspaceResult, ConversionEvent, RomajiConverter};

// ── rakukan 独自モジュール ────────────────────────────────────────────────────
pub mod backend;
pub mod conv_cache;
pub mod dict;
pub mod digits;
pub mod ffi;
pub mod segments;
pub use backend::{BackendSelection, GpuInfo, select_backend};
// Backend は kanji::Backend と名前が被るため、rakukan の Backend は別名でエクスポート
pub use backend::Backend as RakunBackend;

pub use segments::{Candidate, CandidateSource, Segment, Segments};

pub use rakukan_dict::mozc_dict::MozcDict;
pub use rakukan_dict::{DictStore, find_mozc_dict, user_dict_path};

use kanji::{Backend as KarukanBackend, registry};
use thiserror::Error;
use tracing::{debug, info};

// ── コンテキストトリミング ────────────────────────────────────────────────────

/// context への追加を拒否する、ひらがな（+長音・中点）の最小文字数。
/// 変換時の strip（backend.rs の `ECHO_RUN_MIN_CHARS` = 8）より低く設定する:
/// 「きもちは、」のような短いひらがな確定も、同じ読みの再変換でエコーを誘発するため。
const CONTEXT_ECHO_MIN_HIRAGANA_CHARS: usize = 4;

/// context に入れると LLM のエコーアトラクタ（変換ではなく context からの
/// コピー）を誘発するテキストか判定する。
///
/// 対象は「未変換のまま確定された」ひらがな文: 句読点・空白を除いた全文字が
/// ひらがな（+長音・中点）で、その数が `CONTEXT_ECHO_MIN_HIRAGANA_CHARS` 以上。
/// 漢字・カタカナ・英数字を 1 文字でも含むテキストは変換済みとみなして通す
/// （カタカナ確定はエコーしても正しい出力になるため対象外。混在汚染は
/// backend.rs の `strip_echo_context` が保険として捕捉する）。
fn is_context_echo_risk(text: &str) -> bool {
    let mut hiragana_count = 0usize;
    for c in text.chars() {
        let n = c as u32;
        if (0x3041..=0x3096).contains(&n) || c == 'ー' || c == '・' {
            hiragana_count += 1;
        } else if matches!(
            c,
            '、' | '。'
                | '！'
                | '？'
                | '!'
                | '?'
                | '.'
                | '．'
                | '，'
                | ','
                | '\n'
                | ' '
                | '\u{3000}'
                | '「'
                | '」'
                | '『'
                | '』'
                | '（'
                | '）'
                | '('
                | ')'
        ) {
            // 句読点・括弧・空白は無視
        } else {
            // 漢字・カタカナ・英数字等を含む → 変換済みテキストとみなす
            return false;
        }
    }
    hiragana_count >= CONTEXT_ECHO_MIN_HIRAGANA_CHARS
}

/// テキストから末尾 `n` 文の開始バイト位置を返す。
///
/// fast-bunkai の BasicRule / LinebreakAnnotator 相当の純 Rust 実装。
/// 文境界は `。！？!?.．\n` の直後とみなす。
/// 文境界が `n` 個未満の場合はテキスト全体の先頭（0）を返す。
fn last_n_sentences_start(text: &str, n: usize) -> usize {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let len = chars.len();
    let mut boundaries: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < len {
        let ch = chars[i].1;
        if matches!(
            ch,
            '\u{3002}' | '\u{FF01}' | '\u{FF1F}' | '!' | '?' | '.' | '\u{FF0E}' | '\n'
        ) {
            // 句読点・空白が連続する場合はまとめてスキップ
            let mut j = i + 1;
            while j < len
                && matches!(
                    chars[j].1,
                    '\u{3002}'
                        | '\u{FF01}'
                        | '\u{FF1F}'
                        | '!'
                        | '?'
                        | '.'
                        | '\u{FF0E}'
                        | ' '
                        | '\u{3000}'
                        | '\n'
                )
            {
                j += 1;
            }
            if j < len {
                boundaries.push(chars[j].0);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    // 末尾から n 個目の境界を返す。境界が足りなければ先頭。
    if boundaries.len() >= n {
        boundaries[boundaries.len() - n]
    } else {
        0
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("エンジン初期化失敗: {0}")]
    InitFailed(String),
    #[error("変換エラー: {0}")]
    ConversionFailed(String),
    #[error("モデル未初期化（init_kanji() を先に呼んでください）")]
    ModelNotInitialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigitWidth {
    Fullwidth,
    Halfwidth,
}

impl Default for DigitWidth {
    fn default() -> Self {
        DigitWidth::Halfwidth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaWidth {
    Fullwidth,
    Halfwidth,
}

impl Default for AlphaWidth {
    fn default() -> Self {
        AlphaWidth::Fullwidth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolWidth {
    Fullwidth,
    Halfwidth,
}

impl Default for SymbolWidth {
    fn default() -> Self {
        SymbolWidth::Fullwidth
    }
}

fn default_digit_separator_auto() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigitCandidateKind {
    Arabic,
    Fullwidth,
    Positional,
    PerDigit,
    Daiji,
}

pub fn default_digit_candidates_order() -> Vec<DigitCandidateKind> {
    vec![
        DigitCandidateKind::Arabic,
        DigitCandidateKind::Fullwidth,
        DigitCandidateKind::Positional,
        DigitCandidateKind::PerDigit,
        DigitCandidateKind::Daiji,
    ]
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub model_variant: Option<String>,
    pub num_candidates: usize,
    pub n_threads: u32,
    /// GPU レイヤー数 (u32::MAX = 全レイヤー, 0 = CPU のみ)
    pub n_gpu_layers: u32,
    /// 使用する GPU インデックス (0 = 最初の GPU, -1 = 自動)
    pub main_gpu: i32,
    /// 数字の入力幅: "fullwidth" = 全角 (０１２), "halfwidth" = 半角 (012)
    pub digit_width: DigitWidth,
    /// 英字の入力幅: "fullwidth" = 全角 (ＡＢＣ), "halfwidth" = 半角 (ABC)
    #[serde(default)]
    pub alpha_width: AlphaWidth,
    /// 記号の入力幅: "fullwidth" = 全角 (＠＃), "halfwidth" = 半角 (@#)
    #[serde(default)]
    pub symbol_width: SymbolWidth,
    /// 数字直後の句読点を数値区切りとして扱う。
    #[serde(default = "default_digit_separator_auto")]
    pub digit_separator_auto: bool,
    /// 数字だけの reading に対して提示する候補種別と順序。
    #[serde(default = "default_digit_candidates_order")]
    pub digit_candidates_order: Vec<DigitCandidateKind>,
    /// ライブ変換時の候補数（beam 幅に影響）。1 = greedy（高速）、3 = beam（高品質）
    pub live_conv_beam_size: usize,
    /// Space 変換時のビーム幅の**上限**（num_candidates と併せて min をとる）。
    /// デフォルト 30 では実質上限なし、num_candidates がそのまま beam 幅になる。
    pub convert_beam_size: usize,
    /// 異常変換の棄却に使う「最良候補からの平均 log-prob 差」の許容幅 (nats/token)。
    /// `null` で無効。既定 3.0 は寛容で、明らかな外れ値候補のみ落とす。
    /// 詳細は `kanji::ConversionConfig::confidence_margin` を参照。
    #[serde(default = "default_confidence_margin")]
    pub confidence_margin: Option<f32>,
    /// 最良候補の平均 log-prob (nats/token) の絶対下限。これを下回る変換は幻覚の
    /// 可能性が高いため全候補を捨て、かなにフォールバックする。`null`（既定）で無効。
    /// 詳細は `kanji::ConversionConfig::min_top_confidence` を参照。
    #[serde(default)]
    pub min_top_confidence: Option<f32>,
    /// 短文予測（学習済みフレーズの前方一致予測）を有効にする。
    #[serde(default = "default_prediction_enabled")]
    pub prediction_enabled: bool,
    /// 1 回の候補リストに差し込む短文予測の最大件数。
    #[serde(default = "default_prediction_max_candidates")]
    pub prediction_max_candidates: usize,
    /// 短文予測を開始する読みの最小文字数。
    #[serde(default = "default_prediction_min_reading_chars")]
    pub prediction_min_reading_chars: usize,
}

fn is_kana_or_cjk(c: char) -> bool {
    matches!(c,
        '\u{3041}'..='\u{309f}'   // ひらがな
        | '\u{30a0}'..='\u{30ff}' // カタカナ
        | '\u{3400}'..='\u{4dbf}' // CJK 拡張A
        | '\u{4e00}'..='\u{9fff}' // CJK 統合漢字
        | '\u{f900}'..='\u{faff}' // CJK 互換漢字
        | '\u{ff66}'..='\u{ff9f}' // 半角カタカナ
    )
}

/// 「記号だけでできている候補」か。
///
/// MOZC 辞書には 1 つの読みに記号がまとめて登録されていることがあり、
/// 「たんい」は ¢ £ ¤ ¥ ° ‰ ′ ″ ₠… だけで 50 件を占める。辞書候補を
/// 素直に前へ並べると表示スロット（既定 8）が記号で埋まり、LLM が返す
/// 「単位」が 1 件も入らない。記号だけの候補は LLM の後ろへ回す。
///
/// ASCII 英数字だけの候補（"PC" など）は語として扱う。"°C" のように
/// 記号が混ざるものは記号側。
fn is_symbol_only_candidate(s: &str) -> bool {
    !s.is_empty()
        && !s.chars().any(is_kana_or_cjk)
        && !s.chars().all(|c| c.is_ascii_alphanumeric())
}

/// 「ん」補完を試す読みの最小文字数。
const N_INSERTION_MIN_READING_CHARS: usize = 3;
/// 1 回の変換で試す代替読みの上限。
const N_INSERTION_MAX_READINGS: usize = 4;
/// 代替読みから拾う候補の上限。
const N_INSERTION_MAX_CANDIDATES: usize = 3;

/// な行かなを「ん + 母音」に開いた代替読みを列挙する。
///
/// ローマ字入力では `n` + 母音 が な行になるので、「げんいん」を出すには
/// `gennin` と n を 2 回打つ必要がある。1 回で済ませると「げにん」になり、
/// 目的の語が候補に出てこない（原因 / 雰囲気 / 恋愛 / 千円 / 全員 / 金曜 …）。
///
/// 先頭のかなは対象外（「ん」で始まる読みは作らない）。
/// 2 文字以下の読みも対象外。「たに」を「たんい」に開くのは踏み込みすぎで、
/// 谷 のような正当な変換の後ろに無関係な候補を足すだけになる。
fn n_insertion_readings(reading: &str) -> Vec<String> {
    let chars: Vec<char> = reading.chars().collect();
    if chars.len() < N_INSERTION_MIN_READING_CHARS {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut i = 1;
    while i < chars.len() {
        // にゃ/にゅ/にょ は 2 文字まとめて「んや/んゆ/んよ」に開く（きにょう → きんよう）
        let (consumed, vowel) = match (chars[i], chars.get(i + 1)) {
            ('に', Some('ゃ')) => (2, 'や'),
            ('に', Some('ゅ')) => (2, 'ゆ'),
            ('に', Some('ょ')) => (2, 'よ'),
            ('な', _) => (1, 'あ'),
            ('に', _) => (1, 'い'),
            ('ぬ', _) => (1, 'う'),
            ('ね', _) => (1, 'え'),
            ('の', _) => (1, 'お'),
            _ => {
                i += 1;
                continue;
            }
        };
        let mut alt: String = chars[..i].iter().collect();
        alt.push('ん');
        alt.push(vowel);
        alt.extend(chars[i + consumed..].iter());
        out.push(alt);
        if out.len() >= N_INSERTION_MAX_READINGS {
            break;
        }
        i += consumed;
    }
    out
}

/// ASCII 図形文字 (U+0021..U+007E) を全角形 (U+FF01..U+FF5E) に写す。
fn ascii_to_fullwidth(c: char) -> char {
    if ('!'..='~').contains(&c) {
        char::from_u32(c as u32 + 0xFEE0).unwrap_or(c)
    } else {
        c
    }
}

fn default_confidence_margin() -> Option<f32> {
    Some(3.0)
}

fn default_prediction_enabled() -> bool {
    true
}

fn default_prediction_max_candidates() -> usize {
    2
}

fn default_prediction_min_reading_chars() -> usize {
    2
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_variant: None,
            num_candidates: 5,
            n_threads: 0,
            n_gpu_layers: 0u32,
            main_gpu: 0,
            digit_width: DigitWidth::default(),
            alpha_width: AlphaWidth::default(),
            symbol_width: SymbolWidth::default(),
            digit_separator_auto: true,
            digit_candidates_order: default_digit_candidates_order(),
            live_conv_beam_size: 3,
            convert_beam_size: 30,
            confidence_margin: default_confidence_margin(),
            min_top_confidence: None,
            prediction_enabled: default_prediction_enabled(),
            prediction_max_candidates: default_prediction_max_candidates(),
            prediction_min_reading_chars: default_prediction_min_reading_chars(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PreeditState {
    pub hiragana: String,
    pub pending_romaji: String,
}

impl PreeditState {
    pub fn display(&self) -> String {
        format!("{}{}", self.hiragana, self.pending_romaji)
    }
    pub fn is_empty(&self) -> bool {
        self.hiragana.is_empty() && self.pending_romaji.is_empty()
    }
}

fn is_numeric_digit(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c)
}

fn numeric_separator_after_digit(prev: Option<char>, c: char) -> Option<char> {
    if !prev.is_some_and(is_numeric_digit) {
        return None;
    }
    match c {
        ',' | '、' => Some(','),
        '.' | '。' => Some('.'),
        _ => None,
    }
}

fn is_alpha_char(c: char) -> bool {
    c.is_ascii_alphabetic() || ('Ａ'..='Ｚ').contains(&c) || ('ａ'..='ｚ').contains(&c)
}

fn is_symbol_char(c: char) -> bool {
    let n = c as u32;
    // ASCII printable 記号（英数字除く）
    if (0x21..=0x7E).contains(&n) && !c.is_ascii_alphanumeric() {
        return true;
    }
    // 全角記号 (U+FF01..=U+FF5E)、ただし全角英数字を除く
    if (0xFF01..=0xFF5E).contains(&n)
        && !('０'..='９').contains(&c)
        && !('Ａ'..='Ｚ').contains(&c)
        && !('ａ'..='ｚ').contains(&c)
    {
        return true;
    }
    false
}

/// `,` / `.` / `、` / `。` を、直前文字の種類と幅設定に応じて
/// Western 句読点（全角 ， ． or 半角 , .）として返す。
/// 直前が英字でも記号でもなければ `None`（変換せず trie に委ねる）。
fn alpha_symbol_separator_auto(
    prev: Option<char>,
    c: char,
    alpha_width: AlphaWidth,
    symbol_width: SymbolWidth,
) -> Option<char> {
    let prev = prev?;
    let fullwidth = if is_alpha_char(prev) {
        matches!(alpha_width, AlphaWidth::Fullwidth)
    } else if is_symbol_char(prev) {
        matches!(symbol_width, SymbolWidth::Fullwidth)
    } else {
        return None;
    };
    match (c, fullwidth) {
        (',' | '、', true) => Some('，'), // U+FF0C 全角コンマ
        (',' | '、', false) => Some(','),
        ('.' | '。', true) => Some('．'), // U+FF0E 全角ピリオド
        ('.' | '。', false) => Some('.'),
        _ => None,
    }
}

pub struct RakunEngine {
    romaji: RomajiConverter,
    kanji: Option<KanaKanjiConverter>,
    config: EngineConfig,
    hiragana_buf: String,
    pending_romaji_buf: String,
    /// ローマ字入力ログ。`RomajiConverter::Converted` 単位で1エントリとして積む。
    /// 末尾エントリは pending_romaji_buf に対応する未確定分（確定時に上書き）。
    /// F9/F10 でかな→ローマ字復元に使用する。
    romaji_input_log: Vec<String>,
    committed: String,
    dict_store: Option<DictStore>,
}

impl RakunEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            romaji: RomajiConverter::new(),
            kanji: None,
            config,
            hiragana_buf: String::new(),
            pending_romaji_buf: String::new(),
            romaji_input_log: Vec::new(),
            committed: String::new(),
            dict_store: None,
        }
    }

    pub fn init_kanji(&mut self) -> Result<(), EngineError> {
        let converter = Self::build_converter(&self.config)?;
        self.kanji = Some(converter);
        Ok(())
    }

    pub fn build_converter(config: &EngineConfig) -> Result<KanaKanjiConverter, EngineError> {
        let variant_id = config
            .model_variant
            .clone()
            .unwrap_or_else(|| registry().default_model.clone());
        info!(
            "engine::init: loading model={} gpu_layers={} main_gpu={}",
            variant_id, config.n_gpu_layers, config.main_gpu
        );
        let backend = KarukanBackend::from_variant_id(&variant_id)
            .map_err(|e| EngineError::InitFailed(e.to_string()))?
            .with_n_gpu_layers(config.n_gpu_layers)
            .with_main_gpu(config.main_gpu);
        let conv_cfg = kanji::ConversionConfig {
            beam_size: config.convert_beam_size,
            confidence_margin: config.confidence_margin,
            min_top_confidence: config.min_top_confidence,
            ..Default::default()
        };
        let mut converter = KanaKanjiConverter::with_config(backend, conv_cfg)
            .map_err(|e| EngineError::InitFailed(e.to_string()))?;
        if config.n_threads > 0 {
            converter.set_n_threads(config.n_threads);
        }
        info!(
            "engine::init: model ready name={}",
            converter.model_display_name()
        );
        Ok(converter)
    }

    pub fn set_kanji_converter(&mut self, converter: KanaKanjiConverter) {
        self.kanji = Some(converter);
    }

    pub fn take_kanji_converter(&mut self) -> Option<KanaKanjiConverter> {
        self.kanji.take()
    }

    pub fn hiragana_text(&self) -> &str {
        &self.hiragana_buf
    }

    pub fn push_char(&mut self, c: char) -> PreeditState {
        if self.config.digit_separator_auto && self.pending_romaji_buf.is_empty() {
            if let Some(separator) =
                numeric_separator_after_digit(self.hiragana_buf.chars().last(), c)
            {
                self.hiragana_buf.push(separator);
                self.romaji_input_log.push(c.to_string());
                debug!("engine::push: numeric separator {:?} → {:?}", c, separator);
                return self.current_preedit();
            }
        }

        // 英字・記号後の `,` / `.` を Western 句読点 (， / ． or , / .) へ自動置換
        // 幅設定 (alpha_width / symbol_width) に追従する。
        if self.pending_romaji_buf.is_empty() {
            if let Some(separator) = alpha_symbol_separator_auto(
                self.hiragana_buf.chars().last(),
                c,
                self.config.alpha_width,
                self.config.symbol_width,
            ) {
                self.hiragana_buf.push(separator);
                self.romaji_input_log.push(c.to_string());
                debug!(
                    "engine::push: alpha/symbol separator {:?} → {:?}",
                    c, separator
                );
                return self.current_preedit();
            }
        }

        // 数字 0–9（pending_romaji がない場合のみ）
        if self.pending_romaji_buf.is_empty() && c.is_ascii_digit() {
            let out = match self.config.digit_width {
                DigitWidth::Fullwidth => char::from_u32(c as u32 - 0x30 + 0xFF10).unwrap_or(c),
                DigitWidth::Halfwidth => c,
            };
            self.hiragana_buf.push(out);
            self.romaji_input_log.push(c.to_string());
            debug!("engine::push: digit {:?} → {:?}", c, out);
            return self.current_preedit();
        }

        // ASCII 記号の処理（pending_romaji がない場合のみ）
        // ,./[]\- はトライのルール（、。・「」￥ー等）に委ねる。
        // それ以外の印字可能 ASCII 記号（@#$%^&*()+=_:"~!? 等）は
        // symbol_width に従って全角 or 半角で即確定する。
        if self.pending_romaji_buf.is_empty() {
            let n = c as u32;
            let is_ascii_printable = (0x21..=0x7E).contains(&n);
            let is_trie_symbol = matches!(c, ',' | '.' | '/' | '[' | ']' | '\\' | '-');
            if is_ascii_printable && !is_trie_symbol && !c.is_ascii_alphanumeric() {
                let out = match self.config.symbol_width {
                    SymbolWidth::Fullwidth => char::from_u32(n - 0x21 + 0xFF01).unwrap_or(c),
                    SymbolWidth::Halfwidth => c,
                };
                self.hiragana_buf.push(out);
                self.romaji_input_log.push(c.to_string());
                debug!("engine::push: symbol {:?} → {:?}", c, out);
                return self.current_preedit();
            }
        }

        // ,./[]\- および英字 → ローマ字ルール（trie）に委ねる
        // pending_romaji_buf と romaji.buffer は常に同じ状態を保つ。
        // ConversionEvent variant ではなく romaji.output / romaji.buffer の差分から
        // 「確定したひらがな」と「未確定として残っているローマ字」を判定する。
        // （PassThrough の連鎖で複数文字が確定するケースを正しく扱うため）
        self.pending_romaji_buf.push(c);
        let prev_output_len = self.romaji.output().len();
        let _ = self.romaji.push(c);

        let added = self.romaji.output()[prev_output_len..].to_string();
        let new_buffer_len = self.romaji.buffer().len();
        debug_assert!(new_buffer_len <= self.pending_romaji_buf.len());
        let consumed_len = self.pending_romaji_buf.len() - new_buffer_len;
        if consumed_len > 0 {
            let entry: String = self.pending_romaji_buf.drain(..consumed_len).collect();
            self.hiragana_buf.push_str(&added);
            debug!("engine::push: romaji {:?} → {:?}", entry, added);
            self.romaji_input_log.push(entry);
        }
        self.current_preedit()
    }

    /// 末尾の未確定 "n" を「ん」として確定する（Convert / CommitRaw 前に呼ぶ）
    pub fn flush_pending_n(&mut self) -> bool {
        if self.pending_romaji_buf == "n" {
            self.hiragana_buf.push('ん');
            let entry = std::mem::take(&mut self.pending_romaji_buf);
            self.romaji_input_log.push(entry);
            self.romaji = RomajiConverter::new();
            true
        } else {
            false
        }
    }

    /// プリエディット文字列を強制置換する（F6〜F10 の文字種変換用）
    /// romaji_input_log は保持する（F9/F10 サイクル中に再度ローマ字に戻せるよう）
    pub fn force_preedit(&mut self, text: String) {
        self.hiragana_buf = text;
        self.pending_romaji_buf.clear();
        self.romaji = RomajiConverter::new();
    }

    /// ローマ字変換を経由せず hiragana_buf に直接1文字追加する。
    /// テンキー記号など、かなルールに登録されている文字をそのまま入力する場合に使用する。
    pub fn push_raw(&mut self, c: char) {
        self.hiragana_buf.push(c);
        self.romaji_input_log.push(c.to_string());
    }

    /// Shift+アルファベット用: alpha_width 設定に従って全角 or 半角の大文字を hiragana_buf に追加。
    /// `romaji_input_log` には ASCII 大文字を記録する。
    ///
    /// F9/F10 のサイクル変換は romaji_input_log の ASCII 文字を元に動作するため、
    /// log には元の ASCII 文字（'A'–'Z'）を保持する必要がある。
    /// `c` には ASCII 大文字（'A'–'Z'）を渡すこと。
    pub fn push_fullwidth_alpha(&mut self, c: char) {
        debug_assert!(c.is_ascii_uppercase());
        let out = match self.config.alpha_width {
            AlphaWidth::Fullwidth => char::from_u32(c as u32 - 0x41 + 0xFF21).unwrap_or(c),
            AlphaWidth::Halfwidth => c,
        };
        self.hiragana_buf.push(out);
        self.romaji_input_log.push(c.to_string());
    }

    pub fn backspace(&mut self) -> bool {
        use romaji::BackspaceResult;
        match self.romaji.backspace() {
            BackspaceResult::RemovedBuffer(_) => {
                self.pending_romaji_buf.pop();
                // pending_romaji_buf はまだ未確定 → romaji_input_log には記録されていない
                // log 操作は不要
                true
            }
            BackspaceResult::RemovedOutput(_) => {
                self.hiragana_buf.pop();
                // 確定済みのひらがな1文字分 → log エントリを1つ pop
                self.romaji_input_log.pop();
                true
            }
            BackspaceResult::Empty => {
                if self.hiragana_buf.is_empty() {
                    false
                } else {
                    self.hiragana_buf.pop();
                    self.romaji_input_log.pop();
                    true
                }
            }
        }
    }

    pub fn convert(&self, num_candidates: usize) -> Result<Vec<String>, EngineError> {
        if self.hiragana_buf.is_empty() {
            return Ok(vec![]);
        }
        let kanji = self
            .kanji
            .as_ref()
            .ok_or(EngineError::ModelNotInitialized)?;
        digits::convert_with_digit_protection(
            kanji,
            &self.hiragana_buf,
            &self.committed,
            num_candidates,
            &self.config.digit_candidates_order,
            matches!(self.config.alpha_width, AlphaWidth::Fullwidth),
            matches!(self.config.symbol_width, SymbolWidth::Fullwidth),
        )
        .map_err(|e| EngineError::ConversionFailed(e.to_string()))
    }

    pub fn convert_default(&self) -> Result<Vec<String>, EngineError> {
        self.convert(self.config.num_candidates)
    }

    pub fn commit(&mut self, text: &str) {
        info!("engine::commit: {:?}", text);
        if is_context_echo_risk(text) {
            // 未変換のまま確定されたひらがな文を context に入れると、同じ読みの
            // 変換で LLM がコピー（エコー）に収束する（v0.9.15 のエコーアトラクタ）。
            // 確定自体は成立させ、context にだけ入れない。
            info!("engine::commit: hiragana-only text excluded from context");
            self.hiragana_buf.clear();
            self.romaji_input_log.clear();
            self.romaji = RomajiConverter::new();
            return;
        }
        self.committed.push_str(text);
        if self.committed.chars().count() > 200 {
            // 文境界でトリミング: 直近 2 文を残す。
            // 200 文字単純切りより自然な文脈を LLM に渡せる。
            let start = last_n_sentences_start(&self.committed, 2);
            if start > 0 {
                self.committed = self.committed[start..].to_string();
            } else {
                // 文境界が見つからない場合は従来通り直近 200 文字
                let fallback = self
                    .committed
                    .char_indices()
                    .rev()
                    .nth(199)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.committed = self.committed[fallback..].to_string();
            }
        }
        self.hiragana_buf.clear();
        self.romaji_input_log.clear();
        self.romaji = RomajiConverter::new();
    }

    pub fn commit_as_hiragana(&mut self) {
        let text = self.hiragana_buf.clone();
        if !text.is_empty() {
            self.commit(&text);
        }
    }

    pub fn current_preedit(&self) -> PreeditState {
        PreeditState {
            hiragana: self.hiragana_buf.clone(),
            pending_romaji: self.pending_romaji_buf.clone(),
        }
    }

    pub fn preedit_is_empty(&self) -> bool {
        self.hiragana_buf.is_empty() && self.pending_romaji_buf.is_empty()
    }

    /// ローマ字入力ログを結合した文字列を返す（F9/F10 のローマ字復元用）
    pub fn romaji_log_str(&self) -> String {
        self.romaji_input_log.concat()
    }

    /// romaji_input_log からひらがなを復元する（F6/F7/F8 でかなに戻す用）
    /// F9/F10 で force_preedit した後でも log は保持されているため復元可能。
    pub fn hiragana_from_romaji_log(&self) -> String {
        let romaji = self.romaji_input_log.concat();
        if romaji.is_empty() {
            return String::new();
        }
        let mut conv = RomajiConverter::new();
        let mut result = String::new();
        for c in romaji.chars() {
            match conv.push(c) {
                crate::romaji::ConversionEvent::Converted(h) => result.push_str(&h),
                crate::romaji::ConversionEvent::PassThrough(ch) => result.push(ch),
                crate::romaji::ConversionEvent::Buffered => {}
            }
        }
        // pending を flush
        result.push_str(&conv.flush());
        result
    }
    pub fn get_config(&self) -> &EngineConfig {
        &self.config
    }
    pub fn committed_text(&self) -> &str {
        &self.committed
    }
    pub fn is_kanji_ready(&self) -> bool {
        self.kanji.is_some()
    }

    pub fn set_dict_store(&mut self, store: DictStore) {
        info!(
            "engine::dict: store set user_entries={}",
            store.user_entry_count()
        );
        self.dict_store = Some(store);
    }

    /// 確定した候補をユーザー辞書に学習して保存する
    /// 学習語を DictStore に即時反映してファイルにも保存する。
    pub fn learn(&mut self, reading: &str, surface: &str) {
        if let Some(store) = &self.dict_store {
            store.learn(reading, surface);
        } else {
            tracing::warn!("learn: dict_store not initialized");
        }
    }

    pub fn learn_force(&mut self, reading: &str, surface: &str) {
        if let Some(store) = &self.dict_store {
            store.learn_force(reading, surface);
        } else {
            tracing::warn!("learn_force: dict_store not initialized");
        }
    }

    /// 入力中の予測候補（Google 日本語入力の予測ウィンドウ相当）を返す。
    ///
    /// 学習履歴のみを引く（LLM も MOZC 辞書も引かない）ので、打鍵ごとに呼んでも
    /// HashMap の前方一致走査だけで済む。
    pub fn predict(&self, reading: &str, limit: usize) -> Vec<String> {
        if !self.config.prediction_enabled {
            return vec![];
        }
        if reading.chars().count() < self.config.prediction_min_reading_chars {
            return vec![];
        }
        self.dict_store
            .as_ref()
            .map(|d| d.lookup_learn_suggest(reading, limit))
            .unwrap_or_default()
    }

    /// 学習履歴から候補を削除する（候補ウィンドウでの明示削除）。
    ///
    /// `reading` に前方一致するキーもまとめて対象にするため、短文予測で出てきた
    /// 候補（現在の読みより長いキーで登録されている）もその場で消せる。
    pub fn forget(&mut self, reading: &str, surface: &str) -> bool {
        if let Some(store) = &self.dict_store {
            store.forget_matching(reading, surface) > 0
        } else {
            tracing::warn!("forget: dict_store not initialized");
            false
        }
    }

    pub fn is_dict_ready(&self) -> bool {
        self.dict_store.is_some()
    }

    pub fn dict_store_ref(&self) -> Option<&DictStore> {
        self.dict_store.as_ref()
    }

    /// 入力したローマ字をそのまま出す「英数候補」。戻り値は (半角, 全角)。
    ///
    /// `claude` のように日本語のローマ字綴りとして成立しない語は、かな変換を
    /// 通すと `cぁうで` のような無意味な読みになり、どの候補も当たらない。
    /// F9/F10 を押せば `romaji_input_log` から復元できるが、それは「変換候補に
    /// 出ていないだけで、打った文字列はエンジンが保持している」という状態なので、
    /// 候補として提示する。
    ///
    /// `hiragana` がプリエディット全体と一致する時だけ返す。文節分割された
    /// 部分読みに対してプリエディット全体のローマ字を出さないためのガード。
    /// 打鍵したローマ字（半角英数）と、その全角版を返す。
    ///
    /// 読みが打鍵ログから復元できる場合だけ返す。F9/F10 で `force_preedit`
    /// された後や、記号・空白が混ざった入力では `None`（記号混じりは
    /// digits.rs のリテラル保護レイヤーの担当）。
    fn romaji_alnum_candidates(&self, hiragana: &str) -> Option<(String, String)> {
        let romaji = self.romaji_log_str();
        if romaji.is_empty() {
            return None;
        }
        // 英字を含む純粋な英数字列のみ。記号・空白が混ざるものは
        // digits.rs のリテラル保護レイヤーの担当。
        if !romaji.chars().all(|c| c.is_ascii_alphanumeric())
            || !romaji.chars().any(|c| c.is_ascii_alphabetic())
        {
            return None;
        }
        if self.hiragana_from_romaji_log() != hiragana {
            return None;
        }
        let full: String = romaji.chars().map(ascii_to_fullwidth).collect();
        Some((romaji, full))
    }

    /// 候補の**先頭**に置いてよい英数候補。
    ///
    /// 読みに ASCII 英字が残っている = ローマ字がかなに変換しきれていない、
    /// という場合だけ返す。"つづけて" のような普通の読みで先頭を "tudukete" に
    /// 奪われると、候補リストが英数字で埋まるうえ、変換途中で候補が 0 件の
    /// 瞬間にライブ変換の preview まで英字になってしまう。
    /// 普通の読みの英数候補は末尾の文字種候補（`merge_candidates_for_reading`
    /// の 8. ブロック）が拾う。
    fn romaji_literal_candidates(&self, hiragana: &str) -> Option<(String, String)> {
        if !hiragana.chars().any(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        self.romaji_alnum_candidates(hiragana)
    }

    pub fn merge_candidates_for_reading(
        &self,
        hiragana: &str,
        llm_candidates: Vec<String>,
        limit: usize,
    ) -> Vec<String> {
        // 優先順位: ユーザー辞書(normal) → 学習済み辞書候補（スコア順）
        //           → ユーザー辞書(low) → 残り辞書候補 → LLM
        // 学習スコアで上位に来た辞書候補を先に表示し、LLM は空きスロットを埋める。
        //
        // `priority = "low"` のユーザー辞書エントリを学習履歴より後ろに置くことで、
        // 一般語と読みが衝突する固有名詞を大量登録しても通常変換を壊さない。
        // 一度選べば学習履歴に載って前に出て、使わなくなれば学習スコアの減衰で戻る。
        let user_cands: Vec<String> = self
            .dict_store
            .as_ref()
            .map(|d| d.lookup_user(hiragana))
            .unwrap_or_default();

        let user_low_cands: Vec<String> = self
            .dict_store
            .as_ref()
            .map(|d| d.lookup_user_low(hiragana))
            .unwrap_or_default();

        let learn_cands: Vec<String> = self
            .dict_store
            .as_ref()
            .map(|d| d.lookup_learn(hiragana))
            .unwrap_or_default();

        // MOZC 辞書候補は「語」と「記号だけ」に分ける。記号だけのものは LLM の
        // 後ろへ回す（`is_symbol_only_candidate` のコメント参照）。
        let (dict_cands, dict_symbol_cands): (Vec<String>, Vec<String>) = self
            .dict_store
            .as_ref()
            .map(|d| d.lookup_dict(hiragana, limit))
            .unwrap_or_default()
            .into_iter()
            .partition(|c| !is_symbol_only_candidate(c));

        // 「ん」を 1 打鍵で済ませたときの取りこぼしを辞書だけで補う。
        // LLM は呼ばないので変換の待ち時間は増えない。
        let n_fix_cands: Vec<String> = self
            .dict_store
            .as_ref()
            .map(|d| {
                let mut out: Vec<String> = Vec::new();
                for alt in n_insertion_readings(hiragana) {
                    let alt_cands = d
                        .lookup_user(&alt)
                        .into_iter()
                        .chain(d.lookup_learn(&alt))
                        .chain(d.lookup_dict(&alt, N_INSERTION_MAX_CANDIDATES * 2));
                    for c in alt_cands {
                        if is_symbol_only_candidate(&c) || out.contains(&c) {
                            continue;
                        }
                        out.push(c);
                        if out.len() >= N_INSERTION_MAX_CANDIDATES {
                            return out;
                        }
                    }
                }
                out
            })
            .unwrap_or_default();

        // 短文予測（Google 日本語入力の「予測候補」相当）。
        // 読みが前方一致する学習済みフレーズを引く。読みが短いうちは候補が
        // 発散するので `prediction_min_reading_chars` 未満では引かない。
        let prediction_cands: Vec<String> = if self.config.prediction_enabled
            && hiragana.chars().count() >= self.config.prediction_min_reading_chars
        {
            self.dict_store
                .as_ref()
                .map(|d| d.lookup_learn_prefix(hiragana, self.config.prediction_max_candidates))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        debug!(
            "engine::merge: reading={:?} dict_store={} user_cands={:?} user_low_cands={:?} learn_cands={:?} prediction_cands={:?} dict_cands={:?} llm_cands={:?}",
            hiragana,
            if self.dict_store.is_some() {
                "Some"
            } else {
                "None"
            },
            user_cands,
            user_low_cands,
            learn_cands,
            prediction_cands,
            dict_cands,
            llm_candidates
        );
        debug!(
            "engine::merge: dict_symbol_cands={:?} n_fix_cands={:?}",
            dict_symbol_cands, n_fix_cands
        );

        let mut merged: Vec<String> = Vec::new();

        // 1. ユーザー辞書候補（最優先）
        for c in &user_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 2. 学習履歴: スコア順（最近・頻繁に選んだもの優先）で前に出す。
        //    DictStore::learn 側で「ひらがな・CJK漢字を含む surface は辞書ガード必須」と
        //    制御しているため、ここでの二重チェックは不要。辞書外の surface（記号・カタカナ等）
        //    も学習対象になったので、dict_cands チェックは外す。
        for c in &learn_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 3. 低優先ユーザー辞書（学習履歴の後ろ・システム辞書の前）
        for c in &user_low_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 4. 残りの辞書候補（学習で上昇済みのものは既に merged に含まれる）
        for c in &dict_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 4.5 「ん」補完（辞書引きのみ）。げにん → げんいん → 原因。
        for c in &n_fix_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 5. LLM候補（残りスロット、文脈考慮）
        for c in llm_candidates {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(&c) {
                merged.push(c);
            }
        }

        // 5.5 記号だけの辞書候補。表示スロットを奪わないよう LLM の後ろに置く。
        for c in &dict_symbol_cands {
            if merged.len() >= limit {
                break;
            }
            if !merged.contains(c) {
                merged.push(c.clone());
            }
        }

        // 6. 短文予測（Google 日本語入力相当）: 読みが前方一致する学習済みの長い
        //    フレーズを、先頭候補の直後に差し込む。先頭を奪わないのは、ライブ変換の
        //    preview が候補 0 番を採用するため（打鍵途中に長文が出続けるのを避ける）。
        if !prediction_cands.is_empty() {
            // 重複した予測は挿入しないので、挿入位置は「実際に入れた数」で進める
            // （enumerate の添字で進めると len を超えて insert が panic する）。
            let mut at = 1.min(merged.len());
            for c in &prediction_cands {
                if merged.contains(c) {
                    continue;
                }
                merged.insert(at, c.clone());
                at += 1;
            }
            merged.truncate(limit.max(1));
        }

        // 7. 英数候補（先頭）: 入力したローマ字をそのまま出す。
        //
        //    ここに来るのは読みに ASCII 英字が残っている場合だけなので、先頭に置く。
        //    ローマ字がかなに変換しきれていない = 日本語の語として読む余地が無い、
        //    という判定であり、`つづけて` のような普通の読みはそもそも
        //    `romaji_literal_candidates` が None を返して届かない。
        if let Some((half, full)) = self.romaji_literal_candidates(hiragana) {
            let ordered = match self.config.alpha_width {
                AlphaWidth::Halfwidth => [half, full],
                AlphaWidth::Fullwidth => [full, half],
            };
            let mut at = 0usize;
            for c in ordered {
                // 既にリストにある場合は取り除いてから入れ直す。読みそのものが
                // 半角英数（"xy"）だと後段の文字種候補と重複するので、
                // skip すると全角だけが前に出て alpha_width の指定と逆になる。
                merged.retain(|existing| existing != &c);
                at = at.min(merged.len());
                merged.insert(at, c);
                at += 1;
            }
        }
        merged.truncate(limit.max(1));

        // 8. 文字種候補（末尾）: ひらがな → カタカナ → 半角英数 → 全角英数。
        //
        //    Google 日本語入力と同じく、通常候補を出し切った後ろに常に添える。
        //    F6〜F10 を押さなくても候補リストから選べるようにするのが目的で、
        //    「候補が足りないときに読みを末尾へ足す」退避路もここへ統合した。
        //
        //    🔴 必ず他のすべての候補を積んだ *後* に push すること。Space 直後の
        //    同期パスは LLM 未完了で merged が 0 件になりうるので、insert で前へ
        //    差し込むとライブ変換の preview と composition が化ける（候補 0 番が
        //    プリエディットに採用されるため）。
        //
        //    truncate も済ませた後に足すので、返る件数は limit + 4 まで伸びうる。
        //    候補ウィンドウはページャを持っており、表示ページ数（num_candidates）
        //    とは無関係なので問題ない。
        //
        //    🔴 実候補が 1 件も無いときは読みだけを足して打ち切る。TSF のライブ変換は
        //    `merge_candidates_for_reading(reading, vec![], 40)` に「読み以外の候補が
        //    あるか」を尋ねて preview を出すか決めており（`start_live_bg_if_ready` /
        //    `has_immediate_live_preview_candidate` / `on_live_timer`）、候補 0 件の
        //    状態でカタカナを足すと打鍵のたびに preview がカタカナに化ける。
        if merged.is_empty() {
            merged.push(hiragana.to_string());
        } else {
            let mut char_type_cands: Vec<String> =
                vec![hiragana.to_string(), hiragana_to_katakana(hiragana)];
            if let Some((half, full)) = self.romaji_alnum_candidates(hiragana) {
                match self.config.alpha_width {
                    AlphaWidth::Halfwidth => char_type_cands.extend([half, full]),
                    AlphaWidth::Fullwidth => char_type_cands.extend([full, half]),
                }
            }
            for c in char_type_cands {
                if c.is_empty() || merged.contains(&c) {
                    continue;
                }
                merged.push(c);
            }
        }

        if merged.is_empty() {
            vec![hiragana.to_string()]
        } else {
            merged
        }
    }

    pub fn merge_candidates(&self, llm_candidates: Vec<String>, limit: usize) -> Vec<String> {
        self.merge_candidates_for_reading(&self.hiragana_buf, llm_candidates, limit)
    }

    pub fn backend_label(&self) -> String {
        compiled_backend_label().to_string()
    }

    // ─── Background 変換 API ──────────────────────────────────────────────────
    // conv_cache が engine 内部に移動したことで、TSF 側は converter を直接触らない。

    /// バックグラウンド変換を起動する。
    /// is_kanji_ready() == true の場合にのみ converter をキャッシュに渡す。
    /// False: kanji 未準備 or ひらがなが空。
    pub fn bg_start(&mut self, n_cands: usize) -> bool {
        // is_kanji_ready() チェックの前に Done 状態の converter を回収する。
        // キー不一致で take_ready が None を返した場合、converter は Done に戻るが
        // engine.kanji=None のまま → is_kanji_ready()=false → bg_start が永遠にスキップ
        // されてしまう。回収を先に行うことでこの問題を解消する。
        if let Some(old) = conv_cache::try_reclaim_done() {
            tracing::trace!("bg_start: reclaimed converter from Done state");
            self.kanji = Some(old);
        }

        let hiragana = self.hiragana_buf.clone();
        let committed = self.committed.clone();
        if hiragana.is_empty() {
            return false;
        }
        if !self.is_kanji_ready() {
            return false;
        }

        if let Some(conv) = self.kanji.take() {
            match conv_cache::start(
                hiragana,
                committed,
                conv,
                n_cands,
                self.config.digit_candidates_order.clone(),
                matches!(self.config.alpha_width, AlphaWidth::Fullwidth),
                matches!(self.config.symbol_width, SymbolWidth::Fullwidth),
            ) {
                Some(returned) => {
                    self.kanji = Some(returned);
                    false
                }
                None => true,
            }
        } else {
            false
        }
    }

    /// BG 変換の状態文字列（診断用）
    pub fn bg_status(&self) -> &'static str {
        conv_cache::status()
    }

    /// ライブ変換 preview 用にトップ候補だけを覗き見する (M2 §5.2)。
    ///
    /// `bg_take_candidates` と異なり cache 状態を進めず、converter は cache に
    /// 残す。dict マージも行わないため、preview の純度が上がり commit 経路と
    /// 干渉しない。状態を進めない=複数回 peek しても結果は同じ。
    ///
    /// 次回 `bg_start` で別キーが来たときは、`bg_start` 内部で
    /// `conv_cache::reclaim_nonblocking()` が Done state から converter を
    /// 回収するため、converter を engine.kanji に戻す手間は不要。
    pub fn bg_peek_top_candidate(&self, key: &str) -> Option<String> {
        conv_cache::peek_top_candidate(key)
    }

    /// key が一致する BG 変換結果を取得し、converter を engine に戻す。
    /// None = まだ完了していない / キー不一致
    ///
    /// ユーザー辞書ヒットは LLM 結果より優先するため先頭にマージする。
    /// ライブ変換 preview (先頭候補表示) でユーザー辞書が勝つ必要があるため。
    pub fn bg_take_candidates(&mut self, key: &str) -> Option<Vec<String>> {
        let (conv, cands) = conv_cache::take_ready(key)?;
        self.kanji = Some(conv);
        let user_cands: Vec<String> = self
            .dict_store
            .as_ref()
            .map(|d| d.lookup_user(key))
            .unwrap_or_default();
        if user_cands.is_empty() {
            return Some(cands);
        }
        let mut merged = user_cands;
        for c in cands {
            if !merged.contains(&c) {
                merged.push(c);
            }
        }
        Some(merged)
    }

    /// Done 状態の converter を engine に戻す（commit/cancel 時に呼ぶ）
    pub fn bg_reclaim(&mut self) {
        if let Some(conv) = conv_cache::reclaim_nonblocking() {
            self.kanji = Some(conv);
        }
    }

    pub fn reset_preedit(&mut self) {
        self.hiragana_buf.clear();
        self.romaji = RomajiConverter::new();
        self.pending_romaji_buf.clear();
        self.romaji_input_log.clear();
    }

    pub fn reset_all(&mut self) {
        self.hiragana_buf.clear();
        self.committed.clear();
        self.romaji = RomajiConverter::new();
        self.pending_romaji_buf.clear();
        self.romaji_input_log.clear();
    }

    pub fn available_models() -> Vec<ModelInfo> {
        let reg = registry();
        let mut models: Vec<ModelInfo> = reg
            .models
            .values()
            .flat_map(|family| {
                family.variants.values().map(|v| ModelInfo {
                    id: v.id.clone(),
                    display_name: v.display_name.clone(),
                    is_default: v.id == reg.default_model,
                })
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }
}

fn compiled_backend_label() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA"
    }
    #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
    {
        "Vulkan"
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "vulkan")))]
    {
        "CPU"
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
    pub is_default: bool,
}

#[cfg(test)]
mod context_trim_tests {
    use super::last_n_sentences_start;

    #[test]
    fn empty_text() {
        assert_eq!(last_n_sentences_start("", 2), 0);
    }

    #[test]
    fn no_boundary() {
        let text =
            "\u{6587}\u{5883}\u{754C}\u{306E}\u{306A}\u{3044}\u{30C6}\u{30AD}\u{30B9}\u{30C8}";
        assert_eq!(last_n_sentences_start(text, 2), 0);
    }

    #[test]
    fn single_boundary_want_two() {
        let text =
            "\u{6700}\u{521D}\u{306E}\u{6587}\u{3002}\u{4E8C}\u{756A}\u{76EE}\u{306E}\u{6587}";
        // \u{5883}\u{754C}\u{304C}1\u{500B}\u{3057}\u{304B}\u{306A}\u{3044} \u{2192} \u{5148}\u{982D}\u{3092}\u{8FD4}\u{3059}
        assert_eq!(last_n_sentences_start(text, 2), 0);
    }

    #[test]
    fn two_boundaries_want_two() {
        let text = "\u{6700}\u{521D}\u{306E}\u{6587}\u{3002}\u{4E8C}\u{756A}\u{76EE}\u{306E}\u{6587}\u{3002}\u{4E09}\u{756A}\u{76EE}\u{306E}\u{6587}";
        // \u{5883}\u{754C}\u{304C}2\u{500B} [\u{300C}\u{4E8C}\u{756A}\u{76EE}\u{300D}\u{5148}\u{982D}, \u{300C}\u{4E09}\u{756A}\u{76EE}\u{300D}\u{5148}\u{982D}]\u{3001}n=2 \u{2192} \u{5148}\u{982D}\u{304B}\u{3089}2\u{500B}\u{76EE}\u{306E}\u{5883}\u{754C} = \u{300C}\u{4E8C}\u{756A}\u{76EE}\u{300D}\u{5148}\u{982D}
        let start = last_n_sentences_start(text, 2);
        assert_eq!(
            &text[start..],
            "\u{4E8C}\u{756A}\u{76EE}\u{306E}\u{6587}\u{3002}\u{4E09}\u{756A}\u{76EE}\u{306E}\u{6587}"
        );
    }

    #[test]
    fn multiple_punctuation() {
        let text = "A\u{FF01}\u{FF1F}B\u{3002}C";
        // \u{5883}\u{754C}2\u{500B} [\u{300C}B\u{300D}\u{5148}\u{982D}, \u{300C}C\u{300D}\u{5148}\u{982D}]\u{3001}n=2 \u{2192} \u{300C}B\u{300D}\u{5148}\u{982D}
        let start = last_n_sentences_start(text, 2);
        assert_eq!(&text[start..], "B\u{3002}C");
    }

    #[test]
    fn linebreak_as_boundary() {
        let text = "\u{4E00}\u{884C}\u{76EE}\n\u{4E8C}\u{884C}\u{76EE}\n\u{4E09}\u{884C}\u{76EE}";
        // \u{5883}\u{754C}2\u{500B} [\u{300C}\u{4E8C}\u{884C}\u{76EE}\u{300D}\u{5148}\u{982D}, \u{300C}\u{4E09}\u{884C}\u{76EE}\u{300D}\u{5148}\u{982D}]\u{3001}n=2 \u{2192} \u{300C}\u{4E8C}\u{884C}\u{76EE}\u{300D}\u{5148}\u{982D}
        let start = last_n_sentences_start(text, 2);
        assert_eq!(
            &text[start..],
            "\u{4E8C}\u{884C}\u{76EE}\n\u{4E09}\u{884C}\u{76EE}"
        );
    }

    #[test]
    fn want_one_sentence() {
        let text = "\u{6587}A\u{3002}\u{6587}B\u{3002}\u{6587}C";
        // n=1 \u{2192} \u{6700}\u{5F8C}\u{306E}\u{5883}\u{754C} = \u{300C}\u{6587}C\u{300D}\u{5148}\u{982D}
        let start = last_n_sentences_start(text, 1);
        assert_eq!(&text[start..], "\u{6587}C");
    }
}

#[cfg(test)]
mod context_echo_risk_tests {
    use super::{RakunEngine, is_context_echo_risk};

    #[test]
    fn hiragana_only_text_is_echo_risk() {
        // 実機事例と同型: 未変換のまま確定されたひらがな文
        assert!(is_context_echo_risk("きだじゅんいちろうしは、"));
        // 短いひらがな確定（4 文字）も対象
        assert!(is_context_echo_risk("きもちは、"));
        // 句読点・括弧・空白は無視して判定
        assert!(is_context_echo_risk("「よろしくおねがいします。」"));
    }

    #[test]
    fn converted_or_short_text_is_not_echo_risk() {
        // 漢字を含む = 変換済み
        assert!(!is_context_echo_risk("木田純一郎氏は、"));
        // カタカナはエコーしても正しい出力になるため対象外
        assert!(!is_context_echo_risk("コーヒー"));
        // 英数字を含む
        assert!(!is_context_echo_risk("2024ねん"));
        // ひらがな 4 文字未満
        assert!(!is_context_echo_risk("です。"));
        assert!(!is_context_echo_risk(""));
    }

    #[test]
    fn commit_excludes_hiragana_only_text_from_context() {
        let mut e = RakunEngine::new(crate::EngineConfig::default());
        e.commit("今日は晴れ。");
        e.commit("きだじゅんいちろうしは、");
        // ひらがなのみの確定は context（committed）に入らない
        assert_eq!(e.committed_text(), "今日は晴れ。");
        e.commit("紀田順一郎氏は、");
        assert_eq!(e.committed_text(), "今日は晴れ。紀田順一郎氏は、");
    }
}

#[cfg(test)]
mod symbol_input_tests {
    use super::RakunEngine;

    fn push(buf_init: &str, c: char) -> String {
        let mut e = RakunEngine::new(crate::EngineConfig::default());
        // hiragana_buf に初期値をセット
        e.force_preedit(buf_init.to_string());
        e.push_char(c);
        e.hiragana_text().to_string()
    }

    #[test]
    fn comma_to_kuten() {
        assert!(push("", ',').ends_with('、'));
        assert!(push("あ", ',').ends_with('、'));
    }

    #[test]
    fn comma_after_digit_stays_numeric_separator() {
        assert_eq!(push("2", ','), "2,");
        assert_eq!(push("２", '、'), "２,");
    }

    #[test]
    fn period_to_maru() {
        assert!(push("", '.').ends_with('。'));
    }

    #[test]
    fn period_after_digit_stays_numeric_separator() {
        assert_eq!(push("2", '.'), "2.");
        assert_eq!(push("２", '。'), "２.");
    }

    #[test]
    fn digit_separator_auto_can_be_disabled() {
        let config = crate::EngineConfig {
            digit_separator_auto: false,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.force_preedit("2".to_string());
        e.push_char(',');
        assert_eq!(e.hiragana_text(), "2、");
    }

    #[test]
    fn slash_to_nakaten() {
        assert!(push("", '/').ends_with('・'));
    }

    #[test]
    fn bracket_open() {
        assert!(push("", '[').ends_with('「'));
    }

    #[test]
    fn bracket_close() {
        assert!(push("", ']').ends_with('」'));
    }

    #[test]
    fn backslash_to_yen() {
        assert!(push("", '\\').ends_with('￥'));
    }

    #[test]
    fn minus_always_choon() {
        // 文脈依存ロジック廃止 → 常に ー
        assert!(push("", '-').ends_with('ー'));
        assert!(push("あ", '-').ends_with('ー'));
        assert!(push("abc", '-').ends_with('ー'));
    }

    #[test]
    fn other_symbols_fullwidth() {
        assert!(push("", '=').ends_with('＝'));
        assert!(push("", '@').ends_with('＠'));
        assert!(push("", '(').ends_with('（'));
        assert!(push("", ')').ends_with('）'));
    }

    #[test]
    fn symbol_width_halfwidth_keeps_ascii() {
        let config = crate::EngineConfig {
            symbol_width: crate::SymbolWidth::Halfwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_char('@');
        assert_eq!(e.hiragana_text(), "@");
    }

    #[test]
    fn alpha_width_halfwidth_keeps_ascii() {
        let config = crate::EngineConfig {
            alpha_width: crate::AlphaWidth::Halfwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_fullwidth_alpha('U');
        e.push_fullwidth_alpha('S');
        e.push_fullwidth_alpha('B');
        assert_eq!(e.hiragana_text(), "USB");
    }

    #[test]
    fn alpha_width_fullwidth_converts() {
        let config = crate::EngineConfig {
            alpha_width: crate::AlphaWidth::Fullwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_fullwidth_alpha('U');
        e.push_fullwidth_alpha('S');
        e.push_fullwidth_alpha('B');
        assert_eq!(e.hiragana_text(), "ＵＳＢ");
    }

    #[test]
    fn comma_after_alpha_with_fullwidth_uses_zenkaku_comma() {
        let config = crate::EngineConfig {
            alpha_width: crate::AlphaWidth::Fullwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_fullwidth_alpha('A');
        e.push_char(',');
        assert_eq!(e.hiragana_text(), "Ａ，");
    }

    #[test]
    fn comma_after_alpha_with_halfwidth_uses_ascii_comma() {
        let config = crate::EngineConfig {
            alpha_width: crate::AlphaWidth::Halfwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_fullwidth_alpha('A');
        e.push_char(',');
        assert_eq!(e.hiragana_text(), "A,");
    }

    #[test]
    fn period_after_symbol_with_fullwidth_uses_zenkaku_period() {
        let config = crate::EngineConfig {
            symbol_width: crate::SymbolWidth::Fullwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_char('@');
        e.push_char('.');
        assert_eq!(e.hiragana_text(), "＠．");
    }

    #[test]
    fn period_after_symbol_with_halfwidth_uses_ascii_period() {
        let config = crate::EngineConfig {
            symbol_width: crate::SymbolWidth::Halfwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_char('@');
        e.push_char('.');
        assert_eq!(e.hiragana_text(), "@.");
    }

    #[test]
    fn comma_after_kana_stays_touten() {
        // 直前が kana のときは従来通り `、` になる
        let config = crate::EngineConfig {
            alpha_width: crate::AlphaWidth::Fullwidth,
            symbol_width: crate::SymbolWidth::Fullwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.force_preedit("あ".to_string());
        e.push_char(',');
        assert_eq!(e.hiragana_text(), "あ、");
    }
}

#[cfg(test)]
mod digit_width_tests {
    use super::{DigitCandidateKind, DigitWidth, EngineConfig, RakunEngine};

    fn push_digit(width: DigitWidth, c: char) -> String {
        let config = EngineConfig {
            digit_width: width,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        e.push_char(c);
        e.hiragana_text().to_string()
    }

    #[test]
    fn halfwidth_keeps_ascii() {
        assert_eq!(push_digit(DigitWidth::Halfwidth, '0'), "0");
        assert_eq!(push_digit(DigitWidth::Halfwidth, '5'), "5");
        assert_eq!(push_digit(DigitWidth::Halfwidth, '9'), "9");
    }

    #[test]
    fn fullwidth_converts() {
        assert_eq!(push_digit(DigitWidth::Fullwidth, '0'), "０");
        assert_eq!(push_digit(DigitWidth::Fullwidth, '5'), "５");
        assert_eq!(push_digit(DigitWidth::Fullwidth, '9'), "９");
    }

    #[test]
    fn halfwidth_sequence() {
        let config = EngineConfig {
            digit_width: DigitWidth::Halfwidth,
            ..Default::default()
        };
        let mut e = RakunEngine::new(config);
        for c in "2024".chars() {
            e.push_char(c);
        }
        assert_eq!(e.hiragana_text(), "2024");
    }

    #[test]
    fn default_is_halfwidth() {
        assert_eq!(DigitWidth::default(), DigitWidth::Halfwidth);
        assert_eq!(push_digit(DigitWidth::default(), '3'), "3");
    }

    #[test]
    fn engine_config_deserialize_uses_new_digit_defaults() {
        let cfg: EngineConfig = serde_json::from_str(r#"{"num_candidates":5}"#).unwrap();
        assert!(cfg.digit_separator_auto);
        assert_eq!(
            cfg.digit_candidates_order,
            vec![
                DigitCandidateKind::Arabic,
                DigitCandidateKind::Fullwidth,
                DigitCandidateKind::Positional,
                DigitCandidateKind::PerDigit,
                DigitCandidateKind::Daiji,
            ]
        );
    }
}

#[cfg(test)]
mod candidate_merge_tests {
    use super::{EngineConfig, RakunEngine};
    use rakukan_dict::DictStore;
    use std::fs;

    #[test]
    fn merge_candidates_appends_hiragana_and_katakana_at_tail() {
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.force_preedit("てすと".to_string());

        let llm_candidates = (1..=8).map(|n| format!("候補{n}")).collect();
        let merged = engine.merge_candidates(llm_candidates, 40);

        // 通常候補を出し切った後ろに、ひらがな → カタカナ が常に付く。
        assert_eq!(merged.len(), 10, "merged={merged:?}");
        assert_eq!(merged[8], "てすと");
        assert_eq!(merged[9], "テスト");
    }

    #[test]
    fn merge_candidates_keeps_only_reading_when_no_candidates() {
        // 候補 0 件（Space 直後の同期パス / ライブ変換の打鍵途中）では読みだけ。
        // ここでカタカナまで足すと、TSF 側の「読み以外の候補があるか」判定が
        // 常に真になり、打鍵のたびに preview がカタカナに化ける。
        let mut engine = RakunEngine::new(EngineConfig::default());
        engine.force_preedit("てすと".to_string());

        let merged = engine.merge_candidates(vec![], 40);
        assert_eq!(merged, vec!["てすと".to_string()]);
    }

    #[test]
    fn merge_candidates_does_not_duplicate_original_reading() {
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.force_preedit("てすと".to_string());

        let mut llm_candidates: Vec<String> = (1..=7).map(|n| format!("候補{n}")).collect();
        llm_candidates.push("てすと".to_string());
        let merged = engine.merge_candidates(llm_candidates, 40);

        assert_eq!(merged.iter().filter(|c| c.as_str() == "てすと").count(), 1);
    }

    #[test]
    fn merge_candidates_uses_user_dict_even_without_llm_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let user_path = dir.path().join("user_dict.toml");
        fs::write(
            &user_path,
            r#"
[[entries]]
reading = "かっことじ"
surfaces = ["』"]
"#,
        )
        .unwrap();

        let store = DictStore::load(Some(&user_path), None, None).unwrap();
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.set_dict_store(store);
        engine.force_preedit("かっことじ".to_string());

        let merged = engine.merge_candidates(vec![], 40);

        assert_eq!(merged.first().map(String::as_str), Some("』"));
        assert!(merged.iter().any(|candidate| candidate == "かっことじ"));
    }

    #[test]
    fn merge_candidates_for_reading_uses_given_reading_not_internal_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let user_path = dir.path().join("user_dict.toml");
        fs::write(
            &user_path,
            r#"
[[entries]]
reading = "かっことじ"
surfaces = ["』"]
"#,
        )
        .unwrap();

        let store = DictStore::load(Some(&user_path), None, None).unwrap();
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.set_dict_store(store);
        engine.force_preedit("べつのよみ".to_string());

        let merged = engine.merge_candidates_for_reading("かっことじ", vec![], 40);

        assert_eq!(merged.first().map(String::as_str), Some("』"));
        assert!(merged.iter().any(|candidate| candidate == "かっことじ"));
        assert!(!merged.iter().any(|candidate| candidate == "べつのよみ"));
    }

    #[test]
    fn merge_candidates_places_low_priority_user_dict_after_learn_history() {
        let dir = tempfile::tempdir().unwrap();
        let user_path = dir.path().join("user_dict.toml");
        fs::write(
            &user_path,
            r#"
[[entries]]
reading = "みどり"
surfaces = ["ミドリ"]
priority = "low"
"#,
        )
        .unwrap();

        let store = DictStore::load(Some(&user_path), None, None).unwrap();
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.set_dict_store(store);

        // 学習前: 低優先エントリしか無いので先頭に出る
        let merged = engine.merge_candidates_for_reading("みどり", vec!["翠".to_string()], 40);
        assert_eq!(merged.first().map(String::as_str), Some("ミドリ"));

        // 「緑」を一度選んだことにすると、学習履歴が低優先エントリを追い越す
        engine.learn_force("みどり", "緑");
        let merged = engine.merge_candidates_for_reading("みどり", vec!["翠".to_string()], 40);
        let pos_learn = merged.iter().position(|c| c == "緑").unwrap();
        let pos_low = merged.iter().position(|c| c == "ミドリ").unwrap();
        let pos_llm = merged.iter().position(|c| c == "翠").unwrap();
        assert!(
            pos_learn < pos_low,
            "学習履歴は低優先ユーザー辞書より前: {merged:?}"
        );
        assert!(
            pos_low < pos_llm,
            "低優先ユーザー辞書は LLM 候補より前: {merged:?}"
        );
    }

    #[test]
    fn merge_candidates_normal_priority_still_outranks_learn_history() {
        let dir = tempfile::tempdir().unwrap();
        let user_path = dir.path().join("user_dict.toml");
        fs::write(
            &user_path,
            r#"
[[entries]]
reading = "りんぜ"
surfaces = ["凛世"]
"#,
        )
        .unwrap();

        let store = DictStore::load(Some(&user_path), None, None).unwrap();
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.set_dict_store(store);
        engine.learn_force("りんぜ", "臨済");

        let merged = engine.merge_candidates_for_reading("りんぜ", vec![], 40);
        assert_eq!(
            merged.first().map(String::as_str),
            Some("凛世"),
            "既定優先度のユーザー辞書は従来どおり最優先: {merged:?}"
        );
    }
}

#[cfg(test)]
mod passthrough_sync_tests {
    //! pending_romaji_buf と romaji.buffer の同期を検証する。
    //! PassThrough 連鎖で複数文字が確定する場合に、未確定ローマ字が
    //! 表示から落ちないことを保証する（旧バグ: "qwrty" → "qwry" 表示）。
    use super::{EngineConfig, RakunEngine};

    fn type_string(input: &str) -> RakunEngine {
        let mut e = RakunEngine::new(EngineConfig::default());
        for c in input.chars() {
            e.push_char(c);
        }
        e
    }

    #[test]
    fn qwrty_shows_all_typed_chars() {
        let e = type_string("qwrty");
        assert_eq!(e.current_preedit().display(), "qwrty");
    }

    #[test]
    fn kana_then_kq_shows_pending_q() {
        let e = type_string("kanakq");
        assert_eq!(e.current_preedit().display(), "かなkq");
    }

    #[test]
    fn kana_then_kq_then_bs_removes_q_only() {
        let mut e = type_string("kanakq");
        e.backspace();
        assert_eq!(e.current_preedit().display(), "かなk");
    }

    #[test]
    fn romaji_log_matches_typed_input_for_qwrty() {
        // F9/F10 復元のため、log + pending = ユーザーが入力したローマ字列 を保つ。
        let e = type_string("qwrty");
        let log = e.romaji_log_str();
        let pending = e.current_preedit().pending_romaji.clone();
        assert_eq!(format!("{}{}", log, pending), "qwrty");
    }

    fn engine_with_alpha_width(width: crate::AlphaWidth) -> RakunEngine {
        RakunEngine::new(EngineConfig {
            num_candidates: 9,
            alpha_width: width,
            ..Default::default()
        })
    }

    fn type_romaji(engine: &mut RakunEngine, romaji: &str) {
        for c in romaji.chars() {
            engine.push_char(c);
        }
    }

    #[test]
    fn romaji_literal_candidate_leads_when_reading_keeps_ascii() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Halfwidth);
        type_romaji(&mut engine, "claude");

        // "cl" はローマ字表に無いので 'c' が素通しになり、読みに ASCII が残る。
        let reading = engine.hiragana_text().to_string();
        assert!(
            reading.chars().any(|c| c.is_ascii_alphabetic()),
            "reading={reading:?}"
        );

        let merged = engine.merge_candidates(vec!["cぁうで".to_string()], 40);
        assert_eq!(merged.first().map(String::as_str), Some("claude"));
        assert_eq!(merged.get(1).map(String::as_str), Some("ｃｌａｕｄｅ"));
    }

    #[test]
    fn romaji_literal_candidate_leads_when_reading_is_all_ascii() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Halfwidth);
        type_romaji(&mut engine, "xyz");

        let reading = engine.hiragana_text().to_string();
        assert!(
            !reading.is_empty() && reading.chars().all(|c| c.is_ascii()),
            "reading={reading:?}"
        );

        let merged = engine.merge_candidates(vec![], 40);
        assert_eq!(merged.first().map(String::as_str), Some(reading.as_str()));
    }

    #[test]
    fn romaji_literal_candidate_order_follows_alpha_width() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Fullwidth);
        type_romaji(&mut engine, "claude");

        let merged = engine.merge_candidates(vec!["cぁうで".to_string()], 40);
        assert_eq!(merged.first().map(String::as_str), Some("ｃｌａｕｄｅ"));
        assert_eq!(merged.get(1).map(String::as_str), Some("claude"));
    }

    #[test]
    fn romaji_literal_candidate_is_not_promoted_for_normal_readings() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Halfwidth);
        type_romaji(&mut engine, "tudukete");
        assert_eq!(engine.hiragana_text(), "つづけて");

        // 読みに ASCII が残っていない普通の語では、英数候補は末尾の文字種候補
        // としてだけ出す。Space 直後は LLM がまだ返らず候補が空になりうるので、
        // 先頭に置くとライブ変換の preview を英字が奪う。
        // 候補 0 件のときは読みだけ（先頭を英字に奪われない）
        assert_eq!(engine.merge_candidates(vec![], 40), vec!["つづけて".to_string()]);

        // 通常候補があるときは、その後ろに文字種候補として並ぶ
        let merged = engine.merge_candidates(vec!["続けて".to_string()], 40);
        let expected: Vec<String> = ["続けて", "つづけて", "ツヅケテ", "tudukete", "ｔｕｄｕｋｅｔｅ"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(merged, expected);
    }

    #[test]
    fn char_type_candidates_follow_alpha_width_at_tail() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Fullwidth);
        type_romaji(&mut engine, "tudukete");

        let merged = engine.merge_candidates(vec!["続けて".to_string()], 40);
        let expected: Vec<String> = ["続けて", "つづけて", "ツヅケテ", "ｔｕｄｕｋｅｔｅ", "tudukete"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(merged, expected);
    }

    #[test]
    fn romaji_literal_candidate_is_skipped_for_partial_readings() {
        let mut engine = engine_with_alpha_width(crate::AlphaWidth::Halfwidth);
        type_romaji(&mut engine, "claude");

        // 文節分割された部分読みに対して、プリエディット全体のローマ字を出さない
        let merged =
            engine.merge_candidates_for_reading("べつのよみ", vec!["別の読み".to_string()], 40);
        assert!(
            !merged.iter().any(|c| c == "claude"),
            "merged={merged:?}"
        );
    }

    #[test]
    fn symbol_only_candidate_classification() {
        for s in ["¢", "£", "€", "°", "‰", "″", "°C", "「」", "→"] {
            assert!(crate::is_symbol_only_candidate(s), "{s:?} は記号扱いのはず");
        }
        for s in ["単位", "矢印", "カンイ", "たんい", "PC", "R18", "A"] {
            assert!(!crate::is_symbol_only_candidate(s), "{s:?} は語扱いのはず");
        }
        assert!(!crate::is_symbol_only_candidate(""));
    }

    #[test]
    fn n_insertion_opens_na_row_kana() {
        assert_eq!(crate::n_insertion_readings("げにん"), vec!["げんいん"]);
        assert_eq!(crate::n_insertion_readings("ふにき"), vec!["ふんいき"]);
        assert_eq!(crate::n_insertion_readings("れない"), vec!["れんあい"]);
        assert_eq!(crate::n_insertion_readings("せねん"), vec!["せんえん"]);
        // 拗音は 2 文字まとめて開く
        assert_eq!(crate::n_insertion_readings("きにょう"), vec!["きんよう"]);
    }

    #[test]
    fn n_insertion_skips_short_readings_and_leading_kana() {
        // 「たに → たんい」は踏み込みすぎなので 2 文字は対象外
        assert!(crate::n_insertion_readings("たに").is_empty());
        assert!(crate::n_insertion_readings("かに").is_empty());
        // 先頭のかなは開かない（「ん」で始まる読みを作らない）。
        // 「ないよう」の な は先頭なので候補が出ない。
        assert!(crate::n_insertion_readings("ないよう").is_empty());
        // な行が無ければ何も出さない
        assert!(crate::n_insertion_readings("かぎかっこ").is_empty());
    }

    #[test]
    fn n_insertion_enumerates_each_position() {
        // 「の」と「に」の両方をそれぞれ開いた読みが出る
        let out = crate::n_insertion_readings("このに");
        assert!(out.contains(&"こんおに".to_string()), "{out:?}");
        assert!(out.contains(&"このんい".to_string()), "{out:?}");
    }

    #[test]
    fn merge_adds_n_insertion_candidates_from_dictionary() {
        use crate::DictStore;
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let user_path = dir.path().join("user_dict.toml");
        fs::write(
            &user_path,
            r#"
[[entries]]
reading = "げんいん"
surfaces = ["原因"]
"#,
        )
        .unwrap();

        let store = DictStore::load(Some(&user_path), None, None).unwrap();
        let mut engine = RakunEngine::new(EngineConfig {
            num_candidates: 9,
            ..Default::default()
        });
        engine.set_dict_store(store);

        // n を 1 回しか打っていない読みでも、開いた読みの辞書候補が出る
        let merged = engine.merge_candidates_for_reading("げにん", vec!["下人".to_string()], 40);
        assert!(merged.iter().any(|c| c == "原因"), "merged={merged:?}");
        // 元の読みの正当な変換（LLM）を押しのけない
        let pos_llm = merged.iter().position(|c| c == "下人").unwrap();
        let pos_fix = merged.iter().position(|c| c == "原因").unwrap();
        assert!(pos_fix < pos_llm, "merged={merged:?}");
    }
}
