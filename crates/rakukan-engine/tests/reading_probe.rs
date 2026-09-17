//! 使い捨て: 単語を「ひらがなモードのまま打った」ときの読みを出す。
//! 辞書登録の reading を手で推測しないための確認用。
//! `cargo test --release -p rakukan-engine --features vulkan --test reading_probe -- --nocapture`

use rakukan_engine::{AlphaWidth, EngineConfig, RakunEngine, SymbolWidth};

/// (表記, そのまま打った場合, 全部小文字で打った場合も見るか)
const WORDS: &[(&str, bool)] = &[
    ("seedream", false),
    ("WaveSpeed", true),
    ("ComfyUI", true),
    ("NovelAI", true),
    ("ChatGPT", true),
    ("Codex", true),
    ("Gemini", true),
    ("Claude", true),
    ("Opus", true),
    ("Sonnet", true),
    ("LoRA", true),
    ("Qwen", true),
    ("Wan", true),
    ("MiniMax", true),
    ("SenseNova", true),
    ("ControlNet", true),
    ("DLsite", true),
    ("pixiv", false),
    ("Photoshop", true),
    ("Blender", true),
    ("VRChat", true),
    ("prompt", false),
    ("upscale", false),
    ("inpaint", false),
    ("checkpoint", false),
    ("sampler", false),
];

fn reading_of(word: &str) -> String {
    let config = EngineConfig {
        alpha_width: AlphaWidth::Halfwidth,
        symbol_width: SymbolWidth::Fullwidth,
        ..Default::default()
    };
    let mut e = RakunEngine::new(config);
    for c in word.chars() {
        if c.is_ascii_uppercase() {
            e.push_fullwidth_alpha(c);
        } else {
            e.push_char(c);
        }
    }
    // 未確定のローマ字はプリエディット上そのまま見えるので読みに足す
    let st = e.current_preedit();
    format!("{}{}", st.hiragana, st.pending_romaji)
}

#[test]
fn print_readings() {
    for (w, also_lower) in WORDS {
        println!("{}\t{}", w, reading_of(w));
        if *also_lower {
            let lower = w.to_ascii_lowercase();
            let r = reading_of(&lower);
            println!("{}\t{}\t(小文字 {})", w, r, lower);
        }
    }
}
