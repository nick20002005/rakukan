//! ユーザー登録語辞書
//!
//! `%APPDATA%\rakukan\user_dict.toml` に TOML 形式で保存する。
//!
//! # ファイル形式
//! ```toml
//! [[entries]]
//! reading  = "きむら"
//! surfaces = ["木村", "金村"]   # 先頭が最優先候補
//!
//! [[entries]]
//! reading  = "らくかん"
//! surfaces = ["楽漢"]
//!
//! [[entries]]
//! reading  = "みどり"
//! surfaces = ["ミドリ"]
//! priority = "low"              # 学習履歴の後ろ・システム辞書の前に置く
//! ```
//!
//! `priority` は省略可（既定 `"normal"`）。`"low"` を指定したエントリは
//! 候補列の先頭を占有せず、学習履歴より後ろに回る。一般語と読みが衝突する
//! 固有名詞（カタカナ人名など）を大量に登録しても通常変換を壊さないための区分で、
//! 一度選べば学習履歴に載って前に出て、使わなくなれば学習スコアの減衰で戻る。

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct UserDict {
    #[serde(default)]
    pub entries: Vec<UserEntry>,
}

/// エントリの優先度。
///
/// - `Normal`: 候補列の最優先（従来どおり）。
/// - `Low`: 学習履歴の後ろ・システム辞書の前に挿入する。普段は沈んでいるが、
///   一度選べば学習履歴に載って前に出る。使わなくなれば学習スコアの減衰
///   （半減期 30 日）で元の位置に戻る。一般語と読みが衝突する固有名詞
///   （カタカナ人名など）を大量登録しても通常変換を壊さないための区分。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    #[default]
    Normal,
    Low,
}

impl Priority {
    /// TOML 出力から既定値を省くための述語（`skip_serializing_if` 用）。
    fn is_normal(&self) -> bool {
        matches!(self, Priority::Normal)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEntry {
    pub reading: String,
    pub surfaces: Vec<String>,
    #[serde(default, skip_serializing_if = "Priority::is_normal")]
    pub priority: Priority,
}

impl UserDict {
    /// ファイルから読み込む。ファイルが存在しない場合は空の辞書を返す。
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            debug!("user_dict: not found, using empty dict");
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let ud: Self =
            toml::from_str(&text).map_err(|e| anyhow::anyhow!("user_dict parse error: {e}"))?;
        info!(
            "user_dict: loaded {} entries from {}",
            ud.entries.len(),
            path.display()
        );
        Ok(ud)
    }

    /// ファイルに保存する
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("user_dict serialize error: {e}"))?;
        std::fs::write(path, text)?;
        debug!(
            "user_dict: saved {} entries to {}",
            self.entries.len(),
            path.display()
        );
        Ok(())
    }

    /// 読み → 候補リストの HashMap に変換する（優先度を区別しない全件）
    pub fn to_map(&self) -> HashMap<String, Vec<String>> {
        let mut map = HashMap::new();
        for entry in &self.entries {
            map.entry(entry.reading.clone())
                .or_insert_with(Vec::new)
                .extend(entry.surfaces.iter().cloned());
        }
        map
    }

    /// 優先度別に 2 つの HashMap へ分解する（DictStore 構築用）。
    ///
    /// 戻り値は `(normal, low)`。同じ読みが両優先度に存在してもよい。
    pub fn to_maps(&self) -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
        let mut normal = HashMap::new();
        let mut low = HashMap::new();
        for entry in &self.entries {
            let target = match entry.priority {
                Priority::Normal => &mut normal,
                Priority::Low => &mut low,
            };
            target
                .entry(entry.reading.clone())
                .or_insert_with(Vec::new)
                .extend(entry.surfaces.iter().cloned());
        }
        (normal, low)
    }

    /// エントリを追加または更新する
    /// 同じ reading が既にある場合は surfaces の先頭に挿入（重複除去）
    pub fn add(&mut self, reading: &str, surface: &str) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.reading == reading) {
            e.surfaces.retain(|s| s != surface);
            e.surfaces.insert(0, surface.to_string());
        } else {
            self.entries.push(UserEntry {
                reading: reading.to_string(),
                surfaces: vec![surface.to_string()],
                priority: Priority::default(),
            });
        }
    }

    /// エントリを削除する
    pub fn remove(&mut self, reading: &str, surface: &str) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.reading == reading) {
            e.surfaces.retain(|s| s != surface);
        }
        self.entries.retain(|e| !e.surfaces.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_add_and_save_load() {
        let mut ud = UserDict::default();
        ud.add("きむら", "木村");
        ud.add("きむら", "金村");
        assert_eq!(ud.entries[0].surfaces, vec!["金村", "木村"]);

        let f = NamedTempFile::new().unwrap();
        ud.save(f.path()).unwrap();

        let loaded = UserDict::load(f.path()).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].surfaces[0], "金村");
    }

    #[test]
    fn test_remove() {
        let mut ud = UserDict::default();
        ud.add("きむら", "木村");
        ud.add("きむら", "金村");
        ud.remove("きむら", "木村");
        assert_eq!(ud.entries[0].surfaces, vec!["金村"]);
    }

    #[test]
    fn test_priority_defaults_to_normal_and_is_omitted_on_save() {
        let mut ud = UserDict::default();
        ud.add("らくかん", "楽漢");
        assert_eq!(ud.entries[0].priority, Priority::Normal);

        let f = NamedTempFile::new().unwrap();
        ud.save(f.path()).unwrap();
        let text = std::fs::read_to_string(f.path()).unwrap();
        assert!(
            !text.contains("priority"),
            "既定値の priority は TOML に書き出さない: {text}"
        );
    }

    #[test]
    fn test_priority_low_round_trips() {
        let f = NamedTempFile::new().unwrap();
        std::fs::write(
            f.path(),
            r#"
[[entries]]
reading = "みどり"
surfaces = ["ミドリ"]
priority = "low"

[[entries]]
reading = "りんぜ"
surfaces = ["凛世"]
"#,
        )
        .unwrap();

        let ud = UserDict::load(f.path()).unwrap();
        assert_eq!(ud.entries.len(), 2);
        assert_eq!(ud.entries[0].priority, Priority::Low);
        assert_eq!(ud.entries[1].priority, Priority::Normal);

        // 保存し直しても low は残る
        let g = NamedTempFile::new().unwrap();
        ud.save(g.path()).unwrap();
        let reloaded = UserDict::load(g.path()).unwrap();
        assert_eq!(reloaded.entries[0].priority, Priority::Low);
        assert_eq!(reloaded.entries[1].priority, Priority::Normal);
    }

    #[test]
    fn test_to_maps_splits_by_priority() {
        let f = NamedTempFile::new().unwrap();
        std::fs::write(
            f.path(),
            r#"
[[entries]]
reading = "みどり"
surfaces = ["ミドリ"]
priority = "low"

[[entries]]
reading = "りんぜ"
surfaces = ["凛世"]
"#,
        )
        .unwrap();

        let ud = UserDict::load(f.path()).unwrap();
        let (normal, low) = ud.to_maps();
        assert_eq!(normal["りんぜ"], vec!["凛世"]);
        assert!(!normal.contains_key("みどり"));
        assert_eq!(low["みどり"], vec!["ミドリ"]);
        assert!(!low.contains_key("りんぜ"));

        // to_map は優先度を区別せず全件返す
        let all = ud.to_map();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_to_maps_merges_same_reading_within_priority() {
        let f = NamedTempFile::new().unwrap();
        std::fs::write(
            f.path(),
            r#"
[[entries]]
reading = "ゆう"
surfaces = ["ユウ"]
priority = "low"

[[entries]]
reading = "ゆう"
surfaces = ["ユゥ"]
priority = "low"
"#,
        )
        .unwrap();

        let ud = UserDict::load(f.path()).unwrap();
        let (_, low) = ud.to_maps();
        assert_eq!(low["ゆう"], vec!["ユウ", "ユゥ"]);
    }

    #[test]
    fn test_to_map() {
        let mut ud = UserDict::default();
        ud.add("らくかん", "楽漢");
        let map = ud.to_map();
        assert_eq!(map["らくかん"], vec!["楽漢"]);
    }
}
