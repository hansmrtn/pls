use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

const APP_NAME: &str = "pls";
const DEFAULT_MODEL: &str = "llama3.1";
const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub embed_model: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub auto_reindex: bool,
    pub reindex_interval_days: u32,
    pub index_man_pages: bool,
    pub index_tldr: bool,
    pub index_help: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    pub confirm_by_default: bool,
    pub learn_from_history: bool,
    pub history_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub safe_commands: Vec<String>,
    pub dangerous_patterns: Vec<String>,
    pub max_output_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub style: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub index: IndexConfig,
    pub behavior: BehaviorConfig,
    pub safety: SafetyConfig,
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: DEFAULT_MODEL.to_string(),
                embed_model: DEFAULT_EMBED_MODEL.to_string(),
                endpoint: DEFAULT_OLLAMA_URL.to_string(),
            },
            index: IndexConfig {
                auto_reindex: true,
                reindex_interval_days: 7,
                index_man_pages: true,
                index_tldr: true,
                index_help: true,
            },
            behavior: BehaviorConfig {
                confirm_by_default: true,
                learn_from_history: true,
                history_window: 10,
            },
            safety: SafetyConfig {
                safe_commands: vec![
                    "ls", "cat", "head", "tail", "wc", "grep", "find", "du", "df", "ps", "echo",
                    "date", "pwd", "whoami", "which", "file", "stat", "uname", "hostname",
                    "uptime", "free", "id", "env", "printenv",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                dangerous_patterns: vec![
                    "rm -rf /",
                    "rm -rf /*",
                    "dd if=",
                    "mkfs",
                    "> /dev/sd",
                    "chmod -R 777 /",
                    "curl | sh",
                    "wget | sh",
                    ":(){ :|:& };:",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                max_output_lines: 100,
            },
            output: OutputConfig {
                style: "minimal".to_string(),
            },
        }
    }
}

pub fn get_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
        .join("config.toml")
}

pub fn load_config() -> Config {
    let path = get_config_path();
    if path.exists() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(config) = toml::from_str(&content) {
                return config;
            }
        }
    }
    Config::default()
}

pub fn save_config(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(config)?;
    fs::write(path, content)?;
    Ok(())
}
