use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

// ============================================================================
// Configuration
// ============================================================================

const APP_NAME: &str = "pls";
const DEFAULT_MODEL: &str = "llama3.1";
const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const TOP_K_TOOLS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmConfig {
    provider: String,
    model: String,
    embed_model: String,
    endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexConfig {
    auto_reindex: bool,
    reindex_interval_days: u32,
    index_man_pages: bool,
    index_tldr: bool,
    index_help: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BehaviorConfig {
    confirm_by_default: bool,
    learn_from_history: bool,
    history_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SafetyConfig {
    safe_commands: Vec<String>,
    dangerous_patterns: Vec<String>,
    max_output_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutputConfig {
    style: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    llm: LlmConfig,
    index: IndexConfig,
    behavior: BehaviorConfig,
    safety: SafetyConfig,
    output: OutputConfig,
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
                    "ls", "cat", "head", "tail", "wc", "grep", "find", "du", "df",
                    "ps", "echo", "date", "pwd", "whoami", "which", "file", "stat",
                    "uname", "hostname", "uptime", "free", "id", "env", "printenv",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                dangerous_patterns: vec![
                    "rm -rf /", "rm -rf /*", "dd if=", "mkfs", "> /dev/sd",
                    "chmod -R 777 /", "curl | sh", "wget | sh", ":(){ :|:& };:",
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

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Clone)]
struct Tool {
    name: String,
    path: String,
    description: String,
    synopsis: String,
    examples: String,
    flags: String,
    source: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Plan {
    commands: Vec<String>,
    explanation: String,
    warnings: Vec<String>,
    needs_confirmation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RiskLevel {
    Safe,
    Review,
    Dangerous,
    Blocked,
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    query: String,
    commands: Vec<String>,
    executed: bool,
    succeeded: bool,
}

// ============================================================================
// Ollama Client
// ============================================================================

#[derive(Serialize)]
struct OllamaGenerate {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Serialize)]
struct OllamaEmbed {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

struct OllamaClient {
    base_url: String,
    model: String,
    embed_model: String,
    client: reqwest::blocking::Client,
}

impl OllamaClient {
    fn new(config: &LlmConfig) -> Self {
        Self {
            base_url: config.endpoint.clone(),
            model: config.model.clone(),
            embed_model: config.embed_model.clone(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn generate(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/api/generate", self.base_url);
        let body = OllamaGenerate {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };
        let resp: OllamaGenerateResponse = self.client.post(&url).json(&body).send()?.json()?;
        Ok(resp.response)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let url = format!("{}/api/embed", self.base_url);
        let body = OllamaEmbed {
            model: self.embed_model.clone(),
            input: text.to_string(),
        };
        let resp: OllamaEmbedResponse = self.client.post(&url).json(&body).send()?.json()?;
        Ok(resp.embeddings.into_iter().next().unwrap_or_default())
    }

    fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().is_ok()
    }
}

// ============================================================================
// Database / Index
// ============================================================================

fn get_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
}

fn get_db_path() -> PathBuf {
    get_data_dir().join("index").join("tools.db")
}

fn get_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
        .join("config.toml")
}

fn load_config() -> Config {
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

fn save_config(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(config)?;
    fs::write(path, content)?;
    Ok(())
}

fn init_db(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tools (
            name TEXT PRIMARY KEY,
            path TEXT,
            description TEXT,
            synopsis TEXT,
            examples TEXT,
            flags TEXT,
            embedding BLOB,
            source TEXT,
            updated_at INTEGER
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            plan TEXT,
            executed INTEGER,
            succeeded INTEGER,
            output_sample TEXT,
            timestamp INTEGER
        )",
        [],
    )?;

    Ok(())
}

fn save_tool(conn: &Connection, tool: &Tool) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_bytes: Vec<u8> = tool
        .embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;

    conn.execute(
        "INSERT OR REPLACE INTO tools (name, path, description, synopsis, examples, flags, embedding, source, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            tool.name, tool.path, tool.description, tool.synopsis,
            tool.examples, tool.flags, embedding_bytes, tool.source, now
        ],
    )?;
    Ok(())
}

fn load_all_tools(conn: &Connection) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT name, path, description, synopsis, examples, flags, embedding, source FROM tools",
    )?;

    let tools = stmt
        .query_map([], |row| {
            let embedding_bytes: Vec<u8> = row.get(6)?;
            let embedding: Vec<f32> = embedding_bytes
                .chunks(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect();

            Ok(Tool {
                name: row.get(0)?,
                path: row.get(1)?,
                description: row.get(2)?,
                synopsis: row.get(3)?,
                examples: row.get(4)?,
                flags: row.get(5)?,
                source: row.get(7)?,
                embedding,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(tools)
}

fn save_history(
    conn: &Connection,
    query: &str,
    commands: &[String],
    executed: bool,
    succeeded: bool,
    output_sample: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    let plan_json = serde_json::to_string(commands)?;

    conn.execute(
        "INSERT INTO history (query, plan, executed, succeeded, output_sample, timestamp)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![query, plan_json, executed as i32, succeeded as i32, output_sample, now],
    )?;
    Ok(())
}

fn get_recent_history(conn: &Connection, limit: usize) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT query, plan, executed, succeeded FROM history ORDER BY timestamp DESC LIMIT ?1",
    )?;

    let entries = stmt
        .query_map(params![limit as i64], |row| {
            let plan_json: String = row.get(1)?;
            let commands: Vec<String> = serde_json::from_str(&plan_json).unwrap_or_default();
            Ok(HistoryEntry {
                query: row.get(0)?,
                commands,
                executed: row.get::<_, i32>(2)? != 0,
                succeeded: row.get::<_, i32>(3)? != 0,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(entries)
}

fn get_last_command(conn: &Connection) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let result: Result<String, _> = conn.query_row(
        "SELECT plan FROM history WHERE executed = 1 ORDER BY timestamp DESC LIMIT 1",
        [],
        |row| row.get(0),
    );

    match result {
        Ok(plan_json) => {
            let commands: Vec<String> = serde_json::from_str(&plan_json).unwrap_or_default();
            Ok(commands.into_iter().next())
        }
        Err(_) => Ok(None),
    }
}

fn get_tool_count(conn: &Connection) -> usize {
    conn.query_row("SELECT COUNT(*) FROM tools", [], |row| row.get(0))
        .unwrap_or(0)
}

// ============================================================================
// Tool Discovery & Indexing
// ============================================================================

fn discover_binaries() -> Vec<(String, String)> {
    let path_var = env::var("PATH").unwrap_or_default();
    let mut binaries = HashMap::new();

    for dir in path_var.split(':') {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if !name.starts_with('.') && !binaries.contains_key(name) {
                            binaries.insert(name.to_string(), path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    binaries.into_iter().collect()
}

fn get_tool_help(name: &str) -> Option<String> {
    if let Ok(output) = Command::new(name)
        .arg("--help")
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let text = if stdout.len() > stderr.len() { stdout } else { stderr };
        if text.len() > 20 {
            return Some(text.chars().take(2000).collect());
        }
    }

    if let Ok(output) = Command::new(name)
        .arg("-h")
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        if text.len() > 20 {
            return Some(text.chars().take(2000).collect());
        }
    }

    None
}

fn get_man_description(name: &str) -> Option<String> {
    if let Ok(output) = Command::new("whatis").arg(name).output() {
        let text = String::from_utf8_lossy(&output.stdout);
        if output.status.success() && !text.is_empty() {
            return Some(text.trim().to_string());
        }
    }
    None
}

fn get_tldr_content(name: &str) -> Option<String> {
    if let Ok(output) = Command::new("tldr").arg(name).output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn parse_help_synopsis(help_text: &str) -> String {
    for line in help_text.lines().take(10) {
        let lower = line.to_lowercase();
        if lower.starts_with("usage:") || lower.starts_with("synopsis:") {
            return line.to_string();
        }
    }
    help_text.lines().next().unwrap_or("").to_string()
}

fn extract_flags(help_text: &Option<String>) -> String {
    let Some(help) = help_text else { return String::new() };

    let mut flags = Vec::new();
    for line in help.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('-') && !trimmed.starts_with("---") {
            if let Some(flag) = trimmed.split_whitespace().next() {
                flags.push(flag.trim_end_matches(',').to_string());
            }
        }
    }
    flags.into_iter().take(20).collect::<Vec<_>>().join(", ")
}

fn extract_examples(tldr: &Option<String>, help: &Option<String>) -> String {
    if let Some(tldr) = tldr {
        let examples: Vec<&str> = tldr
            .lines()
            .filter(|l| l.trim().starts_with('-') || l.trim().starts_with('`'))
            .take(5)
            .collect();
        if !examples.is_empty() {
            return examples.join("\n");
        }
    }

    if let Some(help) = help {
        let lower = help.to_lowercase();
        if let Some(pos) = lower.find("example") {
            let snippet: String = help[pos..].chars().take(500).collect();
            return snippet.lines().take(5).collect::<Vec<_>>().join("\n");
        }
    }

    String::new()
}

fn determine_source(tldr: &Option<String>, man: &Option<String>, help: &Option<String>) -> String {
    if tldr.is_some() { "tldr".to_string() }
    else if man.is_some() { "man".to_string() }
    else if help.is_some() { "help".to_string() }
    else { "inferred".to_string() }
}

fn index_tools(
    client: &OllamaClient,
    conn: &Connection,
    config: &IndexConfig,
    verbose: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let binaries = discover_binaries();
    let total = binaries.len();
    let mut indexed = 0;

    let priority_tools: Vec<&str> = vec![
        "find", "grep", "awk", "sed", "sort", "uniq", "cut", "tr", "wc", "head", "tail",
        "cat", "less", "more", "ls", "pwd", "mkdir", "rmdir", "rm", "cp", "mv",
        "chmod", "chown", "ln", "tar", "gzip", "gunzip", "zip", "unzip", "curl", "wget",
        "ssh", "scp", "rsync", "ps", "top", "htop", "kill", "killall", "df", "du",
        "free", "uname", "hostname", "date", "cal", "bc", "expr", "xargs", "tee",
        "diff", "comm", "join", "paste", "split", "file", "stat", "touch", "strings",
        "jq", "yq", "fd", "rg", "bat", "exa", "fzf", "ag", "ack", "ncdu", "tree",
        "git", "docker", "kubectl", "make", "cargo", "npm", "pip", "python",
    ];

    let mut sorted_binaries: Vec<_> = binaries.into_iter().collect();
    sorted_binaries.sort_by(|(a, _), (b, _)| {
        let a_priority = priority_tools.iter().position(|&t| t == a);
        let b_priority = priority_tools.iter().position(|&t| t == b);
        match (a_priority, b_priority) {
            (Some(ap), Some(bp)) => ap.cmp(&bp),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.cmp(b),
        }
    });

    let max_tools = 200;

    for (i, (name, path)) in sorted_binaries.into_iter().take(max_tools).enumerate() {
        if verbose {
            eprint!("\r  [{}/{}] {}...", i + 1, total.min(max_tools), name);
            std::io::stderr().flush().ok();
        }

        let man_desc = if config.index_man_pages { get_man_description(&name) } else { None };
        let help_text = if config.index_help { get_tool_help(&name) } else { None };
        let tldr = if config.index_tldr { get_tldr_content(&name) } else { None };

        let description = man_desc.clone()
            .or_else(|| help_text.as_ref().map(|h| h.lines().next().unwrap_or("").to_string()))
            .unwrap_or_default();

        let synopsis = help_text.as_ref().map(|h| parse_help_synopsis(h)).unwrap_or_default();
        let examples = extract_examples(&tldr, &help_text);
        let flags = extract_flags(&help_text);
        let source = determine_source(&tldr, &man_desc, &help_text);

        let embed_text = format!(
            "{} {} {} {}",
            name, description,
            synopsis.chars().take(200).collect::<String>(),
            examples.chars().take(300).collect::<String>()
        );

        let embedding = match client.embed(&embed_text) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let tool = Tool {
            name: name.clone(),
            path,
            description,
            synopsis,
            examples,
            flags,
            source,
            embedding,
        };

        save_tool(conn, &tool)?;
        indexed += 1;
    }

    if verbose {
        eprintln!("\r  indexed {} tools                        ", indexed);
    }

    Ok(indexed)
}

// ============================================================================
// RAG Retrieval
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn retrieve_relevant_tools(
    client: &OllamaClient,
    conn: &Connection,
    query: &str,
    top_k: usize,
) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    let query_embedding = client.embed(query)?;
    let all_tools = load_all_tools(conn)?;

    let mut scored: Vec<(f32, Tool)> = all_tools
        .into_iter()
        .map(|tool| (cosine_similarity(&query_embedding, &tool.embedding), tool))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(top_k).map(|(_, t)| t).collect())
}

// ============================================================================
// Plan Generation
// ============================================================================

fn build_prompt(query: &str, tools: &[Tool], cwd: &str, shell: &str) -> String {
    let tool_docs: String = tools
        .iter()
        .map(|t| {
            let mut doc = format!("### {}\n", t.name);
            if !t.description.is_empty() {
                doc.push_str(&format!("  {}\n", t.description));
            }
            if !t.synopsis.is_empty() {
                doc.push_str(&format!("  Usage: {}\n", t.synopsis));
            }
            if !t.flags.is_empty() {
                doc.push_str(&format!("  Flags: {}\n", t.flags));
            }
            if !t.examples.is_empty() {
                doc.push_str(&format!("  Examples:\n{}\n", t.examples));
            }
            doc
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"You are a Unix command line expert. Generate a shell command to accomplish the task.

AVAILABLE TOOLS:
{tool_docs}

STRICT RULES:
1. Use ONLY tools and flags shown above. Do not invent flags.
2. If you need a tool not listed, say "I need <tool> which is not available"
3. Use simple, common patterns. Prefer find, grep, awk, sort, uniq, wc.
4. For counting lines of code: use find to get files, xargs wc -l
5. For file sizes: use du -sh or find with -size
6. Always use relative paths from current directory

EXAMPLES OF GOOD COMMANDS:
- Count lines by extension: find . -name "*.rs" | xargs wc -l
- Find large files: find . -size +10M -type f
- Disk usage: du -sh */ | sort -h
- Find and count: find . -type f -name "*.log" | wc -l

Current directory: {cwd}

TASK: {query}

Respond with ONLY this JSON, no other text:
{{"commands": ["the command"], "explanation": "what it does", "warnings": [], "needs_confirmation": true}}"#,
        tool_docs = tool_docs, cwd = cwd, query = query
    )
}

fn parse_plan(response: &str) -> Result<Plan, Box<dyn std::error::Error>> {
    let response = response.trim();
    let start = response.find('{');
    let end = response.rfind('}');

    let json_str = match (start, end) {
        (Some(s), Some(e)) if e > s => &response[s..=e],
        _ => response,
    };

    let parsed: serde_json::Value = serde_json::from_str(json_str)?;

    Ok(Plan {
        commands: parsed["commands"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        explanation: parsed["explanation"]
            .as_str()
            .unwrap_or("Execute the command(s)")
            .to_string(),
        warnings: parsed["warnings"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        needs_confirmation: parsed["needs_confirmation"].as_bool().unwrap_or(true),
    })
}

fn generate_plan(
    client: &OllamaClient,
    conn: &Connection,
    query: &str,
) -> Result<Plan, Box<dyn std::error::Error>> {
    let tools = retrieve_relevant_tools(client, conn, query, TOP_K_TOOLS)?;
    if tools.is_empty() {
        return Err("No tools indexed. Run 'pls index' first.".into());
    }

    let cwd = env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let shell = env::var("SHELL").unwrap_or_else(|_| "bash".to_string());

    let prompt = build_prompt(query, &tools, &cwd, &shell);
    let response = client.generate(&prompt)?;
    parse_plan(&response)
}

// ============================================================================
// Safety Assessment
// ============================================================================

fn assess_risk(commands: &[String], config: &SafetyConfig) -> RiskLevel {
    let full_command = commands.join(" ");

    for pattern in &config.dangerous_patterns {
        if full_command.contains(pattern) {
            return RiskLevel::Blocked;
        }
    }

    let dangerous_cmds = ["rm", "dd", "mkfs", "fdisk", "parted", "shred"];
    for cmd in &dangerous_cmds {
        if commands.iter().any(|c| c.split_whitespace().next() == Some(cmd)) {
            return RiskLevel::Dangerous;
        }
    }

    let all_safe = commands.iter().all(|cmd| {
        let first = cmd.split_whitespace().next().unwrap_or("");
        let base = first.rsplit('/').next().unwrap_or(first);
        config.safe_commands.contains(&base.to_string())
    });

    if all_safe { RiskLevel::Safe } else { RiskLevel::Review }
}

// ============================================================================
// Execution
// ============================================================================

fn execute_commands(commands: &[String], max_lines: usize) -> Result<(bool, String), Box<dyn std::error::Error>> {
    let mut output_lines = Vec::new();
    let mut all_succeeded = true;

    for cmd in commands {
        let result = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;

        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);

        if !stdout.is_empty() {
            output_lines.extend(stdout.lines().map(String::from));
        }
        if !stderr.is_empty() {
            output_lines.extend(stderr.lines().map(String::from));
        }

        if !result.status.success() {
            all_succeeded = false;
        }
    }

    let output = if output_lines.len() > max_lines {
        let mut truncated: Vec<String> = output_lines[..max_lines / 2].to_vec();
        truncated.push(format!("... [{} lines truncated] ...", output_lines.len() - max_lines));
        truncated.extend(output_lines[output_lines.len() - max_lines / 2..].to_vec());
        truncated.join("\n")
    } else {
        output_lines.join("\n")
    };

    Ok((all_succeeded, output))
}

// ============================================================================
// UI
// ============================================================================

fn print_plan(plan: &Plan, risk: RiskLevel) {
    println!();

    for (i, cmd) in plan.commands.iter().enumerate() {
        if plan.commands.len() > 1 {
            println!("  {}. {}", i + 1, cmd);
        } else {
            println!("  {}", cmd);
        }
    }

    if risk == RiskLevel::Dangerous {
        println!();
        println!("  warning: this command may be destructive");
    }

    for warning in &plan.warnings {
        println!("  warning: {}", warning);
    }
}

fn print_blocked(plan: &Plan) {
    println!();
    for cmd in &plan.commands {
        println!("  {}", cmd);
    }
    println!();
    println!("  refused: command blocked for safety");
}

fn show_explanation(plan: &Plan) {
    println!();
    println!("explanation: {}", plan.explanation);
    println!();

    for cmd in &plan.commands {
        let parts: Vec<&str> = cmd.split('|').collect();
        for part in parts {
            let trimmed = part.trim();
            println!("  {}", trimmed);
        }
    }
    println!();
}

fn prompt_action() -> Option<char> {
    println!("[enter] run  [e] edit  [?] explain  [q] quit");

    // Simple blocking read from stdin
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok()?;

    let input = input.trim().to_lowercase();
    match input.as_str() {
        "" => Some('r'),
        "e" => Some('e'),
        "?" => Some('?'),
        "q" => Some('q'),
        _ => Some('q'),
    }
}

fn edit_command(cmd: &str) -> Option<String> {
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let temp_path = "/tmp/pls_edit.sh";
    fs::write(temp_path, cmd).ok()?;
    Command::new(&editor).arg(temp_path).status().ok()?;
    fs::read_to_string(temp_path).ok()
}

// ============================================================================
// CLI Commands
// ============================================================================

fn cmd_index(config: &Config, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("indexing system tools...");

    let client = OllamaClient::new(&config.llm);

    if !client.is_available() {
        eprintln!("error: cannot connect to ollama");
        eprintln!("  start it with: ollama serve");
        return Err("ollama not available".into());
    }

    let db_path = get_db_path();
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;

    let count = index_tools(&client, &conn, &config.index, verbose)?;

    println!("done: {} tools indexed", count);
    println!("  db: {:?}", db_path);

    Ok(())
}

fn cmd_stats() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no index found. run 'pls index' first.");
        return Ok(());
    }

    let conn = Connection::open(&db_path)?;
    let count = get_tool_count(&conn);
    let size_kb = fs::metadata(&db_path)?.len() / 1024;

    println!("index stats:");
    println!("  tools: {}", count);
    println!("  size:  {} KB", size_kb);
    println!("  path:  {:?}", db_path);

    Ok(())
}

fn cmd_history(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no history yet.");
        return Ok(());
    }

    let conn = Connection::open(&db_path)?;
    let entries = get_recent_history(&conn, config.behavior.history_window)?;

    if entries.is_empty() {
        println!("no history yet.");
        return Ok(());
    }

    println!("recent queries:");
    println!();

    for entry in entries {
        let status = if entry.executed {
            if entry.succeeded { "+" } else { "x" }
        } else {
            "-"
        };

        println!("{} {}", status, entry.query);
        for cmd in &entry.commands {
            println!("    {}", cmd);
        }
        println!();
    }

    Ok(())
}

fn cmd_edit_last(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no history yet.");
        return Ok(());
    }

    let conn = Connection::open(&db_path)?;

    match get_last_command(&conn)? {
        Some(cmd) => {
            if let Some(edited) = edit_command(&cmd) {
                let edited = edited.trim();
                if !edited.is_empty() {
                    println!("edited: {}", edited);
                    let (succeeded, output) = execute_commands(&[edited.to_string()], config.safety.max_output_lines)?;
                    println!("{}", output);
                    save_history(&conn, "[edited]", &[edited.to_string()], true, succeeded, &output)?;
                }
            }
        }
        None => {
            println!("no previous command to edit.");
        }
    }

    Ok(())
}

fn cmd_doctor(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("diagnostics:");
    println!();

    let client = OllamaClient::new(&config.llm);

    print!("  ollama ... ");
    std::io::stdout().flush().ok();
    if client.is_available() {
        println!("ok");
    } else {
        println!("failed");
        println!("    url: {}", config.llm.endpoint);
        println!("    try: ollama serve");
    }

    print!("  model ({}) ... ", config.llm.model);
    std::io::stdout().flush().ok();
    match client.generate("Say 'ok' and nothing else.") {
        Ok(_) => println!("ok"),
        Err(e) => {
            println!("failed");
            println!("    error: {}", e);
            println!("    try: ollama pull {}", config.llm.model);
        }
    }

    print!("  embeddings ({}) ... ", config.llm.embed_model);
    std::io::stdout().flush().ok();
    match client.embed("test") {
        Ok(_) => println!("ok"),
        Err(e) => {
            println!("failed");
            println!("    error: {}", e);
            println!("    try: ollama pull {}", config.llm.embed_model);
        }
    }

    let db_path = get_db_path();
    print!("  index ... ");
    std::io::stdout().flush().ok();
    if db_path.exists() {
        let conn = Connection::open(&db_path)?;
        let count = get_tool_count(&conn);
        if count > 0 {
            println!("ok ({} tools)", count);
        } else {
            println!("empty");
            println!("    run: pls index");
        }
    } else {
        println!("not found");
        println!("    run: pls index");
    }

    let config_path = get_config_path();
    print!("  config ... ");
    std::io::stdout().flush().ok();
    if config_path.exists() {
        println!("ok");
    } else {
        println!("using defaults");
    }

    println!();
    Ok(())
}

fn cmd_config() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = get_config_path();

    if !config_path.exists() {
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        save_config(&Config::default())?;
    }

    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    Command::new(&editor).arg(&config_path).status()?;

    Ok(())
}

fn cmd_query(
    query: &str,
    config: &Config,
    yolo: bool,
    explain_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new(&config.llm);

    if !client.is_available() {
        eprintln!("error: cannot connect to ollama");
        return Err("ollama not available".into());
    }

    let db_path = get_db_path();
    if !db_path.exists() {
        eprintln!("no index found. running initial indexing...");
        cmd_index(config, true)?;
    }

    let conn = Connection::open(&db_path)?;
    init_db(&conn)?;

    eprint!("thinking...");
    std::io::stderr().flush().ok();

    let plan = generate_plan(&client, &conn, query)?;

    eprint!("\r           \r");

    if plan.commands.is_empty() {
        println!("could not generate a plan for this task.");
        println!("  {}", plan.explanation);
        return Ok(());
    }

    let risk = assess_risk(&plan.commands, &config.safety);

    if risk == RiskLevel::Blocked {
        print_blocked(&plan);
        return Ok(());
    }

    if explain_only {
        print_plan(&plan, risk);
        show_explanation(&plan);
        return Ok(());
    }

    if yolo && risk == RiskLevel::Safe {
        let (succeeded, output) = execute_commands(&plan.commands, config.safety.max_output_lines)?;
        println!("{}", output);
        save_history(&conn, query, &plan.commands, true, succeeded, &output)?;
        return Ok(());
    }

    print_plan(&plan, risk);

    loop {
        match prompt_action() {
            Some('r') => {
                let (succeeded, output) = execute_commands(&plan.commands, config.safety.max_output_lines)?;
                println!("{}", output);
                save_history(&conn, query, &plan.commands, true, succeeded, &output)?;
                break;
            }
            Some('e') => {
                let combined = plan.commands.join(" && ");
                if let Some(edited) = edit_command(&combined) {
                    let edited = edited.trim();
                    if !edited.is_empty() {
                        let new_commands = vec![edited.to_string()];
                        let new_risk = assess_risk(&new_commands, &config.safety);

                        if new_risk == RiskLevel::Blocked {
                            println!("refused: command blocked for safety");
                            continue;
                        }

                        println!("edited: {}", edited);
                        let (succeeded, output) = execute_commands(&new_commands, config.safety.max_output_lines)?;
                        println!("{}", output);
                        save_history(&conn, query, &new_commands, true, succeeded, &output)?;
                        break;
                    }
                }
            }
            Some('?') => show_explanation(&plan),
            Some('q') | None => {
                save_history(&conn, query, &plan.commands, false, false, "")?;
                println!("cancelled.");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn print_usage() {
    println!(
        r#"pls - a polite AI assistant that speaks fluent Unix

usage:
  pls <query>         ask pls to do something
  pls -y <query>      yolo mode (skip confirmation)
  pls -e <query>      explain only, don't run
  pls --edit          edit and re-run last command
  pls --history       show recent queries
  pls index           index system tools
  pls index --stats   show index statistics
  pls config          edit configuration
  pls doctor          check system status

examples:
  pls find large files in my home directory
  pls show processes using the most memory
  pls -y count lines of code in this project
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = load_config();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let result = match args[1].as_str() {
        "index" => {
            if args.get(2).map(|s| s.as_str()) == Some("--stats") {
                cmd_stats()
            } else {
                cmd_index(&config, true)
            }
        }
        "config" => cmd_config(),
        "doctor" => cmd_doctor(&config),
        "--history" | "history" => cmd_history(&config),
        "--edit" | "edit" => cmd_edit_last(&config),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        _ => {
            let mut yolo = false;
            let mut explain = false;
            let mut query_parts = Vec::new();

            for arg in &args[1..] {
                match arg.as_str() {
                    "-y" | "--yolo" => yolo = true,
                    "-e" | "--explain" => explain = true,
                    _ => query_parts.push(arg.clone()),
                }
            }

            let query = query_parts.join(" ");
            if query.is_empty() {
                print_usage();
                Ok(())
            } else {
                cmd_query(&query, &config, yolo, explain)
            }
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
