use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal,
};
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

const APP_NAME: &str = "pls";
const DEFAULT_MODEL: &str = "llama3.2";
const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const TOP_K_TOOLS: usize = 10;
const MAX_OUTPUT_LINES: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    ollama_url: String,
    model: String,
    embed_model: String,
    yolo_mode: bool,
    safe_commands: Vec<String>,
    dangerous_patterns: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            yolo_mode: false,
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
        }
    }
}

#[derive(Debug, Clone)]
struct Tool {
    name: String,
    path: String,
    description: String,
    synopsis: String,
    flags: String,
    examples: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Plan {
    commands: Vec<String>,
    explanation: String,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RiskLevel {
    Safe,
    Review,
    Dangerous,
    Blocked,
}

// Ollama API

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
    fn new(base_url: &str, model: &str, embed_model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            embed_model: embed_model.to_string(),
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

// Paths

fn get_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
}

fn get_db_path() -> PathBuf {
    get_data_dir().join("tools.db")
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
    fs::write(path, toml::to_string_pretty(config)?)?;
    Ok(())
}

// Database

fn init_db(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tools (
            name TEXT PRIMARY KEY,
            path TEXT,
            description TEXT,
            synopsis TEXT,
            flags TEXT,
            examples TEXT,
            embedding BLOB,
            updated_at INTEGER
        )",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            commands TEXT,
            succeeded INTEGER,
            timestamp INTEGER
        )",
        [],
    )?;
    Ok(())
}

fn save_tool(conn: &Connection, tool: &Tool) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_bytes: Vec<u8> = tool.embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    conn.execute(
        "INSERT OR REPLACE INTO tools (name, path, description, synopsis, flags, examples, embedding, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![tool.name, tool.path, tool.description, tool.synopsis, tool.flags, tool.examples, embedding_bytes, now],
    )?;
    Ok(())
}

fn load_all_tools(conn: &Connection) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT name, path, description, synopsis, flags, examples, embedding FROM tools",
    )?;
    let tools = stmt
        .query_map([], |row| {
            let embedding_bytes: Vec<u8> = row.get(6)?;
            let embedding: Vec<f32> = embedding_bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                .collect();
            Ok(Tool {
                name: row.get(0)?,
                path: row.get(1)?,
                description: row.get(2)?,
                synopsis: row.get(3)?,
                flags: row.get(4)?,
                examples: row.get(5)?,
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
    succeeded: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    conn.execute(
        "INSERT INTO history (query, commands, succeeded, timestamp) VALUES (?1, ?2, ?3, ?4)",
        params![query, serde_json::to_string(commands)?, succeeded as i32, now],
    )?;
    Ok(())
}

fn get_tool_count(conn: &Connection) -> usize {
    conn.query_row("SELECT COUNT(*) FROM tools", [], |row| row.get(0))
        .unwrap_or(0)
}

// Tool discovery

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
    for flag in &["--help", "-h"] {
        if let Ok(output) = Command::new(name)
            .arg(flag)
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let text = if stdout.len() > stderr.len() { stdout } else { stderr };
            if text.len() > 20 {
                return Some(text.chars().take(4000).collect());
            }
        }
    }
    None
}

fn get_man_description(name: &str) -> Option<String> {
    Command::new("whatis")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn get_tldr_content(name: &str) -> Option<String> {
    Command::new("tldr")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .filter(|s| !s.is_empty())
}

fn parse_help_synopsis(help_text: &str) -> String {
    for line in help_text.lines().take(15) {
        let lower = line.to_lowercase();
        if lower.contains("usage:") {
            return line.trim().to_string();
        }
    }
    help_text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").to_string()
}

fn extract_flags(help_text: &str) -> String {
    let mut flags = Vec::new();
    for line in help_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('-') && !trimmed.starts_with("---") {
            flags.push(trimmed.chars().take(120).collect::<String>());
            if flags.len() >= 25 {
                break;
            }
        }
    }
    flags.join("\n")
}

fn extract_examples(tldr: &Option<String>, help: &Option<String>) -> String {
    if let Some(tldr) = tldr {
        let examples: Vec<&str> = tldr
            .lines()
            .filter(|l| l.trim().starts_with('-') || l.trim().starts_with('`'))
            .take(6)
            .collect();
        if !examples.is_empty() {
            return examples.join("\n");
        }
    }
    if let Some(help) = help {
        if let Some(pos) = help.to_lowercase().find("example") {
            return help[pos..].chars().take(600).collect::<String>()
                .lines().take(8).collect::<Vec<_>>().join("\n");
        }
    }
    String::new()
}

fn index_tools(client: &OllamaClient, conn: &Connection, verbose: bool) -> Result<usize, Box<dyn std::error::Error>> {
    let binaries = discover_binaries();
    let total = binaries.len();
    let mut indexed = 0;

    let priority: Vec<&str> = vec![
        "find", "grep", "awk", "sed", "sort", "uniq", "cut", "tr", "wc", "head", "tail",
        "cat", "less", "ls", "pwd", "mkdir", "rmdir", "rm", "cp", "mv", "chmod", "chown",
        "ln", "tar", "gzip", "gunzip", "zip", "unzip", "curl", "wget", "ssh", "scp", "rsync",
        "ps", "top", "htop", "kill", "killall", "df", "du", "free", "uname", "hostname",
        "date", "cal", "bc", "expr", "xargs", "tee", "diff", "comm", "join", "paste",
        "split", "file", "stat", "touch", "strings", "od", "xxd", "base64", "md5sum",
        "sha256sum", "jq", "yq", "fd", "rg", "bat", "exa", "eza", "fzf", "ag", "ack",
        "ncdu", "tree", "watch", "timeout", "parallel", "git", "docker", "kubectl",
        "make", "cargo", "npm", "pip", "python", "python3", "node", "lsof", "netstat",
        "ss", "ip", "ifconfig", "ping", "traceroute", "dig", "nslookup", "nc", "nmap",
    ];

    let mut sorted: Vec<_> = binaries.into_iter().collect();
    sorted.sort_by(|(a, _), (b, _)| {
        let ap = priority.iter().position(|&t| t == a);
        let bp = priority.iter().position(|&t| t == b);
        match (ap, bp) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            _ => a.cmp(b),
        }
    });

    let max_tools = 250;
    for (i, (name, path)) in sorted.into_iter().take(max_tools).enumerate() {
        if verbose {
            // Use \r to return to start of line, then print with fixed width padding to clear previous text
            eprint!("\r{:70}\r[{}/{}] {}", "", i + 1, total.min(max_tools), name);
            std::io::stderr().flush().ok();
        }

        let man_desc = get_man_description(&name);
        let help_text = get_tool_help(&name);
        let tldr = get_tldr_content(&name);

        let description = man_desc.clone()
            .or_else(|| help_text.as_ref().and_then(|h| h.lines().next().map(String::from)))
            .unwrap_or_default();
        let synopsis = help_text.as_ref().map(|h| parse_help_synopsis(h)).unwrap_or_default();
        let flags = help_text.as_ref().map(|h| extract_flags(h)).unwrap_or_default();
        let examples = extract_examples(&tldr, &help_text);

        let embed_text = format!("{} {} {} {}", name, description, synopsis, examples.chars().take(500).collect::<String>());
        let embedding = match client.embed(&embed_text) {
            Ok(e) => e,
            Err(_) => continue,
        };

        save_tool(conn, &Tool { name, path, description, synopsis, flags, examples, embedding })?;
        indexed += 1;
    }

    if verbose {
        eprintln!("\r{:70}\r{} tools indexed", "", indexed);
    }
    Ok(indexed)
}

// RAG

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn retrieve_relevant_tools(client: &OllamaClient, conn: &Connection, query: &str, top_k: usize) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    let query_embedding = client.embed(query)?;
    let all_tools = load_all_tools(conn)?;
    let mut scored: Vec<_> = all_tools.into_iter()
        .map(|t| (cosine_similarity(&query_embedding, &t.embedding), t))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(top_k).map(|(_, t)| t).collect())
}

// Plan generation

fn build_prompt(query: &str, tools: &[Tool], cwd: &str) -> String {
    let tool_docs: String = tools.iter().map(|t| {
        let mut doc = format!("=== {} ===\n", t.name);
        if !t.description.is_empty() {
            doc.push_str(&format!("Description: {}\n", t.description));
        }
        if !t.synopsis.is_empty() {
            doc.push_str(&format!("Synopsis: {}\n", t.synopsis));
        }
        if !t.flags.is_empty() {
            doc.push_str(&format!("Flags:\n{}\n", t.flags));
        }
        if !t.examples.is_empty() {
            doc.push_str(&format!("Examples:\n{}\n", t.examples));
        }
        doc
    }).collect::<Vec<_>>().join("\n");

    format!(r#"You are a shell command generator. You ONLY output JSON. No explanations. No markdown. No text before or after the JSON.

AVAILABLE TOOLS (you can ONLY use these):
{tool_docs}

RULES:
1. ONLY use tools listed above.
2. ONLY use flags shown in each tool's Flags section.
3. If the task is impossible with these tools, set commands to empty array.

task: {query}
cwd: {cwd}

Respond with ONLY this JSON structure, nothing else:
{{"commands":["shell command here"],"explanation":"what it does","warnings":[]}}"#, tool_docs=tool_docs, cwd=cwd, query=query)
}

fn parse_plan(response: &str) -> Result<Plan, Box<dyn std::error::Error>> {
    let response = response.trim();
    
    // Find JSON object boundaries
    let start = response.find('{');
    let end = response.rfind('}');
    
    let json_str = match (start, end) {
        (Some(s), Some(e)) if e > s => &response[s..=e],
        _ => {
            eprintln!("pls: no JSON found in response");
            eprintln!("response: {}", &response.chars().take(200).collect::<String>());
            return Err("invalid response from model".into());
        }
    };
    
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pls: failed to parse JSON: {}", e);
            eprintln!("json: {}", &json_str.chars().take(300).collect::<String>());
            return Err("invalid JSON from model".into());
        }
    };
    
    Ok(Plan {
        commands: parsed["commands"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        explanation: parsed["explanation"].as_str().unwrap_or("").to_string(),
        warnings: parsed["warnings"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
    })
}

fn generate_plan(client: &OllamaClient, conn: &Connection, query: &str) -> Result<Plan, Box<dyn std::error::Error>> {
    let tools = retrieve_relevant_tools(client, conn, query, TOP_K_TOOLS)?;
    if tools.is_empty() {
        return Err("no tools indexed; run 'pls index'".into());
    }
    let cwd = env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| ".".to_string());
    let prompt = build_prompt(query, &tools, &cwd);
    
    eprint!("thinking...");
    std::io::stderr().flush().ok();
    let response = client.generate(&prompt)?;
    eprint!("\r           \r");
    std::io::stderr().flush().ok();
    
    parse_plan(&response)
}

// Safety

fn assess_risk(commands: &[String], config: &Config) -> RiskLevel {
    let full = commands.join(" ");
    for pattern in &config.dangerous_patterns {
        if full.contains(pattern) {
            return RiskLevel::Blocked;
        }
    }
    let dangerous = ["rm", "dd", "mkfs", "fdisk", "parted", "shred"];
    for d in &dangerous {
        if commands.iter().any(|c| c.split_whitespace().next() == Some(*d)) {
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

// Execution

fn execute_commands(commands: &[String]) -> Result<(bool, String), Box<dyn std::error::Error>> {
    let mut lines = Vec::new();
    let mut ok = true;
    for cmd in commands {
        let result = Command::new("sh").arg("-c").arg(cmd)
            .stdout(Stdio::piped()).stderr(Stdio::piped()).output()?;
        lines.extend(String::from_utf8_lossy(&result.stdout).lines().map(String::from));
        lines.extend(String::from_utf8_lossy(&result.stderr).lines().map(String::from));
        if !result.status.success() { ok = false; }
    }
    let output = if lines.len() > MAX_OUTPUT_LINES {
        let mut t = lines[..MAX_OUTPUT_LINES/2].to_vec();
        t.push(format!("... {} lines omitted ...", lines.len() - MAX_OUTPUT_LINES));
        t.extend(lines[lines.len()-MAX_OUTPUT_LINES/2..].to_vec());
        t.join("\n")
    } else {
        lines.join("\n")
    };
    Ok((ok, output))
}

// UI

fn prompt_action() -> Option<char> {
    eprint!("[enter]run [e]dit [?]explain [q]uit ");
    std::io::stderr().flush().ok();
    terminal::enable_raw_mode().ok()?;
    let result = loop {
        if event::poll(std::time::Duration::from_millis(100)).ok()? {
            if let Event::Key(k) = event::read().ok()? {
                match k.code {
                    KeyCode::Enter | KeyCode::Char('y') => break Some('r'),
                    KeyCode::Char('e') => break Some('e'),
                    KeyCode::Char('?') => break Some('?'),
                    KeyCode::Char('q') | KeyCode::Char('n') | KeyCode::Esc => break Some('q'),
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break Some('q'),
                    _ => {}
                }
            }
        }
    };
    terminal::disable_raw_mode().ok();
    eprintln!();
    result
}

fn edit_command(cmd: &str) -> Option<String> {
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let path = "/tmp/pls_edit.sh";
    fs::write(path, cmd).ok()?;
    Command::new(&editor).arg(path).status().ok()?;
    fs::read_to_string(path).ok()
}

// Commands

fn cmd_index(config: &Config, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new(&config.ollama_url, &config.model, &config.embed_model);
    if !client.is_available() {
        eprintln!("pls: cannot connect to ollama at {}", config.ollama_url);
        return Err("ollama unavailable".into());
    }
    fs::create_dir_all(get_data_dir())?;
    let conn = Connection::open(get_db_path())?;
    init_db(&conn)?;
    index_tools(&client, &conn, verbose)?;
    Ok(())
}

fn cmd_stats() -> Result<(), Box<dyn std::error::Error>> {
    let path = get_db_path();
    if !path.exists() {
        println!("no index; run 'pls index'");
        return Ok(());
    }
    let conn = Connection::open(&path)?;
    let meta = fs::metadata(&path)?;
    println!("tools: {}", get_tool_count(&conn));
    println!("size:  {} KB", meta.len() / 1024);
    println!("path:  {}", path.display());
    Ok(())
}

fn cmd_doctor(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new(&config.ollama_url, &config.model, &config.embed_model);
    print!("ollama: ");
    if client.is_available() { println!("ok") } else { println!("FAIL") }
    print!("model: ");
    match client.generate("say ok") { Ok(_) => println!("ok"), Err(_) => println!("FAIL") }
    print!("embed: ");
    match client.embed("test") { Ok(_) => println!("ok"), Err(_) => println!("FAIL") }
    print!("index: ");
    let path = get_db_path();
    if path.exists() {
        let conn = Connection::open(&path)?;
        let n = get_tool_count(&conn);
        if n > 0 { println!("ok ({} tools)", n) } else { println!("empty") }
    } else {
        println!("missing");
    }
    Ok(())
}

fn cmd_config() -> Result<(), Box<dyn std::error::Error>> {
    let path = get_config_path();
    if !path.exists() {
        if let Some(p) = path.parent() { fs::create_dir_all(p)?; }
        save_config(&Config::default())?;
    }
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    Command::new(&editor).arg(&path).status()?;
    Ok(())
}

fn cmd_query(query: &str, config: &Config, yolo: bool, explain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new(&config.ollama_url, &config.model, &config.embed_model);
    if !client.is_available() {
        eprintln!("pls: ollama unavailable");
        return Err("ollama unavailable".into());
    }
    let path = get_db_path();
    if !path.exists() {
        eprintln!("pls: indexing...");
        cmd_index(config, true)?;
    }
    let conn = Connection::open(&path)?;
    init_db(&conn)?;

    let plan = generate_plan(&client, &conn, query)?;
    if plan.commands.is_empty() {
        eprintln!("pls: no plan generated");
        return Ok(());
    }

    let risk = assess_risk(&plan.commands, config);
    if risk == RiskLevel::Blocked {
        for c in &plan.commands { eprintln!("  {}", c); }
        eprintln!("pls: blocked");
        return Ok(());
    }

    if explain {
        if !plan.explanation.is_empty() { println!("{}", plan.explanation); }
        for c in &plan.commands { println!("  {}", c); }
        return Ok(());
    }

    if yolo && risk == RiskLevel::Safe {
        let (ok, out) = execute_commands(&plan.commands)?;
        print!("{}", out);
        save_history(&conn, query, &plan.commands, ok)?;
        return Ok(());
    }

    for c in &plan.commands { println!("  {}", c); }
    for w in &plan.warnings { eprintln!("warning: {}", w); }
    if risk == RiskLevel::Dangerous { eprintln!("warning: destructive command"); }

    loop {
        match prompt_action() {
            Some('r') => {
                let (ok, out) = execute_commands(&plan.commands)?;
                print!("{}", out);
                save_history(&conn, query, &plan.commands, ok)?;
                break;
            }
            Some('e') => {
                if let Some(ed) = edit_command(&plan.commands.join(" && ")) {
                    let ed = ed.trim();
                    if !ed.is_empty() {
                        let cmds = vec![ed.to_string()];
                        if assess_risk(&cmds, config) == RiskLevel::Blocked {
                            eprintln!("pls: blocked");
                            continue;
                        }
                        let (ok, out) = execute_commands(&cmds)?;
                        print!("{}", out);
                        save_history(&conn, query, &cmds, ok)?;
                        break;
                    }
                }
            }
            Some('?') => {
                if !plan.explanation.is_empty() { println!("{}", plan.explanation); }
            }
            Some('q') | None => break,
            _ => {}
        }
    }
    Ok(())
}

fn print_usage() {
    eprintln!("usage: pls [-y] [-e] <query>");
    eprintln!("       pls index [--stats]");
    eprintln!("       pls config | doctor");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let config = load_config();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let result = match args[1].as_str() {
        "index" => if args.get(2).map(|s| s.as_str()) == Some("--stats") { cmd_stats() } else { cmd_index(&config, true) },
        "config" => cmd_config(),
        "doctor" => cmd_doctor(&config),
        "-h" | "--help" | "help" => { print_usage(); Ok(()) }
        _ => {
            let mut yolo = false;
            let mut explain = false;
            let mut parts = Vec::new();
            for a in &args[1..] {
                match a.as_str() {
                    "-y" | "--yolo" => yolo = true,
                    "-e" | "--explain" => explain = true,
                    _ => parts.push(a.clone()),
                }
            }
            let q = parts.join(" ");
            if q.is_empty() { print_usage(); Ok(()) } else { cmd_query(&q, &config, yolo || config.yolo_mode, explain) }
        }
    };

    if let Err(e) = result {
        eprintln!("pls: {}", e);
        std::process::exit(1);
    }
}
