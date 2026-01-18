use crate::config::IndexConfig;
use crate::db::save_tool;
use crate::ollama::OllamaClient;
use crate::types::Tool;
use std::{
    collections::HashMap,
    env,
    io::Write,
    process::{Command, Stdio},
};

fn discover_binaries() -> Vec<(String, String)> {
    let path_var = env::var("PATH").unwrap_or_default();
    let mut binaries = HashMap::new();

    for dir in path_var.split(':') {
        if let Ok(entries) = std::fs::read_dir(dir) {
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
        let text = if stdout.len() > stderr.len() {
            stdout
        } else {
            stderr
        };
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
    let Some(help) = help_text else {
        return String::new();
    };

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
    if tldr.is_some() {
        "tldr".to_string()
    } else if man.is_some() {
        "man".to_string()
    } else if help.is_some() {
        "help".to_string()
    } else {
        "inferred".to_string()
    }
}

pub fn index_tools(
    client: &OllamaClient,
    conn: &rusqlite::Connection,
    config: &IndexConfig,
    verbose: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let binaries = discover_binaries();
    let total = binaries.len();
    let mut indexed = 0;

    let priority_tools: Vec<&str> = vec![
        "find", "grep", "awk", "sed", "sort", "uniq", "cut", "tr", "wc", "head", "tail", "cat",
        "less", "more", "ls", "pwd", "mkdir", "rmdir", "rm", "cp", "mv", "chmod", "chown", "ln",
        "tar", "gzip", "gunzip", "zip", "unzip", "curl", "wget", "ssh", "scp", "rsync", "ps",
        "top", "htop", "kill", "killall", "df", "du", "free", "uname", "hostname", "date", "cal",
        "bc", "expr", "xargs", "tee", "diff", "comm", "join", "paste", "split", "file", "stat",
        "touch", "strings", "jq", "yq", "fd", "rg", "bat", "exa", "fzf", "ag", "ack", "ncdu",
        "tree", "git", "docker", "kubectl", "make", "cargo", "npm", "pip", "python",
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

        let man_desc = if config.index_man_pages {
            get_man_description(&name)
        } else {
            None
        };
        let help_text = if config.index_help {
            get_tool_help(&name)
        } else {
            None
        };
        let tldr = if config.index_tldr {
            get_tldr_content(&name)
        } else {
            None
        };

        let description = man_desc
            .clone()
            .or_else(|| {
                help_text
                    .as_ref()
                    .map(|h| h.lines().next().unwrap_or("").to_string())
            })
            .unwrap_or_default();

        let synopsis = help_text
            .as_ref()
            .map(|h| parse_help_synopsis(h))
            .unwrap_or_default();
        let examples = extract_examples(&tldr, &help_text);
        let flags = extract_flags(&help_text);
        let source = determine_source(&tldr, &man_desc, &help_text);

        let embed_text = format!(
            "{} {} {} {}",
            name,
            description,
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
