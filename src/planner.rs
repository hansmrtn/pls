use crate::ollama::OllamaClient;
use crate::retrieval::retrieve_relevant_tools;
use crate::types::{Plan, Tool};
use std::env;

const TOP_K_TOOLS: usize = 8;

fn build_prompt(query: &str, tools: &[Tool], cwd: &str, _shell: &str) -> String {
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
        tool_docs = tool_docs,
        cwd = cwd,
        query = query
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
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        explanation: parsed["explanation"]
            .as_str()
            .unwrap_or("Execute the command(s)")
            .to_string(),
        warnings: parsed["warnings"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        needs_confirmation: parsed["needs_confirmation"].as_bool().unwrap_or(true),
    })
}

pub fn generate_plan(
    client: &OllamaClient,
    conn: &rusqlite::Connection,
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
