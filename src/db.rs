use crate::types::{HistoryEntry, Tool};
use rusqlite::{params, Connection};
use std::path::PathBuf;

const APP_NAME: &str = "pls";

fn get_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP_NAME)
}

pub fn get_db_path() -> PathBuf {
    get_data_dir().join("index").join("tools.db")
}

pub fn init_db(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
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

pub fn save_tool(conn: &Connection, tool: &Tool) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_bytes: Vec<u8> = tool
        .embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;

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

pub fn load_all_tools(conn: &Connection) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
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

pub fn save_history(
    conn: &Connection,
    query: &str,
    commands: &[String],
    executed: bool,
    succeeded: bool,
    output_sample: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;
    let plan_json = serde_json::to_string(commands)?;

    conn.execute(
        "INSERT INTO history (query, plan, executed, succeeded, output_sample, timestamp)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            query,
            plan_json,
            executed as i32,
            succeeded as i32,
            output_sample,
            now
        ],
    )?;
    Ok(())
}

pub fn get_recent_history(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error>> {
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

pub fn get_last_command(conn: &Connection) -> Result<Option<String>, Box<dyn std::error::Error>> {
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

pub fn get_tool_count(conn: &Connection) -> u32 {
    conn.query_row("SELECT COUNT(*) FROM tools", [], |row| row.get(0))
        .unwrap_or(0)
}
