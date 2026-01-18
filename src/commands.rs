use crate::config::{save_config, Config};
use crate::db::{
    get_db_path, get_last_command, get_recent_history, get_tool_count, init_db, save_history,
};
use crate::executor::execute_commands;
use crate::index::index_tools;
use crate::ollama::OllamaClient;
use crate::planner::generate_plan;
use crate::safety::assess_risk;
use crate::types::RiskLevel;
use crate::ui::{edit_command, print_blocked, print_plan, prompt_action, show_explanation};
use std::{env, fs, io::Write, process::Command};

pub fn cmd_index(config: &Config, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
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

    let conn = rusqlite::Connection::open(&db_path)?;
    init_db(&conn)?;

    let count = index_tools(&client, &conn, &config.index, verbose)?;

    println!("done: {} tools indexed", count);
    println!("  db: {:?}", db_path);

    Ok(())
}

pub fn cmd_stats() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no index found. run 'pls index' first.");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;
    let count = get_tool_count(&conn);
    let size_kb = fs::metadata(&db_path)?.len() / 1024;

    println!("index stats:");
    println!("  tools: {}", count);
    println!("  size:  {} KB", size_kb);
    println!("  path:  {:?}", db_path);

    Ok(())
}

pub fn cmd_history(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no history yet.");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;
    let entries = get_recent_history(&conn, config.behavior.history_window)?;

    if entries.is_empty() {
        println!("no history yet.");
        return Ok(());
    }

    println!("recent queries:");
    println!();

    for entry in entries {
        let status = if entry.executed {
            if entry.succeeded {
                "+"
            } else {
                "x"
            }
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

pub fn cmd_edit_last(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_db_path();

    if !db_path.exists() {
        println!("no history yet.");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;

    match get_last_command(&conn)? {
        Some(cmd) => {
            if let Some(edited) = edit_command(&cmd) {
                let edited = edited.trim();
                if !edited.is_empty() {
                    println!("edited: {}", edited);
                    let (succeeded, output) =
                        execute_commands(&[edited.to_string()], config.safety.max_output_lines)?;
                    println!("{}", output);
                    save_history(
                        &conn,
                        "[edited]",
                        &[edited.to_string()],
                        true,
                        succeeded,
                        &output,
                    )?;
                }
            }
        }
        None => {
            println!("no previous command to edit.");
        }
    }

    Ok(())
}

pub fn cmd_doctor(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
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
        let conn = rusqlite::Connection::open(&db_path)?;
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

    let config_path = crate::config::get_config_path();
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

pub fn cmd_config() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = crate::config::get_config_path();

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

pub fn cmd_query(
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

    let conn = rusqlite::Connection::open(&db_path)?;
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
                let (succeeded, output) =
                    execute_commands(&plan.commands, config.safety.max_output_lines)?;
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
                        let (succeeded, output) =
                            execute_commands(&new_commands, config.safety.max_output_lines)?;
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
