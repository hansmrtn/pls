use std::env;

mod commands;
mod config;
mod db;
mod executor;
mod index;
mod ollama;
mod planner;
mod retrieval;
mod safety;
mod types;
mod ui;

fn print_usage() {
    println!(
        r#"pls - a CLI assistant that speaks Unix

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
    let config = config::load_config();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let result = match args[1].as_str() {
        "index" => {
            if args.get(2).map(|s| s.as_str()) == Some("--stats") {
                commands::cmd_stats()
            } else {
                commands::cmd_index(&config, true)
            }
        }
        "config" => commands::cmd_config(),
        "doctor" => commands::cmd_doctor(&config),
        "--history" | "history" => commands::cmd_history(&config),
        "--edit" | "edit" => commands::cmd_edit_last(&config),
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
                commands::cmd_query(&query, &config, yolo, explain)
            }
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
