use crate::types::{Plan, RiskLevel};
use std::{env, fs, process::Command};

pub fn print_plan(plan: &Plan, risk: RiskLevel) {
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

pub fn print_blocked(plan: &Plan) {
    println!();
    for cmd in &plan.commands {
        println!("  {}", cmd);
    }
    println!();
    println!("  refused: command blocked for safety");
}

pub fn show_explanation(plan: &Plan) {
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

pub fn prompt_action() -> Option<char> {
    println!("[enter] run  [e] edit  [?] explain  [q] quit");

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

pub fn edit_command(cmd: &str) -> Option<String> {
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let temp_path = "/tmp/pls_edit.sh";
    fs::write(temp_path, cmd).ok()?;
    Command::new(&editor).arg(temp_path).status().ok()?;
    fs::read_to_string(temp_path).ok()
}
