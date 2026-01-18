use crate::config::SafetyConfig;
use crate::types::RiskLevel;

pub fn assess_risk(commands: &[String], config: &SafetyConfig) -> RiskLevel {
    let full_command = commands.join(" ");

    for pattern in &config.dangerous_patterns {
        if full_command.contains(pattern) {
            return RiskLevel::Blocked;
        }
    }

    let dangerous_cmds = ["rm", "dd", "mkfs", "fdisk", "parted", "shred"];
    for cmd in &dangerous_cmds {
        if commands
            .iter()
            .any(|c| c.split_whitespace().next() == Some(cmd))
        {
            return RiskLevel::Dangerous;
        }
    }

    let all_safe = commands.iter().all(|cmd| {
        let first = cmd.split_whitespace().next().unwrap_or("");
        let base = first.rsplit('/').next().unwrap_or(first);
        config.safe_commands.contains(&base.to_string())
    });

    if all_safe {
        RiskLevel::Safe
    } else {
        RiskLevel::Review
    }
}
