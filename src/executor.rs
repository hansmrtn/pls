use std::process::{Command, Stdio};

pub fn execute_commands(
    commands: &[String],
    max_lines: usize,
) -> Result<(bool, String), Box<dyn std::error::Error>> {
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
        truncated.push(format!(
            "... [{} lines truncated] ...",
            output_lines.len() - max_lines
        ));
        truncated.extend(output_lines[output_lines.len() - max_lines / 2..].to_vec());
        truncated.join("\n")
    } else {
        output_lines.join("\n")
    };

    Ok((all_succeeded, output))
}
