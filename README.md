# pls

```
$ pls find duplicate files by comparing checksums
```

Natural language to shell commands via local LLM.

You know what you want to do, you just can't remember
if it's `find -exec` or `xargs`, whether `grep` needs `-E` or `-P`, or how
`awk` field separators work. pls figures out the incantation.

## Synopsis

```
pls [-y] [-e] <query>
pls index [--stats]
pls config
pls doctor
pls --history
pls --edit
```

## Description

pls translates natural language into shell commands. It indexes the tools
installed on your system (via `--help`, `man`, `tldr`) and uses RAG to ground
the LLM, so it only suggests commands and flags that actually exist, most of the time.

Requires Ollama running locally. Model quality matters -- small models
hallucinate flags and other strange things.

## Installation

```
ollama pull gemma3:4b
ollama pull nomic-embed-text

cargo build --release
cp target/release/pls ~/.local/bin/
```

## Usage

```
$ pls index
indexed 250 tools

$ pls find python files larger than 1mb

  find . -name "*.py" -size +1M

[enter] run  [e] edit  [?] explain  [q] quit

$ pls -y show listening ports
tcp  0  0 0.0.0.0:22   0.0.0.0:*  LISTEN
tcp  0  0 0.0.0.0:5432 0.0.0.0:*  LISTEN

$ pls -e count lines of code by language

  find . -name "*.py" -o -name "*.rs" -o -name "*.go" | xargs wc -l | sort -n

explanation: finds source files and counts lines, sorted by count
```

## Options

```
-y, --yolo     YOLO it for safe commands
-e, --explain  show plan without executing
```

## Commands

```
index          index system tools (run once, or after installing new tools)
index --stats  show index statistics  
config         edit configuration file
doctor         check ollama connection and index status
--history      show recent queries
--edit         edit and re-run last command
```

## Files

```
~/.local/share/pls/index/tools.db   tool index (sqlite + embeddings)
~/.config/pls/config.toml           configuration
```

## Configuration

```toml
[llm]
model = "llama3.1"
embed_model = "nomic-embed-text"
endpoint = "http://localhost:11434"

[safety]
safe_commands = ["ls", "cat", "grep", ...]
dangerous_patterns = ["rm -rf /", ...]
max_output_lines = 100

[behavior]
confirm_by_default = true
learn_from_history = true
```

## How it works

1. `pls index` scans $PATH, extracts help text, embeds each tool
2. your query gets embedded and matched against the index  
3. LLM sees only the top-k relevant tools and their documented flags
4. you see the plan, hit enter to run

## License

MIT
