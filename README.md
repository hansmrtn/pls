# `pls`

Natural language to shell commands via local LLM.

## Synopsis

```sh
    pls [-y] [-e] <query>
    pls index [--stats]
    pls config
    pls doctor
```

## Description

`pls` translates natural language into shell commands. It indexes the tools
installed on your system (via `--help`, `man`, `tldr`) and uses RAG to ensure
the LLM only suggests commands and flags that actually exist.

Requires Ollama running locally.

## Installation

```sh 
    ollama pull llama3.2
    ollama pull nomic-embed-text
    
    cargo build --release
    cp target/release/pls ~/.local/bin/
```

## Usage

```sh
    $ pls index
    250 tools indexed

    $ pls find python files larger than 1mb
      find . -name "*.py" -size +1M
    [enter]run [e]dit [?]explain [q]uit 

    $ pls -y show listening ports
    tcp  0  0 0.0.0.0:22   0.0.0.0:*  LISTEN
    tcp  0  0 0.0.0.0:5432 0.0.0.0:*  LISTEN

    $ pls -e count lines of code by language
    find . -name "*.py" -o -name "*.rs" -o -name "*.go" | xargs wc -l | sort -n
```

## Options

```sh
    -y, --yolo     skip confirmation for safe commands
    -e, --explain  show plan without executing
```

## Commands

```sh
    index          index system tools (run once, or after installing new tools)
    index --stats  show index statistics
    config         edit configuration file
    doctor         check ollama connection and index status
```

## Files

```
    ~/.local/share/pls/tools.db    tool index (sqlite)
    ~/.config/pls/config.toml      configuration
```

## Configuration

```
    ollama_url = "http://localhost:11434"
    model = "llama3.2"
    embed_model = "nomic-embed-text"
    yolo_mode = false
    safe_commands = ["ls", "cat", "grep", ...]
    dangerous_patterns = ["rm -rf /", ...]
```

## How it works

1. Query is embedded via ollama
2. Top-k similar tools retrieved from RAG index
3. LLM generates command using only those tools and their documented flags
4. User confirms, edits, or cancels
5. Command executes, output displayed

## License

MIT
