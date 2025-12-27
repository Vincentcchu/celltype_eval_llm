# Configuration

This directory contains configuration files for the cell-type standardization system.

## config.json

Main configuration file. Key settings:

- **openai.api_key**: Your OpenAI API key (alternatively set `OPENAI_API_KEY` environment variable)
- **openai.model**: Model to use for semantic matching (default: "gpt-4o-mini")
- **paths**: Locations of mapping store and L3 vocabulary files
- **standardization/evaluation**: Default parameters for workflows

## Setup

1. Copy `config.json` to a local version if needed
2. Set your OpenAI API key:
   - Option A: Edit `config.json` and set `openai.api_key`
   - Option B: Set environment variable: `export OPENAI_API_KEY=your-key-here`

## Security Note

Never commit API keys to version control. Add your local config files to `.gitignore` if they contain sensitive information.
