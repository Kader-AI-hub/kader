# Configuration

Kader can be configured through environment variables, YAML files, and the `.kader` directory in your home folder.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KADER_DIR` | Kader config directory (default: `~/.kader`) | No |
| `OLLAMA_API_KEY` | Ollama Cloud API key (get from https://ollama.com/settings) | For Ollama Cloud |
| `GEMINI_API_KEY` | Google Gemini API key | For Google Provider |
| `MISTRAL_API_KEY` | Mistral API key | For Mistral Provider |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Anthropic Provider |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI Provider |
| `MOONSHOT_API_KEY` | Moonshot AI key | For Moonshot Provider |
| `ZAI_API_KEY` | Z.ai (GLM) API key | For Z.ai Provider |
| `OPENROUTER_API_KEY` | OpenRouter key | For OpenRouter Provider |
| `OPENCODE_API_KEY` | OpenCode API key | For OpenCode Provider |
| `GROQ_API_KEY` | Groq API key | For Groq Provider |

## .kader Directory

When the kader module is imported for the first time, it automatically creates a `.kader` directory in your home directory:

```
~/.kader/
├── .env                   # Environment variables
├── memory/               # Memory and session storage
│   └── sessions/         # Saved conversation sessions
├── skills/               # User-level skills
├── commands/             # User-level special commands
└── KADER.md              # Agent instructions file
```

## YAML Agent Configuration

BaseAgent supports loading configuration from YAML files:

```yaml
# agent.yaml
name: ConfiguredAgent
system_prompt: "You are a helpful coding assistant."
tools:
  - read_file
  - write_file
  - todo_tool
provider:
  model: llama3.2
  provider: ollama
persistence: true
retry_attempts: 3
interrupt_before_tool: true
```

```python
agent = BaseAgent.from_yaml("agent.yaml")
```

## Provider Configuration Format

### Ollama (Local)

```yaml
provider:
  provider: ollama
  model: llama3.2
  base_url: "http://localhost:11434"
  timeout: 120
```

### Ollama (Cloud)

```yaml
provider:
  provider: ollama
  model: minimax-m2.5
  base_url: "https://ollama.com"
  # api_key: "your-ollama-api-key"  # Or set OLLAMA_API_KEY env var
```

### Google Gemini

```yaml
provider:
  provider: google
  model: gemini-2.0-flash
  temperature: 0.7
  max_tokens: 2048
```

### Anthropic

```yaml
provider:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
```

### OpenAI-Compatible

```yaml
provider:
  provider: openai
  model: gpt-4o
  base_url: "https://api.openai.com/v1"
  api_key: "your-openai-key"
```

For Groq, OpenRouter, etc., use the appropriate `base_url`.

## Session Storage

Sessions are saved to `~/.kader/memory/sessions/` and include:
- Conversation history
- Agent state
- Tool execution logs
- Sub-agent contexts (for aggregated context)
