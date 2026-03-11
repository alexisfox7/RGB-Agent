# Read-Grep-Bash Agent

An agent for [ARC-AGI-3](https://three.arcprize.org/) that completes all three preview games in 1,069 actions, the lowest publicly reported count.

For details on approach and findings, see our [blog post](https://blog.alexisfox.dev/arcagi3).

![Architecture](assets/architecture.png)

## Setup

Requires Python (3.12 recommended) and Docker.

```bash
git clone git@github.com:alexisfox7/RGB-Agent.git
cd RGB-Agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cd docker/opencode-sandbox && bash build.sh   # build analyzer sandbox image
```

Create a `.env` file:

```
ARC_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Usage

```bash
rgb-swarm --suite all --max-actions 500
rgb-swarm --game ls20,ft09
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | — | Predefined game suites (e.g. `ls20`, `vc33`, `ft09`, or `all`) |
| `--game` | — | Comma-separated game names or IDs (alternative to `--suite`) |
| `--max-actions` | 500 | Max actions per game |
| `--interval`, `-n` | 10 | Actions per analyzer batch plan |
| `--model`, `-m` | `claude-opus-4-6` | Analyzer model (see below) |
| `--operation-mode` | `online` | `online` / `offline` / `normal` |

### Models

Anthropic models can be passed without a prefix. For other providers, use `provider/model`.

| Model | `--model` value |
|-------|-----------------|
| Claude Opus 4.6 | `claude-opus-4-6` (default) |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| GPT 5.2 | `openai/gpt-5.2` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |

Any model available via OpenRouter can also be used with the `openrouter/` prefix (e.g. `openrouter/google/gemini-3.1-pro-preview`).

Set the matching API key in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`).

Results are saved to `evaluation_results/`.

## Architecture

The analyzer agent ([OpenCode](https://github.com/opencode-ai/opencode)) runs in a sandboxed Docker container, reads the game's prompt log with Read, Grep, and Python, and outputs a JSON action plan. The action queue drains these one per step with zero LLM calls. When the queue empties or the score changes, the analyzer re-fires.

```
rgb_agent/
├── agent/              
│   ├── opencode_agent.py # Runs OpenCode in Docker to produce action plans
│   ├── action_queue.py # Drains one action per step (to support batched action plans + score-change flush)
│   ├── game_state.py   # Formatting
│   └── prompts.py      
├── environment/        
│   ├── arcagi3.py      # ARC-AGI-3 API wrapper (reset, step, scoring)
│   ├── runner.py       # Per-game orchestration loop
│   ├── swarm.py        # Runs multiple games in parallel on a scorecard
│   └── config.py     
├── metrics/            
└── utils/            
```
