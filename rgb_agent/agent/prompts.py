"""Prompt templates for the analyzer agent."""

INITIAL_PROMPT = """\
You are a strategic advisor for an AI agent playing a grid-based puzzle game.
The agent's full prompt log for this run is at this ABSOLUTE path: {log_path}

You may only access this single file (use its absolute path directly with Read and Grep).

Most games have some form of timer mechanism. A score increase means a level was solved.

Deeply analyze this log to understand what the agent has been doing, what has worked,
what hasn't, and what patterns explain the game's behavior.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

RESUME_PROMPT = """\
The prompt log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest actions (from where you left off) and update your strategic briefing.
Focus on what changed: new moves, score transitions, and whether the agent followed
your previous plan or diverged. Parse the board programmatically from the file using
section markers ([POST-ACTION BOARD STATE], etc.) — do NOT visually copy the grid.

Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "ACTION1"}}, {{"action": "ACTION6", "x": 3, "y": 7}}, ...], "reasoning": "why these steps"}}

Available actions: ACTION1-4 (moves), ACTION6 (click at x,y), ACTION5 (no-op), RESET.
Each action MUST be a JSON object: {{"action": "ACTION6", "x": <row>, "y": <col>}} for clicks, {{"action": "ACTION1"}} for moves. Never use string shorthand like "ACTION6(x,y)".
Plan 1–{plan_size} actions. IMPORTANT: shorter plans (3-5 steps) are strongly preferred
because the agent can re-evaluate sooner. Only use more than 5 if you have very high
confidence AND the extra steps are critical. Even on a clear straight path, prefer
stopping early so the agent can observe the game's response and adapt.
\
"""

PYTHON_ADDENDUM = (
    "\n\nBash (and therefore Python) is available to you. **Always** use Python to "
    "parse the board — do NOT try to visually read the ASCII grid.\n\n"
    "The log file uses section markers to delimit board grids:\n"
    "  [INITIAL BOARD STATE]   — the grid at the start (after Action 0 header)\n"
    "  [POST-ACTION BOARD STATE] — the grid after each action\n"
    "\n"
    "To extract the latest board into a matrix:\n"
    "```python\n"
    "import re\n"
    "data = open('{log_path}').read()\n"
    "# Find the last board state section\n"
    "boards = re.split(r'\\[(?:POST-ACTION|INITIAL) BOARD STATE\\]', data)\n"
    "last_board = boards[-1].strip()\n"
    "# Skip 'Score: N' line if present, then parse rows into a 2-D list\n"
    "lines = last_board.split('\\n')\n"
    "if lines[0].startswith('Score:'):\n"
    "    lines = lines[1:]\n"
    "grid = [list(row) for row in lines if row.strip()]\n"
    "# Now slice, count, compare programmatically\n"
    "```\n"
    "Run Python inline."
)
