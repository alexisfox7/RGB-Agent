#!/usr/bin/env bash
# Wrapper script that runs Claude Code inside a Docker container.
# Used as cli_path for the Claude Agent SDK.
#
# Auth: uses CLAUDE_CODE_OAUTH_TOKEN env var. On macOS, if not set,
# automatically extracts from Keychain (Claude Max/Pro subscription).
# Falls back to ANTHROPIC_API_KEY for API-key auth.

set -euo pipefail

IMAGE="rgb-agent/claude-sandbox:latest"

MOUNT_DIR="$(cd "${CLAUDE_SANDBOX_CWD:-$(pwd)}" && pwd)"

# Resolve auth token
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    # Try extracting from macOS Keychain
    if command -v security &>/dev/null; then
        CLAUDE_CODE_OAUTH_TOKEN="$(security find-generic-password -s 'Claude Code-credentials' -w 2>/dev/null || true)"
    fi
fi

AUTH_FLAGS=()
if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    AUTH_FLAGS+=(-e "CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}")
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    AUTH_FLAGS+=(-e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
else
    echo "ERROR: No auth available. Set CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY, or log in to Claude Code." >&2
    exit 1
fi

exec docker run --rm -i \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --tmpfs /tmp:rw,nosuid,size=256m \
    --tmpfs /home/sandbox/.npm:rw,noexec,nosuid,size=64m \
    --tmpfs /home/sandbox/.claude:rw,noexec,nosuid,size=64m \
    -v "${MOUNT_DIR}:/workspace:ro" \
    "${AUTH_FLAGS[@]}" \
    -e "HOME=/home/sandbox" \
    -e "DISABLE_AUTOUPDATER=1" \
    -e "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" \
    -w /workspace \
    "${IMAGE}" "$@"
