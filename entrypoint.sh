#!/bin/bash
set -e

# Fix ownership on mounted volumes so the whisper user can write to them.
# This runs as root before dropping privileges.
chown -R whisper:whisper /app/.cache/huggingface 2>/dev/null || true
chown -R whisper:whisper /app/output 2>/dev/null || true

exec gosu whisper "$@"
