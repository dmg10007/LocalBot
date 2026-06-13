#!/bin/sh
# Fix ownership of bind-mounted directories that Docker overwrites with
# root-owned host directories, preventing the localbot user from writing.
chown -R localbot:localbot /app/storage /app/logs /app/sandbox 2>/dev/null || true
exec gosu localbot "$@"
