#!/bin/zsh
cd "$(dirname "$0")"
pids="$(lsof -ti:3724)"
if [ -n "$pids" ]; then
  echo "$pids" | xargs kill
fi
nohup python3 -u server.py "$@" > output.log 2>&1 < /dev/null &
pid="$!"
disown "$pid" 2>/dev/null || true
echo "Server started on port 3724, pid ${pid}, log: output.log"
