#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-manga-team}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEAM_DIR="$ROOT_DIR/agent_team"
WORKER_CMD="${CODEX_WORKER_CMD:-}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n board -c "$ROOT_DIR"
tmux send-keys -t "$SESSION_NAME:board" "clear" C-m
tmux send-keys -t "$SESSION_NAME:board" "printf 'Manga-Set multi-role workspace\\n\\n'; printf 'Read: %s\\n' '$TEAM_DIR/README.md' '$TEAM_DIR/TASK_BOARD.md'" C-m

create_role_window() {
  local window_name="$1"
  local role_file="$2"
  tmux new-window -t "$SESSION_NAME" -n "$window_name" -c "$ROOT_DIR"
  tmux send-keys -t "$SESSION_NAME:$window_name" "clear" C-m
  tmux send-keys -t "$SESSION_NAME:$window_name" "printf 'Role: $window_name\\n'; printf 'Brief: %s\\n' '$role_file'; printf 'Board: %s\\n' '$TEAM_DIR/TASK_BOARD.md'" C-m

  if [[ -n "$WORKER_CMD" ]]; then
    tmux send-keys -t "$SESSION_NAME:$window_name" "printf 'Worker command configured but not auto-launched.\\nCommand: %s\\n' \"$WORKER_CMD\"" C-m
  fi
}

create_role_window orchestrator "$TEAM_DIR/roles/orchestrator.md"
create_role_window developer "$TEAM_DIR/roles/developer.md"
create_role_window reviewer "$TEAM_DIR/roles/reviewer.md"
create_role_window writer "$TEAM_DIR/roles/writer.md"

tmux select-window -t "$SESSION_NAME:board"
if [[ -x "$ROOT_DIR/scripts/open_agent_workspace.sh" ]]; then
  bash "$ROOT_DIR/scripts/open_agent_workspace.sh" >/dev/null 2>&1 || true
fi
echo "Created tmux session '$SESSION_NAME'"
echo "Attach with: tmux attach -t $SESSION_NAME"
