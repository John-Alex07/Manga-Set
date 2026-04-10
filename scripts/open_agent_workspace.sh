#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-manga-team}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEAM_DIR="$ROOT_DIR/agent_team"

show_role() {
  local role="$1"
  local pane_target="$SESSION_NAME:$role"
  tmux send-keys -t "$pane_target" "clear" C-m
  tmux send-keys -t "$pane_target" "printf 'Role: $role\\n\\n'" C-m
  tmux send-keys -t "$pane_target" "sed -n '1,220p' '$TEAM_DIR/roles/$role.md'" C-m
  tmux send-keys -t "$pane_target" "printf '\\nAssignment\\n\\n'" C-m
  tmux send-keys -t "$pane_target" "sed -n '1,220p' '$TEAM_DIR/assignments/$role.md'" C-m
}

tmux send-keys -t "$SESSION_NAME:board" "clear" C-m
tmux send-keys -t "$SESSION_NAME:board" "sed -n '1,220p' '$TEAM_DIR/STATUS.md'" C-m
tmux send-keys -t "$SESSION_NAME:board" "printf '\\nTask Board\\n\\n'" C-m
tmux send-keys -t "$SESSION_NAME:board" "sed -n '1,260p' '$TEAM_DIR/TASK_BOARD.md'" C-m

show_role orchestrator
show_role developer
show_role reviewer
show_role writer

echo "Loaded role briefs into tmux session '$SESSION_NAME'"
