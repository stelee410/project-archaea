#!/usr/bin/env bash
# Build the React WebUI bundle and launch the FastAPI server that serves it.
#
# Usage:
#   ./start-webui.sh                 # default: 127.0.0.1:8000, build webui first
#   ./start-webui.sh --port 9000     # custom port
#   ./start-webui.sh --host 0.0.0.0  # bind all interfaces
#   ./start-webui.sh --no-build      # skip webui build (use existing webui/dist)
#   ./start-webui.sh --skip-install  # skip the npm install check
#
# The script always uses .venv/bin/python (NOT system python) so the FastAPI
# / uvicorn / websockets dependencies are guaranteed to be available.

set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Build the React WebUI bundle and launch the FastAPI server that serves it.

Usage:
  ./start-webui.sh                 # default: 127.0.0.1:8000, build webui first
  ./start-webui.sh --port 9000     # custom port
  ./start-webui.sh --host 0.0.0.0  # bind all interfaces
  ./start-webui.sh --no-build      # skip webui build (use existing webui/dist)
  ./start-webui.sh --skip-install  # skip the npm install check

The script always uses .venv/bin/python (NOT system python) so the FastAPI
/ uvicorn / websockets dependencies are guaranteed to be available.
EOF
}

# ── parse args ───────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8000"
DO_BUILD=1
DO_INSTALL_CHECK=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)         HOST="$2"; shift 2 ;;
    --host=*)       HOST="${1#*=}"; shift ;;
    --port)         PORT="$2"; shift 2 ;;
    --port=*)       PORT="${1#*=}"; shift ;;
    --no-build)     DO_BUILD=0; shift ;;
    --skip-install) DO_INSTALL_CHECK=0; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── pretty logging ───────────────────────────────────────────────────────────
log()  { printf "\033[1;36m[archaea]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[archaea]\033[0m %s\n" "$*" >&2; }
err()  { printf "\033[1;31m[archaea]\033[0m %s\n" "$*" >&2; }

# ── 1. Python venv check ─────────────────────────────────────────────────────
VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  err "找不到 $VENV_PY"
  err "先建好 venv 再来一遍："
  err "    python3 -m venv .venv"
  err "    .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ "$DO_INSTALL_CHECK" -eq 1 ]]; then
  if ! "$VENV_PY" -c "import fastapi, uvicorn, websockets" >/dev/null 2>&1; then
    warn "venv 里缺 fastapi / uvicorn / websockets，正在安装…"
    "$VENV_PY" -m pip install -r requirements.txt
  fi
fi

# ── 2. WebUI build ───────────────────────────────────────────────────────────
if [[ "$DO_BUILD" -eq 1 ]]; then
  if [[ ! -d "webui/node_modules" ]]; then
    log "首次构建：先 npm install"
    (cd webui && npm install)
  fi
  log "构建前端 (cd webui && npm run build)"
  (cd webui && npm run build)
else
  if [[ ! -d "webui/dist" ]]; then
    err "--no-build 但找不到 webui/dist 产物。先去掉 --no-build 跑一次。"
    exit 1
  fi
  log "跳过前端构建 (--no-build)，使用现有 webui/dist"
fi

# ── 3. Launch server ─────────────────────────────────────────────────────────
log "启动 FastAPI server: http://${HOST}:${PORT}"
log "Ctrl-C 退出"
exec "$VENV_PY" -m archaea.server --host "$HOST" --port "$PORT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
