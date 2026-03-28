#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SCRIPT="$SCRIPT_DIR/simView.py"

if [[ ! -f "$APP_SCRIPT" ]]; then
  echo "Error: simView.py not found at: $APP_SCRIPT" >&2
  exit 1
fi

# ParaVision may set LD_LIBRARY_PATH to Qt/system libraries that conflict with
export LD_LIBRARY_PATH=

# Usage modes:
# 1) ParaVision: called in simulation output folder, extra args are selected files.
#    -> use current working directory.
# 2) Manual: run_simview.sh /path/to/simulation_data
#    -> switch to that folder first.
if [[ $# -ge 1 && -d "$1" ]]; then
  TARGET_DIR="$1"
  shift
  cd "$TARGET_DIR"
else
  TARGET_DIR="$PWD"
fi

# Error log (override with SIMVIEW_ERROR_LOG=/path/to/file).
LOG_FILE="${SIMVIEW_ERROR_LOG:-$TARGET_DIR/simview_viewer_errors.log}"

# Log stderr to file so ParaVision launch/runtime errors are persisted.
exec 2>>"$LOG_FILE"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] run_simview.sh started in $TARGET_DIR" >&2
trap 'rc=$?; echo "[$(timestamp)] ERROR rc=$rc line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

can_import_pyqt6() {
  local py="$1"
  "$py" -c "import PyQt6" >/dev/null 2>&1
}

VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[$(timestamp)] Required venv python not found: $VENV_PY" >&2
  echo "[$(timestamp)] Create it with: poetry -C $SCRIPT_DIR install" >&2
  exit 1
fi

if ! can_import_pyqt6 "$VENV_PY"; then
  echo "[$(timestamp)] .venv python exists but PyQt6 is missing: $VENV_PY" >&2
  echo "[$(timestamp)] Reinstall dependencies with: poetry -C $SCRIPT_DIR install" >&2
  exit 1
fi

build_platform_list() {
  if [[ -n "${SIMVIEW_QT_PLATFORM:-}" ]]; then
    echo "$SIMVIEW_QT_PLATFORM"
    return
  fi

  # Prefer Wayland when available to avoid xcb system-library issues.
  if [[ -n "${WAYLAND_DISPLAY:-}" || "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
    echo "wayland"
    echo "xcb"
    return
  fi

  if [[ -n "${DISPLAY:-}" ]]; then
    echo "xcb"
    echo "wayland"
    return
  fi

  # Headless/unknown session fallback.
  echo "offscreen"
}

LAST_RC=1
while IFS= read -r platform; do
  export QT_QPA_PLATFORM="$platform"
  echo "[$(timestamp)] launching with .venv python: $VENV_PY (QT_QPA_PLATFORM=$platform)" >&2
  set +e
  "$VENV_PY" "$APP_SCRIPT" "$TARGET_DIR"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    exit 0
  fi
  LAST_RC=$rc
  echo "[$(timestamp)] launch failed rc=$rc with QT_QPA_PLATFORM=$platform" >&2
done < <(build_platform_list)

echo "[$(timestamp)] All Qt platform launch attempts failed." >&2
echo "[$(timestamp)] If xcb fails, install system package: libxcb-cursor0 (Debian/Ubuntu)." >&2
echo "[$(timestamp)] You can also force backend, e.g. SIMVIEW_QT_PLATFORM=wayland $SCRIPT_DIR/run_simview.sh" >&2
exit "$LAST_RC"
