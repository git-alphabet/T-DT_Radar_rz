#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

if [[ -z "${WS_DIR}" || "${WS_DIR}" == "/" || "${WS_DIR}" == "/home" || "${WS_DIR}" == "${HOME:-}" ]]; then
    echo "[ERROR] Dangerous operation blocked: invalid workspace path '${WS_DIR}'."
    exit 1
fi

if [ ! -f "${WS_DIR}/.git/HEAD" ] || [ ! -d "${WS_DIR}/src" ] || [ ! -f "${WS_DIR}/README.md" ]; then
    echo "[ERROR] Dangerous operation blocked: '${WS_DIR}' does not look like this project workspace."
    exit 1
fi

if [ ! -x "$(command -v find)" ] || [ ! -x "$(command -v chown)" ]; then
    echo "[ERROR] Required commands are missing: find/chown."
    exit 1
fi

TARGET_UID=""
TARGET_GID=""
TARGET_USER=""

if [ "$(id -u)" -eq 0 ]; then
    if [ -n "${SUDO_USER:-}" ]; then
        TARGET_USER="${SUDO_USER}"
        TARGET_UID="$(id -u "${TARGET_USER}")"
        TARGET_GID="$(id -g "${TARGET_USER}")"
    else
        WS_UID="$(stat -c %u "${WS_DIR}")"
        WS_GID="$(stat -c %g "${WS_DIR}")"
        if [ "${WS_UID}" -eq 0 ]; then
            echo "[ERROR] Running as root without SUDO_USER and workspace owner is root."
            echo "[ERROR] Refusing ambiguous chown target. Set TARGET_USER explicitly or use sudo from normal user."
            exit 1
        fi
        TARGET_UID="${WS_UID}"
        TARGET_GID="${WS_GID}"
        TARGET_USER="$(getent passwd "${TARGET_UID}" | cut -d: -f1 || true)"
    fi
else
    TARGET_USER="$(id -un)"
    TARGET_UID="$(id -u)"
    TARGET_GID="$(id -g)"
fi

if [ -z "${TARGET_UID}" ] || [ -z "${TARGET_GID}" ]; then
    echo "[ERROR] Failed to resolve target uid/gid."
    exit 1
fi

echo "========================================="
echo "Safely unlocking workspace: ${WS_DIR}"
echo "Target owner: ${TARGET_USER:-uid=${TARGET_UID}} (${TARGET_UID}:${TARGET_GID})"
echo "Automatically finding files locked by Docker root and returning them to target owner..."
echo "========================================="

if [ "$(id -u)" -eq 0 ]; then
    RUN_AS_ROOT=""
elif command -v sudo >/dev/null 2>&1; then
    RUN_AS_ROOT="sudo"
    if ! sudo -n true >/dev/null 2>&1; then
        if [ -t 0 ] && [ -t 1 ]; then
            echo "[INFO] sudo authentication required. Please enter your password to continue chown..."
            if ! sudo -v; then
                echo "[WARN] sudo authentication failed. Skip chown."
                echo "[SUCCESS] Unlock complete (directory layout prepared; ownership unchanged)."
                exit 0
            fi
        else
            echo "[WARN] sudo requires password but current shell is non-interactive."
            echo "[WARN] Skip chown. If needed, run manually with privileges:"
            echo "       sudo find ${WS_DIR} -xdev -path ${WS_DIR}/.git -prune -o -uid 0 ! -xtype l -exec chown ${TARGET_UID}:${TARGET_GID} {} +"
            echo "       sudo find ${WS_DIR} -xdev -path ${WS_DIR}/.git -prune -o -uid 0 -xtype l -exec chown -h ${TARGET_UID}:${TARGET_GID} {} +"
            echo "[SUCCESS] Unlock complete (directory layout prepared; ownership unchanged)."
            exit 0
        fi
    fi
else
    echo "[WARN] sudo is unavailable and current user is not root. Skip chown."
    echo "[SUCCESS] Unlock complete (directory layout prepared; ownership unchanged)."
    exit 0
fi

CHOWN_PREFIX=()
if [ -n "$RUN_AS_ROOT" ]; then
    CHOWN_PREFIX=("$RUN_AS_ROOT")
fi

# Regular files/dirs. Exclude .git to avoid rewriting git-internal ownership.
"${CHOWN_PREFIX[@]}" find "${WS_DIR}" -xdev -path "${WS_DIR}/.git" -prune -o -uid 0 ! -xtype l -exec chown "${TARGET_UID}:${TARGET_GID}" {} +
# Symlinks (no-dereference). Exclude .git for the same reason.
"${CHOWN_PREFIX[@]}" find "${WS_DIR}" -xdev -path "${WS_DIR}/.git" -prune -o -uid 0 -xtype l -exec chown -h "${TARGET_UID}:${TARGET_GID}" {} +

echo "[SUCCESS] Unlock complete. All files in the workspace have been returned to you!"
