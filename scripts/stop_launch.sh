#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
HOST_WORKSPACE_PATH="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
WORKSPACE_NAME="$(basename "${HOST_WORKSPACE_PATH}")"

CONTAINER_NAME="tdt-radar-dev"
CONTAINER_WORKSPACE_PATH="/workspace/${WORKSPACE_NAME}"
HOST_ONLY=0
CONTAINER_ONLY=0

usage() {
    cat <<'EOF'
Usage: scripts/stop_related_launch.sh [options]

Stop only tdt_vision related ros2 launch processes.

Options:
  --container <name>  Container name (default: tdt-radar-dev)
    --container-workspace <path>
                                         Workspace path inside container
                                         (default: /workspace/<repo-name>)
  --host-only         Only stop host-side launch processes
  --container-only    Only stop container-side launch processes
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for --container"
                exit 2
            fi
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --host-only)
            HOST_ONLY=1
            shift
            ;;
        --container-workspace)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for --container-workspace"
                exit 2
            fi
            CONTAINER_WORKSPACE_PATH="$2"
            shift 2
            ;;
        --container-only)
            CONTAINER_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ "$HOST_ONLY" -eq 1 && "$CONTAINER_ONLY" -eq 1 ]]; then
    echo "[ERROR] --host-only and --container-only cannot be used together"
    exit 2
fi

LAUNCH_PATTERN='ros2 launch tdt_vision (radar\.launch\.py|run_rosbag\.launch\.py|calibrate_radar\.launch\.py|calib_rosbag\.launch\.py|map_server_launch\.py)'

filter_host_pids_by_workspace() {
    local candidates="$1"
    local out=()
    local pid=""
    local cwd=""

    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
        [[ -n "${cwd}" ]] || continue
        if [[ "${cwd}" == "${HOST_WORKSPACE_PATH}" || "${cwd}" == "${HOST_WORKSPACE_PATH}"/* ]]; then
            out+=("${pid}")
        fi
    done <<< "${candidates}"

    printf '%s\n' "${out[@]}" 2>/dev/null || true
}

filter_container_pids_by_workspace() {
    local candidates="$1"
    local out=()
    local pid=""
    local cwd=""

    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        cwd="$(docker exec "${CONTAINER_NAME}" readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
        [[ -n "${cwd}" ]] || continue
        if [[ "${cwd}" == "${CONTAINER_WORKSPACE_PATH}" || "${cwd}" == "${CONTAINER_WORKSPACE_PATH}"/* ]]; then
            out+=("${pid}")
        fi
    done <<< "${candidates}"

    printf '%s\n' "${out[@]}" 2>/dev/null || true
}

get_matching_pids() {
    local scope="$1"
    local raw=""
    local filtered=""

    if [[ "${scope}" == host ]]; then
        raw="$(pgrep -f "${LAUNCH_PATTERN}" || true)"
        [[ -n "${raw}" ]] || return 0
        filtered="$(filter_host_pids_by_workspace "${raw}" || true)"
    else
        raw="$(docker exec "${CONTAINER_NAME}" pgrep -f "${LAUNCH_PATTERN}" || true)"
        [[ -n "${raw}" ]] || return 0
        filtered="$(filter_container_pids_by_workspace "${raw}" || true)"
    fi

    [[ -n "${filtered}" ]] && printf '%s\n' "${filtered}" || true
}

stop_scope() {
    local scope="$1"
    local pids=""
    local left=""

    pids="$(get_matching_pids "${scope}" || true)"

    if [[ -z "${pids}" ]]; then
        echo "[INFO] ${scope}: no matching launch process in workspace"
        return 0
    fi

    echo "[INFO] ${scope}: stopping launch pid(s): ${pids//$'\n'/ }"
    if [[ "${scope}" == host ]]; then
        kill -INT ${pids} || true
    else
        docker exec "${CONTAINER_NAME}" kill -INT ${pids} || true
    fi

    for _ in 1 2 3 4 5; do
        sleep 0.4
        left="$(get_matching_pids "${scope}" || true)"
        if [[ -z "${left}" ]]; then
            echo "[OK] ${scope}: launch stopped"
            return 0
        fi
    done

    if [[ -n "${left}" ]]; then
        echo "[WARN] ${scope}: still running, sending TERM to pid(s): ${left//$'\n'/ }"
        if [[ "${scope}" == host ]]; then
            kill -TERM ${left} || true
        else
            docker exec "${CONTAINER_NAME}" kill -TERM ${left} || true
        fi
        sleep 0.5
    fi

    left="$(get_matching_pids "${scope}" || true)"
    if [[ -n "${left}" ]]; then
        echo "[WARN] ${scope}: remaining launch pid(s): ${left//$'\n'/ }"
        return 1
    fi

    echo "[OK] ${scope}: launch stopped"
    return 0
}

ret=0

if [[ "$CONTAINER_ONLY" -eq 0 ]]; then
    echo "[INFO] host workspace filter: ${HOST_WORKSPACE_PATH}"
    stop_scope "host" || ret=1
fi

if [[ "$HOST_ONLY" -eq 0 ]]; then
    if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
        echo "[INFO] container workspace filter: ${CONTAINER_WORKSPACE_PATH}"
        stop_scope "container:${CONTAINER_NAME}" || ret=1
    else
        echo "[INFO] container:${CONTAINER_NAME} is not running, skip"
    fi
fi

exit "$ret"
