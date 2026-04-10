#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "用法: bash /workspace/docker/run_as_local_user.sh <command> [args...]"
  exit 2
fi

TARGET_UID="${LOCAL_UID:-1000}"
TARGET_GID="${LOCAL_GID:-1000}"

if [ "$(id -u)" -ne 0 ]; then
  exec "$@"
fi

TARGET_USER="$(getent passwd "${TARGET_UID}" | cut -d: -f1 || true)"
if [ -n "${TARGET_USER}" ]; then
  exec runuser -u "${TARGET_USER}" -- "$@"
fi

if command -v setpriv >/dev/null 2>&1; then
  exec setpriv --reuid "${TARGET_UID}" --regid "${TARGET_GID}" --clear-groups "$@"
fi

echo "[错误] 找不到 UID=${TARGET_UID} 对应用户，且系统无 setpriv，无法降权执行。"
exit 1