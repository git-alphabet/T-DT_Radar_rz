#!/usr/bin/env bash
set -euo pipefail

TARGET_UID="${LOCAL_UID:-1000}"
TARGET_GID="${LOCAL_GID:-1000}"
USBFS_MEMORY_MB="${USBFS_MEMORY_MB:-2000}"

if [ "$(id -u)" -eq 0 ]; then
  if [ -w /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    echo "${USBFS_MEMORY_MB}" > /sys/module/usbcore/parameters/usbfs_memory_mb || true
  fi

  mkdir -p /workspace/images /workspace/MvSdkLog
  chown -R "${TARGET_UID}:${TARGET_GID}" /workspace/images /workspace/MvSdkLog 2>/dev/null || true
fi

exec "$@"