# Docker workflow (Ubuntu 24.04 + ROS2 Jazzy)

本目录改为 Compose 为主，适配 VS Code 右键 Compose Up。

## 1) 准备环境变量

在项目根目录执行：

```bash
cp docker/.env.example .env
sed -i "s/^LOCAL_UID=.*/LOCAL_UID=$(id -u)/" .env
sed -i "s/^LOCAL_GID=.*/LOCAL_GID=$(id -g)/" .env
```

## 2) 启动开发容器（root 默认）

```bash
docker compose --env-file .env -f docker/compose.dev.yml up -d --build
docker compose --env-file .env -f docker/compose.dev.yml exec radar bash
```

说明：

- `docker/Dockerfile.base` 已内置 CUDA + cuDNN + TensorRT 安装，不需要再手工进容器配置。
- Compose 构建上下文已限制在 `docker/` 目录，避免把整仓库（模型、数据）发送给 builder。
- 默认启用 TensorRT 10 对应版本锁定：`TensorRT 10.7.0.23-1+cuda12.6`，并配套 `cuda-nvcc-12-6` 与 `cuda-cudart-dev-12-6`。
- 默认会对 `libnvinfer*` 与 `libnvonnxparsers*` 执行 `apt-mark hold`，避免构建后被意外升级到不匹配版本。
- 默认 root 运行，减少设备访问和排障时的权限阻碍。
- 只有在你明确要避免宿主机 root 产物时，再使用 `radar_user`。

低空间实测（apt 模拟安装）：

- 旧方案（`cuda-toolkit-12-5 + tensorrt-dev`）：下载约 10.8 GB，落盘约 20.6 GB
- 新方案（当前默认，TensorRT10.7+CUDA12.6）：下载约 3.6 GB，落盘约 10.7 GB

容器内编译：

```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/T-DT_Radar
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

容器内可快速验证 NVIDIA 环境：

```bash
nvidia-smi
nvcc --version
python3 - <<'PY'
import ctypes
for lib in ["libnvinfer.so", "libnvonnxparser.so"]:
	ctypes.CDLL(lib)
	print(f"loaded: {lib}")
PY
```

## 3) 仅在“写宿主机文件”时切本地 UID/GID

```bash
docker compose --env-file .env -f docker/compose.dev.yml --profile user run --rm radar_user bash
```

这一步不是必须。你可以全程 root 开发；只有在遇到宿主机权限不便时再用本地 UID/GID。

## 4) 容器快照成镜像（compose 方式）

先确认目标容器名：

```bash
docker ps --format '{{.Names}}'
```

执行快照：

```bash
TARGET_CONTAINER=tdt-radar-dev docker compose --env-file .env -f docker/compose.snapshot.yml up --abort-on-container-exit --exit-code-from snapshot
```

若设置了 DOCKER_HUB_USER 和 IMAGE_NAME，会自动 push 时间戳 tag 与 latest。

## 5) 停止开发容器

```bash
docker compose --env-file .env -f docker/compose.dev.yml down
```

## 说明

- 代码目录直接挂载，修改即时生效。
- 默认 root 运行，符合排障和设备访问习惯。
- 需要避免宿主机 root 产物时，可切到 radar_user 服务运行。
- 如果你要带更多调试工具（gdb/vim/tmux 等），在 `.env` 设置 `INSTALL_EXTRA_TOOLS=1`。
- 当前默认版本锁定为：`TENSORRT_VERSION=10.7.0.23-1+cuda12.6`、`CUDA_NVCC_PKG=cuda-nvcc-12-6`、`CUDA_CUDART_DEV_PKG=cuda-cudart-dev-12-6`。
- `CUDNN_DEV_VERSION` 默认留空，交由 apt 解析与当前 CUDA12 兼容的 cuDNN 版本；如需固定，填入具体版本并会同时锁定 `libcudnn9-cuda-12`。
- 如需关闭版本锁定，在 `.env` 设置 `PIN_TENSORRT=0`。
- NVIDIA 栈“一次成功”依赖宿主机驱动兼容；若容器内 `nvidia-smi` 正常但 CUDA 程序报 driver/version mismatch，需要升级宿主机驱动或下调镜像中的 CUDA 版本。
