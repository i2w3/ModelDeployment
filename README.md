# WorkList
- [x] ONNX 模型转 TensorRT Engine (FP32、FP16) (scripts/build_engine.py)
- [x] YOLO11 五种模型(cls/det/obb/pose/seg) TensorRT 推理 (libs/YOLO-TRT)
- [x] OPENCV RTSP 取流和推流 (src/test_rtsp.cpp 依赖 GSTREAMER，src/test_rtsp2.cpp 依赖 FFMPEG)
- [x] YOLO11 五种模型预测 + 推流 (src/test_yolo_trt_stream.cpp)
- [x] 跨平台编译测试 (目前仅在 Windows 上测试通过 -> linux 上编译通过)
- [x] 优化推流代码 (gst 推流帧率不稳 + 目前刚开始启动推流后，正常播放几秒后必卡几秒，随后正常 -> 修改 gst 命令后，卡断时间减少)
- [ ] 优化 src/test_yolo_trt_stream.cpp （帧率不稳，考虑添加生产者消费者模型）

# Config RTSP Server
## Windows
下载 [mediamtx](https://mediamtx.org/) 保持后台运行，配合 [FFMPEG](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z) 即可：
```powershell
ffmpeg -re -stream_loop -1 -hwaccel cuda -i ./face2.mp4 -c:v h264_nvenc -c:a copy -g 30 -rtsp_transport tcp -f rtsp rtsp://localhost:8554/live
```

## Linux
- 拉取 mediamtx 容器并运行：
```bash
docker network create stream-net
# 最小化实现 RTSP (TCP) 收/发
docker run -d \
    --name mediamtx \
    --network stream-net \
    -e MTX_RTSPTRANSPORTS=tcp \
    -p 8554:8554 \
    bluenviron/mediamtx:1

# 官方设置
[RTSP] listener opened on :8554 (TCP), :8000 (UDP/RTP), :8001 (UDP/RTCP)
[RTMP] listener opened on :1935
[HLS] listener opened on :8888
[WebRTC] listener opened on :8889 (HTTP), :8189 (ICE/UDP)
[SRT] listener opened on :8890 (UDP)

# 本地循环推流（注意不是在 mediamtx 容器中运行，可在 TensorRT 容器中运行）
## VLC 取流播放刚开始会有卡顿，后面能稳定到 speed=0.99x
## 注意对比与 Windows 中的 RTSP Server address，dockers 中需要使用 rtsp://mediamtx:8554/live
${VCPKG_ROOT}/installed/x64-linux-dynamic/tools/ffmpeg/ffmpeg -re -stream_loop -1 -hwaccel cuda -i ./face2.mp4 -c:v h264_nvenc -c:a copy -g 30 -rtsp_transport tcp -f rtsp rtsp://mediamtx:8554/live
```

# Build config
## Windows
Windows 必须通过 `runVSCode.ps1` 进入项目才能激活 MSVC 的环境，默认使用的是 x64 - x64 的设置，需要修改查看[微软 MSVC 的说明](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=visualstudio)来修改 `runVSCode.ps1`。

Windows 还需要在环境变量中添加 `VCPKG_ROOT` 和 vcpkg 可执行文件路径到 `PATH`，此外 cudnn、tensorRT 都是 zip package 形式，配置环境变量较为复杂，建议先安装 cuda，然后把 cudnn.zip 的内容复制到 cuda 的相同路径下，注意 cudnn 会有一级额外的 cuda 版本。

例如：`${CUDNN_PATH}/bin/12.8/${.dll}` 复制 `${.dll}` 到 `{CUDA_PATH}/bin` 下即可，`{CUDNN_PATH}/include&lib`也需要复制到`{CUDA_PATH}`的相应路径。

此外还需要[安装 VCPKG](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)，安装和导出必要的二进制文件：
```powershell
# 使用 VCPKG 编译二进制库
vcpkg install spdlog
vcpkg install ffmpeg[avcodec,avdevice,avfilter,avformat,core,gpl,ffmpeg,nvcodec,swresample,swscale,x264]
vcpkg install gstreamer[plugins-base,plugins-good,plugins-bad,plugins-ugly,libav,nvcodec,x264]
vcpkg install gst-rtsp-server
vcpkg install opencv4[ade,contrib,cuda,cudnn,dnn,dnn-cuda,ffmpeg,freetype,gstreamer,highgui,ipp,jpeg,nonfree,openjpeg,png,quirc,thread,tiff,webp]
## 注意：windows 端无法编译 gstreamer[nvcodec]
## vcpkg onnxruntime port 未实现 linux 端编译，仅供参考
vcpkg install onnxruntime[cuda,tensorrt]
```

开始编译：
```powershell
# 具体 debug 和 release 含义看 CMakePresets.json
cmake --preset debug
cmake --build build/debug

cmake --preset release
cmake --build build/release
```

## Linux
Linux 拉取 `TensorRT` 镜像，并运行下面命令完成 `VCPKG` 安装：
```bash
docker run -it \
    --gpus all \
    --name trt \
    --network stream-net \
    nvcr.io/nvidia/tensorrt:24.12-py3
# 进入容器（第一次运行上面的命令应该是已经进入的了，可以跳过）
docker start trt
docker exec -it trt bash
# 基础依赖
apt-get update && apt-get -y install curl zip unzip tar pkg-config autoconf autoconf-archive automake libtool python3-venv bison libx11-dev libxft-dev libxext-dev nasm flex libxrandr-dev libxi-dev libxtst-dev
apt-get autoremove
# 设置代理（可选）
export HTTP_PROXY="http://x.x.x.x:7890"
export HTTPS_PROXY="http://x.x.x.x:7890"
# 安装 VCPKG 并配置 TensorRT 路径
cd $HOME
git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
echo 'export VCPKG_ROOT="$HOME/vcpkg"' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
echo 'export VCPKG_DEFAULT_TRIPLET=x64-linux-dynamic' >> ~/.bashrc
echo 'export TensorRT_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu' >> ~/.bashrc
echo 'export TensorRT_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
# 使用 VCPKG 安装依赖
vcpkg install spdlog
vcpkg install ffmpeg[avcodec,avdevice,avfilter,avformat,core,gpl,ffmpeg,nvcodec,swresample,swscale,x264]
vcpkg install gstreamer[plugins-base,plugins-good,plugins-bad,plugins-ugly,libav,nvcodec,x264]
vcpkg install gst-rtsp-server
vcpkg install opencv4[ade,contrib,cuda,cudnn,dnn,dnn-cuda,ffmpeg,freetype,gstreamer,highgui,ipp,jpeg,nonfree,openjpeg,png,quirc,thread,tiff,webp]
## vcpkg onnxruntime port 未实现 linux 端编译，仅供参考
vcpkg install onnxruntime[cuda,tensorrt]
```

开始编译：
```bash
# 建议使用 vcpkg 中的 cmake
${VCPKG_ROOT}/downloads/tools/cmake-*/cmake-*/bin/cmake --preset linux_debug
${VCPKG_ROOT}/downloads/tools/cmake-*/cmake-*/bin/cmake --build build/debug
```

测试运行：
```bash
# GSTREAMER CPU 编解码
build/debug/bin/test_rtsp \
"rtspsrc location=rtsp://mediamtx:8554/live protocols=tcp latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=false sync=false" \
"videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 threads=4 key-int-max=30 ! rtspclientsink location=rtsp://mediamtx:8554/output protocols=tcp"

# GSTREAMER GPU 编解码
build/debug/bin/test_rtsp \
"rtspsrc location=rtsp://mediamtx:8554/live protocols=tcp latency=100 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=false sync=false" \
"videoconvert ! video/x-raw,format=NV12 ! nvh264enc bitrate=2000 rc-mode=cbr zerolatency=true ! video/x-h264,profile=high,stream-format=byte-stream ! rtspclientsink location=rtsp://mediamtx:8554/output protocols=tcp"

# GSTREAMER GPU 编解码 + YOLO Model
build/debug/bin/test_rtsp \
"rtspsrc location=rtsp://mediamtx:8554/live protocols=tcp latency=100 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=false sync=false" \
"videoconvert ! video/x-raw,format=NV12 ! nvh264enc bitrate=2000 rc-mode=cbr zerolatency=true ! video/x-h264,profile=high,stream-format=byte-stream ! rtspclientsink location=rtsp://mediamtx:8554/output protocols=tcp" \
<trtModelType> \
<trtModelPath>

# FFMPEG GPU 编解码
build/debug/bin/test_rtsp2 rtsp://mediamtx:8554/live rtsp://mediamtx:8554/output
```

### GStreamer Pipeline 元素说明
#### 输入 Pipeline (取流)
| 元素 | 参数 | 说明 |
| - | - | - |
| `rtspsrc`                | `location=rtsp://...` | RTSP 源地址 |
|                          | `protocols=tcp`       | 使用 TCP 传输 |
|                          | `latency=100`         | 延迟缓冲时间 (ms) |
| `rtph264depay`           | -                     | 从 RTP 载荷中提取 H.264 数据 |
| `h264parse`              | -                     | 解析 H.264 码流，获取关键帧信息 |
| **解码器**                | **GPU**: `nvh264dec`  | NVIDIA GPU H.264 解码器 |
|                          | **CPU**: `avdec_h264` | CPU H.264 解码器 |
| `videoconvert`           | -                     | 颜色空间转换 |
| `video/x-raw,format=BGR` | -                     | Caps 过滤器，指定输出格式为 BGR |
| `appsink`                | `max-buffers=2`       | 最多缓冲 2 帧 |
|                          | `drop=false`          | 不丢弃帧 |
|                          | `sync=false`          | 不同步时钟 |

#### 输出 Pipeline (推流)
**./src/test_rtsp.cpp** 中自动添加 `appsrc` 和 caps 部分，命令行只需提供从 `videoconvert` 开始的部分

**完整 Pipeline 结构** (代码自动构建)：
```cpp
appsrc is-live=true format=time do-timestamp=true !
video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 !
[用户提供的 pipeline]
```
GSTREAMER GPU 解码：
| 元素 | 参数 | 说明 |
| - | - | - |
| `appsrc`                                                   | `is-live=true`       | 实时模式  |
|                                                            | `format=time`        | 使用时间格式  |
|                                                            | `do-timestamp=true`  | 自动添加时间戳  |
| `video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1` | -                    | Caps 过滤器，格式/分辨率/帧率  |
| `videoconvert`                                             | -                    | 颜色空间转换 (BGR -> NV12) |
| `video/x-raw,format=NV12`                                  | -                    | NVENC 需要 NV12 格式 |
| **编码器**                                                  | **GPU**: `nvh264dec` | NVIDIA GPU H.264 编码器  |
|                                                            | `bitrate=2000`       | 比特率 2000 kbps |
|                                                            | `rc-mode=cbr`        | 固定码率模式 |
|                                                            | `zerolatency=true`   | 零延迟模式 |
| `video/x-h264,profile=high,stream-format=byte-stream`      | -                    | 指定 H.264 输出格式 |
| `rtspclientsink`                                           | `location=rtsp://...`| RTSP 推流目标地址 |
|                                                            | `protocols=tcp`      | 使用 TCP 传输 |

### 项目导出
将 vcpkg 编译好的二进制文件复制到新容器（跨设备验证未实现，目前开发容器和部署容器均在同一台设备上），减少编译时间和加快部署：
```bash
# 开发容器
## 导出 release 版本
cmake --preset linux_release
cmake --build build/release
## 导出 VCPKG 编译好的二进制文件

## 宿主机内运行
docker cp trt:/root/ModelDeployment/build/release/bin ./MD_bin
docker stop trt

# 部署容器（先宿主机上运行，拉取一样的容器并加入 RTSP Server 的网络）
docker run -it \
    --gpus all \
    --name trt_deploy \
    --network stream-net \
    nvcr.io/nvidia/tensorrt:24.12-py3
## 进入容器（第一次运行上面的命令应该是已经进入的了，可以跳过）
docker start trt_deploy
docker exec -it trt_deploy bash
```

# NOTE
重新开发下载 VSCode 的 CMake Tools 插件，切换到侧边栏的 CMake 点击，选择预设并配置按钮，可以搜索到相关的头文件(逻辑见.vscode/c_cpp_properties.json)

# TODO

# 测试 onnx 和 tensorRT engine 的输出差异
使用工具[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)，注意要先到 tensorRT 的安装路径找到 python wheel 安装先
```
pip install "C:\Program Files\NVIDIA\TensorRT\v10.13.3.9.Windows.win10.cuda-12.9\python\tensorrt-10.13.3.9-cp312-none-win_amd64.whl"
pip install colored polygraphy polygraphy-trtexec --extra-index-url https://pypi.ngc.nvidia.com
```

```
polygraphy run model.onnx --trt --onnxrt
polygraphy run model.onnx --trt --onnxrt --fp16 --atol 1e-3 --rtol 1e-3
```

## note
### 集成配置
```CMakeLists.txt
# spdlog 随项目文件一起提供，自动区分 debug 和 release 版本
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Build spdlog example" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "Build spdlog header-only example" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Build spdlog tests" FORCE)
set(SPDLOG_BUILD_TESTS_HO OFF CACHE BOOL "Build spdlog header-only tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Build spdlog benchmarks" FORCE)
set(SPDLOG_BUILD_WARNINGS OFF CACHE BOOL "Enable spdlog's own warnings flags" FORCE)
add_subdirectory(libs/spdlog-1.16.0) # 注意不要再添加 find_package 了
```