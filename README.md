# WorkList
- [x] ONNX 模型转 TensorRT Engine (FP32、FP16) (scripts/build_engine.py)
- [x] YOLO11 五种模型(cls/det/obb/pose/seg) TensorRT 推理 (libs/YOLO-TRT)
- [x] OPENCV RTSP 取流和推流 (src/test_rtsp.cpp 依赖 GSTREAMER，src/test_rtsp2.cpp 依赖 FFMPEG)
- [x] YOLO11 五种模型预测 + 推流 (src/test_yolo_trt_stream.cpp)
- [ ] 跨平台编译测试 (目前仅在 Windows 上测试通过)
- [ ] 优化推流代码 (目前刚开始启动推流后，正常播放几秒后必卡几秒，随后正常)


# config
Windows 必须通过 runVSCode.ps1 进入项目才能激活 MSVC 的环境，默认使用的是 x64 - x64 的设置，需要修改查看[微软 MSVC 的说明](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=visualstudio)来修改 runVSCode.ps1。

此外还需要[安装 vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started)，安装和导出必要的二进制文件：

Linux 拉取 tensorRT 镜像，并运行下面命令完成 VCPKG 安装：
```bash
docker pull nvcr.io/nvidia/tensorrt:24.12-py3
docker network create stream-net
docker run -d \
    --name mediamtx \
    --network stream-net \
    -p 8554:8554 \
    -p 1935:1935 \
    -p 8888:8888 \
    bluenviron/mediamtx
# 基础依赖
apt-get update && apt-get -y install curl zip unzip tar pkg-config autoconf autoconf-archive automake libtool python3-venv bison libx11-dev libxft-dev libxext-dev nasm flex libxrandr-dev libxi-dev libxtst-dev
# 设置代理（可选）
export HTTP_PROXY="http://catman:JajU2dPJzifAy2U6EVgEkC@10.23.51.149:7897"
export HTTPS_PROXY="http://catman:JajU2dPJzifAy2U6EVgEkC@10.23.51.149:7897"
# 安装 VCPKG
cd $HOME
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
echo 'export VCPKG_ROOT="$HOME/vcpkg"' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
echo 'export VCPKG_DEFAULT_TRIPLET=x64-linux-dynamic' >> ~/.bashrc
source ~/.bashrc
# 使用 VCPKG 编译二进制库
vcpkg install ncnn spdlog --recurse
vcpkg install ffmpeg[avcodec,avdevice,avfilter,avformat,core,gpl,swresample,swscale,x264] --recurse
vcpkg install gstreamer[plugins-base,plugins-good,plugins-bad,plugins-ugly,libav] gst-rtsp-server --recurse
vcpkg install opencv[ade,contrib,cuda,cudnn,dnn,dnn-cuda,ffmpeg,freetype,gstreamer,highgui,ipp,jpeg,nonfree,openjpeg,png,quirc,thread,tiff,webp] --recurse
# vcpkg port 未实现 windows 端编译，仅供参考
vcpkg install onnxruntime[cuda,tensorrt] --recurse
# 清理
apt-get autoremove
```

windows 同理在环境变量中添加 VCPKG_ROOT 和 vcpkg 可执行文件，此外 cudnn、tensorRT 都是 zip package 形式，配置环境变量较为复杂，建议先安装 cuda，然后把 cudnn.zip 的内容复制到 cuda 的相同路径下，注意 cudnn 会有一级额外的 cuda 版本，eg:${CUDNN_PATH}/bin/12.8/${.dll} 复制 ${.dll} 到 {CUDA_PATH}/bin 下即可，{CUDNN_PATH} 的 include 和 lib 都需要同样复制过去。

# build

https://blog.csdn.net/zz460833359/article/details/123333030

## test
```
cmake --preset debug
cmake --build build/debug
```

重新开发需要下载 VSCode 的 CMake Tools，切换的侧边栏的 CMake 点击配置，可以搜索到相关的头文件(逻辑见.vscode/c_cpp_properties.json)，添加新的头文件重新配置即可

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