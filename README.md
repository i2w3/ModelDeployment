# config
Windows 必须通过 runVSCode.ps1 进入项目才能激活 MSVC 的环境，默认使用的是 x64 - x64 的设置，需要修改查看[微软 MSVC 的说明](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=visualstudio)来修改 runVSCode.ps1 

# build
先准备好 opencv 和 spdlog

## opencv
Windows 从[这里](https://opencv.org/releases/)下载编译好的文件即可，将 exe 文件以压缩包形式打开，提取里面的 opencv/build 文件到 libs/opencv/build，其余平台重新编译即可。

## spdlog
```
cd libs
git clone https://github.com/gabime/spdlog.git
cd spdlog
mkdir build
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=dist
cmake --build build
```

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