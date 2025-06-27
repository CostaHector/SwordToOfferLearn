
# Usage in windoes
## Download cmake for windows
https://cmake.org/download/

Windows x64 Installer:	cmake-4.1.0-rc1-windows-x86_64.msi

## Download cmake for windows
https://github.com/mstorsjo/llvm-mingw/releases

llvm-mingw-20250613-msvcrt-x86_64.zip

将压缩包解压到无空格目录（如 C:\mingw64），避免权限问题。‌配置环境变量‌

右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
在「系统变量」找到 Path → 点击「编辑」→「新建」→ 输入 MinGW 的 bin 路径（如 C:\mingw64\bin）711：
C:\mingw64\bin

确保如下可执行文件在系统环境变量path中
where gcc
where mingw32-make

## 重新打开VSCode
```bash
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
./SwordToOffer.exe

cmake .. -G "MinGW Makefiles";mingw32-make; ./SwordToOffer.exe
```