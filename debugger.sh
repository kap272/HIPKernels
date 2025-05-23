#!/bin/bash

set -e  

ROCM_VERSION="6.3.0"
ROCM_PATH="/opt/rocm-${ROCM_VERSION}"
ROCM_LIB="${ROCM_PATH}/lib"
ROCM_INCLUDE="${ROCM_PATH}/include"
NUM_CORES=$(nproc)
BUILD_ROOT="$HOME/rocm-gdb-build"

sudo apt update
sudo apt install -y build-essential cmake ninja-build texinfo   python3 python3-dev zlib1g-dev libexpat-dev liblzma-dev libgmp-dev   libbabeltrace-dev libncurses-dev flex bison pkg-config git
sudo apt install -y bison flex gcc make ncurses-dev texinfo g++ zlib1g-dev libexpat-dev python3-dev liblzma-dev libgmp-dev libmpfr-dev
sudo apt install -y python3-dev libpython3-dev

export PATH=/usr/bin:$PATH
export PKG_CONFIG_PATH="${ROCM_LIB}/pkgconfig"
export LDFLAGS="-Wl,-rpath=${ROCM_LIB}"

mkdir -p "${BUILD_ROOT}"
cd "${BUILD_ROOT}"

if [ ! -d ROCdbgapi  ]; then
  git clone https://github.com/ROCm/ROCdbgapi.git
fi
cd  ROCdbgapi
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${ROCM_PATH}
make -j${NUM_CORES}
sudo make install

PKG_FILE="${ROCM_LIB}/pkgconfig/amd-dbgapi.pc"
sudo mkdir -p "$(dirname ${PKG_FILE})"
sudo tee "${PKG_FILE}" > /dev/null <<EOF
prefix=${ROCM_PATH}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: amd-dbgapi
Description: AMD Debugger API library
Version: 0.77.3
Libs: -L\${libdir} -lamd-dbgapi
Cflags: -I\${includedir}
EOF

cd "${ROCM_LIB}"
if [ ! -e libamd-dbgapi.so ]; then
  sudo ln -sf librocm-dbgapi.so libamd-dbgapi.so
fi

cd "${BUILD_ROOT}"
if [ ! -d ROCgdb ]; then
  git clone https://github.com/ROCm/ROCgdb.git
fi
cd ROCgdb
rm -rf build
mkdir build && cd build

export PKG_CONFIG_PATH="/opt/rocm-6.3.0/lib/pkgconfig"
export CFLAGS="-I/opt/rocm-6.3.0/include"
export LDFLAGS="-L/opt/rocm-6.3.0/lib -Wl,-rpath=/opt/rocm-6.3.0/lib"

../configure --program-prefix=roc \
  --enable-64-bit-bfd --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
  --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
  --disable-gdbtk --disable-gprofng --disable-shared --with-expat \
  --with-system-zlib --without-guile --with-babeltrace --with-lzma \
  --with-python=/usr/bin/python3 --with-amd-dbgapi

make -j192 

mkdir -p /root/rocm-gdb-build/ROCgdb/build/gdb/data-directory/python
cp -r /root/rocm-gdb-build/ROCgdb/gdb/python/lib/gdb /root/rocm-gdb-build/ROCgdb/build/gdb/data-directory/python/

sudo ln -sf "$(realpath ./gdb/gdb)" /usr/local/bin/rocgdb

