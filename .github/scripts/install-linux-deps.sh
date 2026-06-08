#!/usr/bin/env sh
set -eu

target="${1:-host}"

case "${target}" in
  host)
    apt-get update
    apt-get install -y --no-install-recommends \
      pkg-config \
      libfreetype6-dev \
      libfontconfig1-dev \
      clang \
      ninja-build
    ;;
  x86_64)
    yum install -y \
      freetype-devel \
      fontconfig-devel \
      pkgconfig \
      clang \
      gcc-toolset-12-gcc-c++ \
      ninja-build

    # AlmaLinux 8 ships GCC 8.5 which lacks C++20 <bit>; point clang at GCC 12 headers
    printf '#!/bin/sh\nexec /usr/bin/clang++ --gcc-toolchain=/opt/rh/gcc-toolset-12/root/usr "$@"\n' \
      > /usr/local/bin/clang++
    chmod +x /usr/local/bin/clang++
    ;;
  aarch64)
    . /etc/os-release

    dpkg --add-architecture arm64
    sed -i 's#^deb #deb [arch=amd64] #' /etc/apt/sources.list
    cat >/etc/apt/sources.list.d/arm64.list <<EOF
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME} main
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-updates main
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-security main
EOF

    apt-get update
    apt-get install -y --no-install-recommends \
      pkg-config \
      libfreetype6-dev:arm64 \
      libfontconfig1-dev:arm64 \
      clang \
      g++-aarch64-linux-gnu \
      ninja-build
    export PKG_CONFIG_ALLOW_CROSS=1
    export PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig
    ;;
  *)
    echo "Unsupported Linux dependency target: ${target}" >&2
    exit 1
    ;;
esac
