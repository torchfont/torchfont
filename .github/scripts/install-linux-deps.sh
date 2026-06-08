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
  x86_64|aarch64)
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
  *)
    echo "Unsupported Linux dependency target: ${target}" >&2
    exit 1
    ;;
esac
