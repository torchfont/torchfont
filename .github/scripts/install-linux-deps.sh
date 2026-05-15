#!/usr/bin/env bash
set -euo pipefail

target="${1:-host}"

case "${target}" in
  host)
    apt-get update
    apt-get install -y --no-install-recommends \
      pkg-config \
      libfreetype6-dev \
      libfontconfig1-dev
    ;;
  x86_64)
    yum install -y \
      freetype-devel \
      fontconfig-devel \
      pkgconfig
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
      libfontconfig1-dev:arm64
    export RUSTFLAGS="${RUSTFLAGS:-} -L native=/usr/lib/aarch64-linux-gnu"
    ;;
  *)
    echo "Unsupported Linux dependency target: ${target}" >&2
    exit 1
    ;;
esac
