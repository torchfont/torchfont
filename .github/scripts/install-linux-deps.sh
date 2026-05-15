target="${1:-host}"

if [ "${target}" = "host" ]; then
  apt-get update
  apt-get install -y --no-install-recommends pkg-config libssl-dev libfreetype6-dev libfontconfig1-dev
elif [ "${target}" = "x86_64" ]; then
  yum install -y freetype-devel fontconfig-devel pkgconfig
elif [ "${target}" = "aarch64" ]; then
  . /etc/os-release

  dpkg --add-architecture arm64
  sed -i 's#^deb #deb [arch=amd64] #' /etc/apt/sources.list
  printf '%s\n' \
    "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME} main" \
    "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-updates main" \
    "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-security main" \
    >/etc/apt/sources.list.d/arm64.list

  apt-get update
  apt-get install -y --no-install-recommends pkg-config libfreetype6-dev:arm64 libfontconfig1-dev:arm64
else
  echo "Unsupported Linux dependency target: ${target}" >&2
  exit 1
fi
