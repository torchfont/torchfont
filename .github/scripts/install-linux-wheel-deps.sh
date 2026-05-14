# This script is sourced by PyO3/maturin-action's before-script-linux hook.
set -e

if command -v yum >/dev/null 2>&1; then
  yum install -y freetype-devel fontconfig-devel pkgconfig
else
  case "${1:-}" in
    aarch64)
      deb_arch=arm64
      libdir=/usr/lib/aarch64-linux-gnu
      ;;
    *)
      echo "Unsupported apt-based Linux wheel target: ${1:-unknown}" >&2
      return 1
      ;;
  esac

  . /etc/os-release

  dpkg --add-architecture "${deb_arch}"
  sed -i -E 's#^deb (\[arch=[^]]+\] )?(http://archive.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu)#deb [arch=amd64] \2#' /etc/apt/sources.list
  printf '%s\n' \
    "deb [arch=${deb_arch}] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME} main universe restricted multiverse" \
    "deb [arch=${deb_arch}] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-updates main universe restricted multiverse" \
    "deb [arch=${deb_arch}] http://ports.ubuntu.com/ubuntu-ports ${VERSION_CODENAME}-security main universe restricted multiverse" \
    >/etc/apt/sources.list.d/"${deb_arch}".list

  apt-get update
  apt-get install -y --no-install-recommends pkg-config libfreetype6-dev:"${deb_arch}" libfontconfig1-dev:"${deb_arch}"

  export PKG_CONFIG_ALLOW_CROSS=1
  export PKG_CONFIG_LIBDIR="${libdir}/pkgconfig:/usr/share/pkgconfig"
  export RUSTFLAGS="${RUSTFLAGS:-} -L native=${libdir} -C link-arg=-Wl,-rpath-link,${libdir}"
fi
