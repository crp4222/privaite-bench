#!/bin/sh
# Fetch the systems that do not install with pip, at pinned versions, into
# ~/.cache/privaite-bench/tools. Each download is checked against its sha256.
set -e
T="$HOME/.cache/privaite-bench/tools"
mkdir -p "$T"

check() {  # file expected-sha256
  got=$(shasum -a 256 "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "sha256 mismatch for $1: $got" >&2; exit 1; }
}

# agentgateway 1.6.0, official macOS arm64 release binary.
mkdir -p "$T/agentgateway"
if [ ! -x "$T/agentgateway/agentgateway" ]; then
  curl -sL -o "$T/agentgateway/agentgateway" \
    https://github.com/agentgateway/agentgateway/releases/download/v1.6.0/agentgateway-darwin-arm64
  chmod +x "$T/agentgateway/agentgateway"
fi
check "$T/agentgateway/agentgateway" 6cebe8bd57262edce23a65a376d2f5a27b97cdfef609740642edbe75e1e9c685

# occludra/gateway, source at a pinned commit (it ships no release).
[ -d "$T/occludra/.git" ] || git clone -q https://github.com/occludra/gateway "$T/occludra"
git -C "$T/occludra" fetch -q origin
git -C "$T/occludra" checkout -q e948e230d78f83852c5662d312ed846c32520c17

# Kiji Privacy Proxy 1.6.4: the official macOS arm64 app (signed by Dataiku, team
# 5KP9F667SJ). Only its Go backend and bundled model are used. 1.6.7 ships no
# macOS build, so 1.6.4 is the latest one that runs here.
mkdir -p "$T/kiji"
if [ ! -d "$T/kiji/app/Kiji Privacy Proxy.app" ]; then
  until [ "$(stat -f %z "$T/kiji/kiji.zip" 2>/dev/null || echo 0)" -ge 1408000000 ]; do
    curl -sL -C - --retry 10 -o "$T/kiji/kiji.zip" \
      https://github.com/dataiku/kiji-proxy/releases/download/v1.6.4/Kiji-Privacy-Proxy-1.6.4-arm64-mac.zip
  done
  check "$T/kiji/kiji.zip" 9819977ead9df8d2e4f2c8aad3097f8d3ca377f2838996a176e385aad6e0cbd4
  rm -rf "$T/kiji/app" && mkdir -p "$T/kiji/app" && unzip -q "$T/kiji/kiji.zip" -d "$T/kiji/app"
fi

echo "tools ready in $T"
