#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  base)  REPO=depth-anything/DA3-BASE;      REVISION=f4a6c9b3c95e41c82048423d3493a81ec3fa810e ;;
  large) REPO=depth-anything/DA3-LARGE-1.1; REVISION=0e109ae307c5982f319a67cf6f9f99ccdc0ec97c ;;
  giant) REPO=depth-anything/DA3-GIANT-1.1; REVISION=72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19 ;;
  *) echo "usage: $(basename "$0") base|large|giant" >&2; exit 1 ;;
esac
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/thirdparty/da3_weights/$1"
if [ -f "$DEST/SHA256SUMS" ]; then
  echo "already present: $DEST"; exit 0
fi
command -v hf >/dev/null 2>&1 || { echo "'hf' not found: pip install -U huggingface_hub" >&2; exit 1; }
mkdir -p "$DEST"
hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"
( cd "$DEST" && find . -type f -not -path './.cache/*' -not -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "fetched $REPO @ $REVISION into $DEST"
