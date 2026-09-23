#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${HERE}/thirdparty/da3_weights"

usage() {
  echo "usage: $(basename "$0") base|large|<hf-repo-id> [--revision <sha>]" >&2
  echo >&2
  echo "  base   -> depth-anything/DA3-BASE       (0.12B, Apache-2.0)" >&2
  echo "  large  -> depth-anything/DA3-LARGE-1.1   (0.35B, CC BY-NC 4.0 -- non-commercial;" >&2
  echo "            see docs/SETUP.md, the HF card's own license tag disagrees with" >&2
  echo "            the upstream README's model-card table, which this pins to)" >&2
  echo "  A raw HF repo id REQUIRES --revision <sha>: this script refuses an unpinned" >&2
  echo "  'main' download, since a paper's numbers must trace to one fixed checkpoint." >&2
  exit 1
}

[ $# -ge 1 ] || usage
SELECTOR="$1"; shift || true
REVISION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --revision) REVISION="${2:?--revision needs a value}"; shift 2 ;;
    *) usage ;;
  esac
done

case "${SELECTOR}" in
  base)
    REPO_ID="depth-anything/DA3-BASE"
    REVISION="f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
    ALIAS="base"
    ;;
  large)
    REPO_ID="depth-anything/DA3-LARGE-1.1"
    REVISION="0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
    ALIAS="large"
    ;;
  *)
    REPO_ID="${SELECTOR}"
    ALIAS="$(echo "${REPO_ID}" | tr '/' '_')"
    if [ -z "${REVISION}" ]; then
      echo "error: a raw repo id needs --revision <sha> (refusing an unpinned 'main')" >&2
      usage
    fi
    ;;
esac

DEST="${DEST_ROOT}/${ALIAS}"
MANIFEST="${DEST}/SHA256SUMS"

if [ -f "${MANIFEST}" ]; then
  echo "already present and manifested: ${DEST} (rm -rf it to force a re-fetch)"
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' not found. pip install -U huggingface_hub (installed in" >&2
  echo "  docker/Dockerfile.gpu; on the host, 'pip install -U huggingface_hub')." >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "downloading ${REPO_ID} @ ${REVISION} -> ${DEST} ..."
hf download "${REPO_ID}" --revision "${REVISION}" --local-dir "${DEST}"

( cd "${DEST}" \
  && find . -type f -not -path './.cache/*' -not -name 'SHA256SUMS*' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${MANIFEST}.part" )
mv "${MANIFEST}.part" "${MANIFEST}"

echo "verified, manifest written: ${MANIFEST}"
echo "record ${REPO_ID} @ ${REVISION} in docs/SETUP.md if not already present."
