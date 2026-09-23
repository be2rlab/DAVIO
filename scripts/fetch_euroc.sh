#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: fetch_euroc.sh <data_root> [seq ...]}"
shift || true
SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
  SEQS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult)
fi

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
DEST_ROOT="${DATA_ROOT}/euroc"
CACHE="${DEST_ROOT}/.download_cache"
mkdir -p "${CACHE}"

_category_for() {
  case "$1" in
    MH_*) echo "machine_hall" ;;
    V1_*) echo "vicon_room1" ;;
    V2_*) echo "vicon_room2" ;;
    *) echo "unknown: ${1}" >&2; exit 1 ;;
  esac
}
_dest_dir_for() {
  case "$1" in
    machine_hall) echo "machine_hall" ;;
    vicon_room1|vicon_room2) echo "vicon_room" ;;
    *) echo "unknown category ${1}" >&2; exit 1 ;;
  esac
}
_url_for() {
  case "$1" in
    machine_hall) echo "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content" ;;
    vicon_room1)  echo "https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content" ;;
    vicon_room2)  echo "https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content" ;;
    *) echo "no URL for category ${1}" >&2; exit 1 ;;
  esac
}

# Categories actually needed for the requested sequences, de-duplicated.
declare -A NEEDED
for SEQ in "${SEQS[@]}"; do
  NEEDED["$(_category_for "${SEQ}")"]=1
done

for CATEGORY in "${!NEEDED[@]}"; do
  DEST_DIR="${DEST_ROOT}/$(_dest_dir_for "${CATEGORY}")"
  ZIP="${CACHE}/${CATEGORY}.zip"
  MARKER="${DEST_DIR}/.extracted_${CATEGORY}"
  if [ -f "${MARKER}" ]; then
    echo "already extracted: ${CATEGORY} -> ${DEST_DIR}"
    continue
  fi
  if [ ! -f "${ZIP}" ]; then
    echo "downloading ${CATEGORY} (this is a multi-GB bundle covering every sequence in the category) ..."
    curl -fL --progress-bar -A "${UA}" -o "${ZIP}.part" "$(_url_for "${CATEGORY}")"
    mv "${ZIP}.part" "${ZIP}"
  fi
  echo "extracting ${CATEGORY} into ${DEST_DIR} ..."
  mkdir -p "${DEST_DIR}"
  TMP_EXTRACT="${CACHE}/${CATEGORY}_extract"
  rm -rf "${TMP_EXTRACT}"
  mkdir -p "${TMP_EXTRACT}"
  unzip -q -o "${ZIP}" -d "${TMP_EXTRACT}"
  while IFS= read -r -d '' SEQ_ZIP; do
    SEQ_NAME="$(basename "${SEQ_ZIP}" .zip)"
    mkdir -p "${DEST_DIR}/${SEQ_NAME}"
    unzip -q -o "${SEQ_ZIP}" -d "${DEST_DIR}/${SEQ_NAME}"
  done < <(find "${TMP_EXTRACT}" -name "*.zip" -print0)
  rm -rf "${TMP_EXTRACT}"
  touch "${MARKER}"
  rm -f "${ZIP}"
  echo "  -> ${DEST_DIR}"
done

echo
echo "done. Use it with:"
echo "  make eval SEQ=${SEQS[0]} DATA_ROOT=${DATA_ROOT}"
