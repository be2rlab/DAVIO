#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: fetch_tumvi.sh <data_root> [seq ...]}"
shift || true
SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
  SEQS=(room1 room2 room3 room4 room5 room6)
fi

BASE="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"
DEST_ROOT="${DATA_ROOT}/tumvi"
CACHE="${DEST_ROOT}/.download_cache"
mkdir -p "${CACHE}"

for seq in "${SEQS[@]}"; do
  name="dataset-${seq}_512_16"
  tarball="${CACHE}/${name}.tar"
  dest="${DEST_ROOT}/${seq}"

  if [ -d "${dest}/mav0" ]; then
    echo "already extracted: ${dest}"
    continue
  fi

  if [ ! -f "${tarball}" ]; then
    echo "downloading ${name}.tar ..."
    # --continue-at so an interrupted multi-gigabyte fetch resumes instead of restarting.
    curl -fL --continue-at - --retry 5 --retry-delay 5 \
         -o "${tarball}.part" "${BASE}/${name}.tar"
    mv "${tarball}.part" "${tarball}"
  fi

  echo "extracting ${name} -> ${dest}"
  mkdir -p "${dest}"
  # The archive holds <name>/mav0/...; strip its top level so the layout matches EuRoC's.
  tar -xf "${tarball}" -C "${dest}" --strip-components=1
  test -d "${dest}/mav0" || { echo "error: ${dest}/mav0 missing after extract" >&2; exit 1; }
done

echo "TUM-VI ready under ${DEST_ROOT}"
