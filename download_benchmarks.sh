#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_benchmarks.sh -- fetch and verify the MAGICC benchmark assemblies.
#
# The assemblies are distributed as GitHub Release assets on this repository
# (they are far too large for git).  Any archive over 1.8 GB is uploaded as
# 1.5 GiB parts -- safely under GitHub's 2 GiB per-asset limit -- which this
# script concatenates back into a single .tar.gz before verifying it.
#
#   bash download_benchmarks.sh                      # all eight sets
#   bash download_benchmarks.sh set_C_clean set_E    # named sets only
#   bash download_benchmarks.sh --dest /data/bench   # choose a destination
#   bash download_benchmarks.sh --no-extract         # keep the tarballs
#   bash download_benchmarks.sh --verify-only        # re-verify what is here
#   bash download_benchmarks.sh --full-verify        # also check every assembly
#
# Every download is checked against SHA256SUMS (the assets as uploaded) and
# SHA256SUMS.tar (the reassembled archives).  The script is resumable: curl -C -
# continues a partial download and finished files are skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="renmaotian/magicc-data"
TAG="v1.0.0"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
ALL_SETS="set_A set_B set_C_clean set_D_clean set_E set_F set_G set_H"

DEST="benchmarks"
EXTRACT=1
VERIFY_ONLY=0
FULL_VERIFY=0
SETS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --no-extract) EXTRACT=0; shift ;;
    --verify-only) VERIFY_ONLY=1; EXTRACT=0; shift ;;
    --full-verify) FULL_VERIFY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) SETS="$SETS $1"; shift ;;
  esac
done
[ -z "$SETS" ] && SETS="$ALL_SETS"

for tool in curl sha256sum tar; do
  command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 1; }
done

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DEST"
cd "$DEST"

fetch () {  # fetch <name>
  local name="$1"
  if [ -f "$name" ]; then echo "  have  $name"; return 0; fi
  echo "  get   $name"
  curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - \
       -o "$name.partial" "$BASE/$name"
  mv "$name.partial" "$name"
}

# --- checksum manifests ---------------------------------------------------
for m in SHA256SUMS SHA256SUMS.tar; do
  [ -f "$m" ] || curl -fsSL -o "$m" "$BASE/$m"
done

# --- per-set download, reassembly and verification ------------------------
rc=0
for s in $SETS; do
  echo "== $s"
  parts=$(grep -E "  ${s}\.tar\.gz\.part[0-9]+$" SHA256SUMS | awk '{print $2}' | sort || true)
  if [ -n "$parts" ]; then
    if [ "$VERIFY_ONLY" -eq 0 ]; then for p in $parts; do fetch "$p"; done; fi
    grep -E "  ${s}\.tar\.gz\.part[0-9]+$" SHA256SUMS > ".${s}.sums"
    if ! sha256sum -c ".${s}.sums"; then echo "  FAIL: part checksum mismatch for $s" >&2; rc=1; continue; fi
    if [ ! -f "${s}.tar.gz" ]; then
      echo "  join  ${s}.tar.gz"
      cat $parts > "${s}.tar.gz"
    fi
  else
    if [ "$VERIFY_ONLY" -eq 0 ]; then fetch "${s}.tar.gz"; fi
    [ -f "${s}.tar.gz" ] || { echo "  FAIL: ${s}.tar.gz not present" >&2; rc=1; continue; }
  fi

  grep -E "  ${s}\.tar\.gz$" SHA256SUMS.tar > ".${s}.tar.sums"
  if ! sha256sum -c ".${s}.tar.sums"; then echo "  FAIL: archive checksum mismatch for $s" >&2; rc=1; continue; fi
  echo "  ok    ${s}.tar.gz verified"

  if [ "$EXTRACT" -eq 1 ]; then
    echo "  untar ${s}/"
    tar -xzf "${s}.tar.gz"
    if [ "$FULL_VERIFY" -eq 1 ]; then
      man="$REPO_ROOT/provenance/assemblies/${s}_sha256.txt"
      if [ -f "$man" ]; then
        echo "  check every assembly of $s"
        sha256sum -c "$man" --quiet || { echo "  FAIL: per-assembly checksum mismatch for $s" >&2; rc=1; }
      else
        echo "  note  per-assembly manifest not found at $man" >&2
      fi
    else
      echo "  note  per-assembly checksums: provenance/assemblies/${s}_sha256.txt (--full-verify)"
    fi
  fi
  rm -f ".${s}.sums" ".${s}.tar.sums"
done

if [ "$rc" -eq 0 ]; then
  echo
  echo "All requested sets downloaded and verified into $(pwd)"
  echo "Per-assembly SHA256 manifests (12,848 entries in total) are at"
  echo "  provenance/assemblies/<set>_sha256.txt"
  echo "Re-run with --full-verify to check every assembly file."
else
  echo; echo "One or more sets failed verification. Re-run to resume." >&2
fi
exit "$rc"
