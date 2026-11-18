#!/usr/bin/env bash
# Fetch the Woodpecker CI documentation (Apache-2.0) at a pinned commit.
# Usage: bash scripts/fetch_corpus.sh
set -euo pipefail

REPO="https://github.com/woodpecker-ci/woodpecker.git"
COMMIT="fd00b3d79e7cdb193c6629b28b3171e91c2205f8"  # 2026-11-14
DEST="data/raw/woodpecker"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin "$REPO"
git -C "$TMP" sparse-checkout set --no-cone docs/docs/ LICENSE
git -C "$TMP" fetch -q --depth 1 origin "$COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$TMP/docs/docs/." "$DEST/"
cp "$TMP/LICENSE" "$DEST/LICENSE.woodpecker"
echo "$COMMIT" > "$DEST/SOURCE_COMMIT"

echo "Fetched $(find "$DEST" -name '*.md' | wc -l | tr -d ' ') markdown files into $DEST @ ${COMMIT:0:7}"
