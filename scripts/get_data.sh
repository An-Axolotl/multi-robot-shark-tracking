#!/usr/bin/env bash
# Clone the (private) CSULB white-shark VPS data and link it as ./data_src.
# Needs read access to ciherrera20/WhiteShark-DataExploration (SSH key on GitHub).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-../WhiteShark-DataExploration}
[ -d "$DEST" ] || git clone git@github.com:ciherrera20/WhiteShark-DataExploration.git "$DEST"
ln -sfn "$(cd "$DEST" && pwd)/data" data_src
ls data_src/SharkArray-01-animals.csv
