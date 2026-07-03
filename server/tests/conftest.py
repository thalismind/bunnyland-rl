from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
BUNNYLAND_SERVER_SRC = ROOT.parent / "bunnyland-server" / "src"

for path in (SERVER_SRC, BUNNYLAND_SERVER_SRC):
    if path.exists():
        sys.path.insert(0, str(path))
