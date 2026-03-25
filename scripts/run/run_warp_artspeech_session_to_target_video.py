from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grid_transform.apps.warp_artspeech_session_to_target_video import main


if __name__ == "__main__":
    raise SystemExit(main())
