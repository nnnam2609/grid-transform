from __future__ import annotations

import _bootstrap  # noqa: F401

from tools.report.export_nnunet_elements import main


if __name__ == "__main__":
    raise SystemExit(main(["--mode", "legend"]))
