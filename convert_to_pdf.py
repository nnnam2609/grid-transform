from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LOCAL_TOOL = Path(__file__).resolve().parent / "local" / "report_tools" / "convert_to_pdf.py"


def load_local_main():
    if not LOCAL_TOOL.exists():
        raise SystemExit(
            "The tracked PDF conversion command was moved to local/report_tools. "
            "Use the local copy or restore that folder first."
        )
    local_dir = str(LOCAL_TOOL.parent)
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    spec = importlib.util.spec_from_file_location("local_report_convert_to_pdf", LOCAL_TOOL)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load local report tool: {LOCAL_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    raise SystemExit(load_local_main()())
