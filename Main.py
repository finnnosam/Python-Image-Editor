"""Compatibility launcher. Install with `pip install -e .`; implementation is pypaint."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from pypaint.app import main
    main()
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from pypaint import window
    sys.modules[__name__] = window
