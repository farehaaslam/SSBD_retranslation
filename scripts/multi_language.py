"""Compatibility launcher for the modular live translation application."""

from pathlib import Path
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main import main


if __name__ == "__main__":
    main()
