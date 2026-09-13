"""Splices screens_data.json into template.html -> output.html.

Run after build_screens.py. output.html is what gets published via the
Artifact tool (same two-step pattern as the CE/PE Intraday Archive webpage).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    with open(HERE / "screens_data.json", encoding="utf-8") as f:
        screens_data = f.read()
    with open(HERE / "template.html", encoding="utf-8") as f:
        template = f.read()

    out = template.replace("__SCREENS_DATA_JSON__", screens_data)
    out_path = HERE / "output.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {len(out):,} bytes -> {out_path}")


if __name__ == "__main__":
    main()
