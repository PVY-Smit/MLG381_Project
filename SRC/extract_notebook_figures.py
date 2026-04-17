"""Export PNG outputs from project notebooks into SRC/assets/notebook_figures/."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "NOTEBOOKS"
OUT_DIR = Path(__file__).resolve().parent / "assets" / "notebook_figures"


def extract_from_notebook(nb_path: Path) -> list[dict]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    last_heading = "Notebook figure"
    manifest: list[dict] = []
    fig_i = 0
    stem = nb_path.stem
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            m = re.search(r"^#+\s*(.+)$", src, re.MULTILINE)
            if m:
                last_heading = m.group(1).strip()
        if cell.get("cell_type") != "code":
            continue
        for out_cell in cell.get("outputs", []):
            data = out_cell.get("data") or {}
            if "image/png" not in data:
                continue
            fig_i += 1
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            png_bytes = base64.b64decode(raw)
            fname = f"{stem}_{fig_i:02d}.png"
            (OUT_DIR / fname).write_bytes(png_bytes)
            manifest.append(
                {
                    "file": fname,
                    "caption": last_heading,
                    "notebook": stem,
                }
            )
    return manifest


def main() -> None:
    all_rows: list[dict] = []
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        all_rows.extend(extract_from_notebook(nb_path))
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(all_rows, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_rows)} figure(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
