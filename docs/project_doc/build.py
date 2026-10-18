"""
Build docs/SmartSpray_Project_Documentation.docx (and the Markdown copy
docs/PROJECT_DOCUMENTATION.md) from the chapters in docs/project_doc/.

Two passes: the first build is rendered to PDF with LibreOffice to find the
page of every heading; the second build writes those page numbers into the
contents page.

    python docs/project_doc/build.py
Needs: node + the `docx` npm package, LibreOffice (soffice), poppler (pdftotext).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "project_doc"
OUT = ROOT / "docs" / "SmartSpray_Project_Documentation.docx"
SOFFICE = "/mnt/skills/public/docx/scripts/office/soffice.py"


def build(out, pages=None):
    args = ["node", str(SRC / "build_docx.js"), str(out)]
    if pages:
        args.append(str(pages))
    subprocess.run(args, check=True, cwd=ROOT)


def to_pdf(docx, workdir):
    cmd = [sys.executable, SOFFICE] if Path(SOFFICE).exists() else ["soffice"]
    subprocess.run(cmd + ["--headless", "--convert-to", "pdf", "--outdir", str(workdir), str(docx)],
                   check=True, cwd=workdir, capture_output=True, timeout=300)
    return workdir / (docx.stem + ".pdf")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def headings():
    out = []
    ch = 0
    sec = 0
    in_app = False
    for f in sorted(SRC.glob("[0-9][0-9]_*.md")):
        code = False
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                code = not code
                continue
            if code:
                continue
            m = re.match(r"^(#{1,2})\s+(.*)$", line)
            if not m:
                continue
            text = re.sub(r"\*\*|`", "", m.group(2).strip())
            if len(m.group(1)) == 1:
                sec = 0
                in_app = text.startswith("Appendix")
                if not in_app:
                    ch += 1
                    out.append((f"{ch}  {text}", 1))
                else:
                    out.append((text, 1))
            else:
                sec += 1
                out.append((text if in_app else f"{ch}.{sec}  {text}", 2))
    return out


def fix_bookmark_ids(docx: Path) -> None:
    """docx-js writes every bookmark with w:id="1"; give each a unique id."""
    import shutil
    import zipfile

    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                n = iter(range(1, 10 ** 6))
                xml = re.sub(r'(<w:bookmarkStart w:name="h(\d+)") w:id="\d+"',
                             lambda m: f'{m.group(1)} w:id="{m.group(2)}"', xml)
                counter = iter(range(1, 10 ** 6))
                xml = re.sub(r'<w:bookmarkEnd w:id="\d+"/>',
                             lambda m: f'<w:bookmarkEnd w:id="{next(counter)}"/>', xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, docx)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        first = tmp / "pass1.docx"
        build(first)
        pdf = to_pdf(first, tmp)
        n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
        texts = [norm(subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(pdf), "-"],
                                     capture_output=True, text=True).stdout) for i in range(1, n + 1)]
        # skip the title page and the contents pages (they list every heading)
        start = next(i for i, t in enumerate(texts) if "executive summary" in t and "contents" not in t[:200])
        pages, cursor = {}, start
        for label, _ in headings():
            key = norm(label)
            for i in range(cursor, n):
                if key in texts[i]:
                    pages[label] = i + 1
                    cursor = i
                    break
            else:
                print("not found:", label)
        pmap = tmp / "pages.json"
        pmap.write_text(json.dumps(pages, indent=1), encoding="utf-8")
        build(OUT, pmap)
        fix_bookmark_ids(OUT)
        print(f"{len(pages)} headings placed; {n} pages")
    # Markdown copy for the repository (headings shifted down one level,
    # image paths made relative to docs/)
    md = "\n\n".join(f.read_text(encoding="utf-8") for f in sorted(SRC.glob("[0-9][0-9]_*.md")))
    out, code = [], False
    for line in md.splitlines():
        if line.startswith("```"):
            code = not code
        elif not code and re.match(r"^#{1,5} ", line):
            line = "#" + line
        elif not code and line.startswith("!["):
            line = line.replace("](docs/figures/", "](figures/").replace("](models/", "](../models/")
        out.append(line)
    title = ("# SmartSpray — Vision-Guided Autonomous UAV for Plant-Level Disease Detection and "
             "Variable-Rate Spraying\n\nComplete technical documentation · Team 134 · VIT Chennai · "
             "Guide: Rajesh M\n\n")
    (ROOT / "docs" / "PROJECT_DOCUMENTATION.md").write_text(title + "\n".join(out) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
