#!/usr/bin/env python3
import datetime
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content"
OUTPUT_PATH = CONTENT_DIR / "slug-index.md"
SECTIONS = ["posts", "blogs"]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def parse_front_matter(text: str) -> dict:
    lines = text.splitlines()
    if not lines:
        return {}

    if lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is None:
            return {}
        front_lines = lines[1:end]
        data = {}
        for line in front_lines:
            if not line.strip() or line.strip().startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = _strip_quotes(value)
            data[key] = value
        return data

    if lines[0].strip() == "+++":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "+++":
                end = i
                break
        if end is None:
            return {}
        front_lines = lines[1:end]
        data = {}
        for line in front_lines:
            match = re.match(r"\s*([A-Za-z0-9_-]+)\s*=\s*(.+)\s*", line)
            if not match:
                continue
            key = match.group(1).strip().lower()
            value = _strip_quotes(match.group(2))
            data[key] = value
        return data

    return {}


def collect_pages():
    pages = []
    for section in SECTIONS:
        section_dir = CONTENT_DIR / section
        if not section_dir.exists():
            continue
        for path in section_dir.rglob("*.md"):
            if path.name == "_index.md":
                continue
            text = path.read_text(encoding="utf-8")
            fm = parse_front_matter(text)
            title = fm.get("title") or path.stem
            slug = fm.get("slug") or fm.get("url") or path.stem
            slug = slug.strip()
            pages.append(
                {
                    "title": title.strip(),
                    "slug": slug,
                    "section": section,
                    "rel_url": f"/{section}/{slug}/",
                }
            )
    pages.sort(key=lambda item: (item["section"], item["title"]))
    return pages


def build_markdown(pages):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        'title: "Slug 索引"',
        "draft: true",
        "---",
        "",
        f"生成时间：{now}",
        "",
    ]

    for section in SECTIONS:
        section_pages = [p for p in pages if p["section"] == section]
        if not section_pages:
            continue
        lines.append(f"## {section}")
        for page in section_pages:
            lines.append(f"- {page['title']} — `{page['slug']}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    pages = collect_pages()
    content = build_markdown(pages)
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
