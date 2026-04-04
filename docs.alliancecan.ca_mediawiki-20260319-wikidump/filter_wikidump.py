#!/usr/bin/env python3
"""
Stream-filter a MediaWiki 0.11 XML dump for English / coding-assistant use.

Default drops:
  - Any page whose title ends with /fr (French subpages)
  - Translations namespace (1198) and Translations talk (1199) — Translate extension units
  - Talk namespaces: 1, 3, 5, 7, 9, 11, 13, 15

Original dump is left untouched; writes sibling files *-filtered.xml and *-filtered-titles.txt
unless --out is set.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MW_NS = "http://www.mediawiki.org/xml/export-0.11/"
MW = f"{{{MW_NS}}}"

TALK_NS = frozenset({1, 3, 5, 7, 9, 11, 13, 15, 1199})
TRANSLATIONS_NS = frozenset({1198, 1199})

# Title-prefix heuristics for plain-text title lists (no namespace id).
TALK_TITLE_PREFIXES = (
    "Talk:",
    "User talk:",
    "CCWiki talk:",
    "File talk:",
    "MediaWiki talk:",
    "Template talk:",
    "Help talk:",
    "Category talk:",
    "Translations talk:",
)


def xml_source_path(src: Path) -> tuple[Path, Path | None]:
    """Return path safe for parsing. Many API dumps omit the final </mediawiki>."""
    raw = src.read_bytes()
    stripped = raw.rstrip()
    if stripped.endswith(b"</mediawiki>"):
        return src, None
    fd, name = tempfile.mkstemp(suffix=".xml", prefix="wikidump-repaired-")
    with os.fdopen(fd, "wb") as wf:
        wf.write(stripped + b"\n</mediawiki>\n")
    p = Path(name)
    return p, p


def should_keep(title: str, ns: int, args: argparse.Namespace) -> bool:
    if args.drop_fr and title.endswith("/fr"):
        return False
    if args.drop_translations and ns in TRANSLATIONS_NS:
        return False
    if args.drop_talk and ns in TALK_NS:
        return False
    if args.drop_template and ns == 10:
        return False
    if args.drop_mediawiki and ns == 8:
        return False
    if args.drop_category and ns == 14:
        return False
    if args.drop_file and ns == 6:
        return False
    return True


def title_index_keep(title: str, args: argparse.Namespace) -> bool:
    """Same policy as should_keep for all-pages title exports (one title per line, no ns field)."""
    if args.drop_fr and title.endswith("/fr"):
        return False
    if args.drop_translations and title.startswith("Translations:"):
        return False
    if args.drop_talk and any(title.startswith(p) for p in TALK_TITLE_PREFIXES):
        return False
    if args.drop_template and title.startswith("Template:"):
        return False
    if args.drop_mediawiki and title.startswith("MediaWiki:"):
        return False
    if args.drop_category and title.startswith("Category:"):
        return False
    if args.drop_file and title.startswith("File:"):
        return False
    return True


def filter_title_index(src: Path, dst: Path, args: argparse.Namespace) -> tuple[int, int]:
    kept = skipped = 0
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t:
            continue
        if title_index_keep(t, args):
            lines_out.append(t)
            kept += 1
        else:
            skipped += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept, skipped


def filter_dump(
    src: Path,
    out_xml: Path,
    out_titles: Path | None,
    args: argparse.Namespace,
) -> tuple[int, int]:
    kept = skipped = 0
    titles_out: list[str] = []

    parse_path, tmp_repair = xml_source_path(src)
    try:
        with open(parse_path, "rb") as rf:
            header_parts: list[bytes] = []
            while True:
                line = rf.readline()
                if not line:
                    break
                header_parts.append(line)
                if line.lstrip().startswith(b"<mediawiki"):
                    break
            header = b"".join(header_parts)

        context = ET.iterparse(str(parse_path), events=("end",))

        with open(out_xml, "wb") as wf:
            wf.write(header)

            for _event, elem in context:
                tag = elem.tag
                if tag == f"{MW}siteinfo":
                    wf.write(ET.tostring(elem, encoding="utf-8", default_namespace=None))
                    wf.write(b"\n")
                    continue
                if tag != f"{MW}page":
                    continue

                title_el = elem.find(f"{MW}title")
                ns_el = elem.find(f"{MW}ns")
                title = (title_el.text or "").strip() if title_el is not None else ""
                try:
                    ns = int((ns_el.text or "0").strip()) if ns_el is not None else 0
                except ValueError:
                    ns = 0

                if should_keep(title, ns, args):
                    wf.write(ET.tostring(elem, encoding="utf-8", default_namespace=None))
                    wf.write(b"\n")
                    titles_out.append(title)
                    kept += 1
                else:
                    skipped += 1

            wf.write(b"</mediawiki>\n")
    finally:
        if tmp_repair is not None:
            tmp_repair.unlink(missing_ok=True)

    if out_titles is not None:
        out_titles.write_text("\n".join(titles_out) + ("\n" if titles_out else ""), encoding="utf-8")

    return kept, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "src",
        type=Path,
        nargs="?",
        default=None,
        help="Source MediaWiki XML dump (default: docs...-current.xml next to this script)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output XML path (default: <src-stem>-filtered.xml)",
    )
    p.add_argument(
        "--titles-out",
        type=Path,
        default=None,
        help="Write kept titles, one per line (default: same stem as --out + -titles.txt)",
    )
    p.add_argument("--no-drop-fr", action="store_true", help="Keep /fr subpages")
    p.add_argument("--no-drop-translations", action="store_true", help="Keep Translations: namespace")
    p.add_argument("--no-drop-talk", action="store_true", help="Keep talk namespaces")
    p.add_argument("--drop-template", action="store_true", help="Also drop Template (ns 10)")
    p.add_argument("--drop-mediawiki", action="store_true", help="Also drop MediaWiki (ns 8)")
    p.add_argument("--drop-category", action="store_true", help="Also drop Category (ns 14)")
    p.add_argument("--drop-file", action="store_true", help="Also drop File (ns 6)")
    p.add_argument(
        "--title-index",
        type=Path,
        default=None,
        help="Also filter a newline-separated title list (e.g. api export of alltitles)",
    )
    p.add_argument(
        "--title-index-out",
        type=Path,
        default=None,
        help="Output for --title-index (default: <title-index-stem>-filtered.txt)",
    )
    p.add_argument(
        "--only-title-index",
        action="store_true",
        help="Only run --title-index filtering (no XML read)",
    )
    args = p.parse_args()

    args.drop_fr = not args.no_drop_fr
    args.drop_translations = not args.no_drop_translations
    args.drop_talk = not args.no_drop_talk

    default_xml = Path(__file__).resolve().parent / "docs.alliancecan.ca_mediawiki-20260325-current.xml"

    if args.only_title_index:
        if args.title_index is None:
            raise SystemExit("--only-title-index requires --title-index")
        ti = args.title_index.resolve()
        if not ti.is_file():
            raise SystemExit(f"missing --title-index: {ti}")
        ti_out = args.title_index_out or ti.with_name(f"{ti.stem}-filtered.txt")
        k2, s2 = filter_title_index(ti, ti_out, args)
        print(f"wrote {ti_out} ({k2} titles)")
        print(f"kept {k2}, skipped {s2}")
        return

    src = (args.src or default_xml).resolve()
    if not src.is_file():
        raise SystemExit(f"missing source: {src}")

    out_xml = args.out or src.with_name(f"{src.stem}-filtered.xml")
    titles_path = args.titles_out
    if titles_path is None:
        titles_path = out_xml.with_name(f"{out_xml.stem}-titles.txt")

    kept, skipped = filter_dump(src, out_xml, titles_path, args)
    print(f"wrote {out_xml}")
    print(f"wrote {titles_path} ({kept} titles)")
    print(f"kept {kept} pages, skipped {skipped}")

    if args.title_index is not None:
        ti = args.title_index.resolve()
        if not ti.is_file():
            raise SystemExit(f"missing --title-index: {ti}")
        ti_out = args.title_index_out or ti.with_name(f"{ti.stem}-filtered.txt")
        k2, s2 = filter_title_index(ti, ti_out, args)
        print(f"wrote {ti_out} ({k2} titles from index)")
        print(f"title index: kept {k2}, skipped {s2}")


if __name__ == "__main__":
    main()
