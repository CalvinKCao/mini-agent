# Slidev (LNAI skill)

Agent skill for [Slidev](https://sli.dev) — Markdown/Vite slides with code, diagrams, and presenter mode.

## Source

Derived from the [`slidevjs/slidev` repository `skills/slidev/`](https://github.com/slidevjs/slidev/tree/main/skills/slidev) layout. Maintenance notes and doc provenance live in [`GENERATION.md`](GENERATION.md).

## In this repo

The canonical copy is under [`.ai/skills/slidev/`](../../) (this directory). [LNAI](https://lnai.sh) syncs it to each tool’s skills folder (symlinks), for example:

- Cursor: `.cursor/skills/slidev`
- Claude Code: `.claude/skills/slidev`
- Copilot: `.github/skills/slidev`
- Codex / OpenCode: `.agents/skills/slidev`
- Windsurf: `.windsurf/skills/slidev`
- Gemini CLI: `.gemini/skills/slidev`

After pulling changes, run `lnai sync` from the project root so generated paths stay current (see [version control](https://lnai.sh/reference/version-control/)).

## Contents

- **`SKILL.md`** — Main skill: quick start, syntax, and tables linking to `references/`.
- **`references/`** — Topic files (`core-*`, `code-*`, `diagram-*`, etc.).

## External links

- [Slidev docs](https://sli.dev)
- [Theme gallery](https://sli.dev/resources/theme-gallery)

License: MIT (same as Slidev).
