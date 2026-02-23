#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DOC_GLOBS = [
    'getting-started.mdx',
    'concepts.mdx',
    'bagel-api/**/*.mdx',
    'customization/**/*.mdx',
    'case-studies/**/*.mdx',
]

# Approved third-party import roots used in docs snippets.
ALLOWED_THIRD_PARTY_IMPORTS = {
    'bagel',
    'numpy',
    'biotite',
}


@dataclass
class Snippet:
    path: Path
    start_line: int
    code: str


def collect_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in DOC_GLOBS:
        files.update(root.glob(pattern))
    return sorted(files)


def extract_python_snippets(path: Path) -> list[Snippet]:
    snippets: list[Snippet] = []
    lines = path.read_text().splitlines()
    in_block = False
    block_start = 0
    block_lang = ''
    buf: list[str] = []

    for i, line in enumerate(lines, start=1):
        if not in_block:
            m = re.match(r'^```\s*([A-Za-z0-9_-]+)?\s*$', line)
            if m:
                lang = (m.group(1) or '').lower()
                in_block = True
                block_start = i + 1
                block_lang = lang
                buf = []
            continue

        if line.startswith('```'):
            if block_lang in {'python', 'py'}:
                code = '\n'.join(buf).strip('\n')
                if code and '# docs-snippet-skip' not in code:
                    snippets.append(Snippet(path=path, start_line=block_start, code=code))
            in_block = False
            block_start = 0
            block_lang = ''
            buf = []
            continue

        buf.append(line)

    return snippets


def check_no_todo_examples(files: list[Path]) -> list[str]:
    errors: list[str] = []
    todo_pattern = re.compile(r'\{\/\*\s*TODO:\s*add example\s*\*\/\}', re.IGNORECASE)
    for p in files:
        if 'bagel-api/' not in p.as_posix():
            continue
        txt = p.read_text()
        if todo_pattern.search(txt):
            errors.append(f'{p}: contains TODO example placeholder')
    return errors


def collect_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split('.', 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split('.', 1)[0])
    return roots


def validate_snippets(snippets: list[Snippet], import_check: bool) -> list[str]:
    errors: list[str] = []
    stdlib = set(getattr(sys, 'stdlib_module_names', set()))

    for snip in snippets:
        try:
            tree = ast.parse(snip.code)
        except SyntaxError as exc:
            errors.append(
                f'{snip.path}:{snip.start_line}: syntax error: {exc.msg} (line {exc.lineno}, col {exc.offset})'
            )
            continue

        if not import_check:
            continue

        for root in sorted(collect_import_roots(tree)):
            if root in stdlib or root in ALLOWED_THIRD_PARTY_IMPORTS:
                continue
            errors.append(
                f'{snip.path}:{snip.start_line}: import root not allowed by docs policy: {root}'
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate docs snippets and placeholders.')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root')
    parser.add_argument(
        '--import-check',
        action='store_true',
        help='Enable import resolution checks (recommended in CI).',
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = collect_files(root)

    todo_errors = check_no_todo_examples(files)
    snippets: list[Snippet] = []
    for p in files:
        snippets.extend(extract_python_snippets(p))

    snippet_errors = validate_snippets(snippets, import_check=args.import_check)
    errors = todo_errors + snippet_errors

    print(f'Checked {len(files)} docs files and {len(snippets)} python snippets.')
    if errors:
        print('\nValidation failed:')
        for e in errors:
            print(f'- {e}')
        return 1

    print('Docs validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
