# Mintlify documentation

## Working relationship
- You can push back on ideas-this can lead to better documentation. Cite sources and explain your reasoning when you do so
- ALWAYS ask for clarification rather than making assumptions
- NEVER lie, guess, or make up information

## Project context
- Format: MDX files with YAML frontmatter
- Config: docs.json for navigation, theme, settings
- Components: Mintlify components

## Content strategy
- Document just enough for user success - not too much, not too little
- Prioritize accuracy and usability of information
- Make content evergreen when possible
- Search for existing information before adding new content. Avoid duplication unless it is done for a strategic reason
- Check existing patterns for consistency
- Start by making the smallest reasonable changes

## Frontmatter requirements for pages
- title: Clear, descriptive page title
- description: Concise summary for SEO/navigation

## Writing standards
- Second-person voice ("you")
- Prerequisites at start of procedural content
- Test all code examples before publishing
- Match style and formatting of existing pages
- Include both basic and advanced use cases
- Language tags on all code blocks
- Alt text on all images
- Relative paths for internal links

## Git workflow
- NEVER use --no-verify when committing
- Ask how to handle uncommitted changes before starting
- Create a new branch when no clear branch exists for changes
- Commit frequently throughout development
- NEVER skip or disable pre-commit hooks

## Do not
- Skip frontmatter on any MDX file
- Use absolute URLs for internal links
- Include untested code examples
- Make assumptions - always ask for clarification

## Dev Server

- Uses Mintlify framework. Node 25+ is NOT supported.
- Run: `source ~/.nvm/nvm.sh && nvm use 20 && npx mintlify dev --port 3333`
- Valid themes: `mint`, `maple`, `palm`, `willow`, `linden`, `almond`, `aspen`, `sequoia`
- `docs.json` does NOT support a `background` key inside `colors` — only `primary`, `light`, `dark`.

## Configuration

- `docs.json` — main config: navigation tabs, theme, colors, logo, navbar, footer.
- Two tabs: "Guides" (existing content) and "BAGEL API" (Python API reference).

## API Reference Pages (`bagel-api/`)

- 49 class pages + 1 overview page = 50 total MDX files.
- Organized into subdirs: `core/`, `energies/`, `minimizer/`, `mutation/`, `oracles/`, `callbacks/`, `analysis/`.
- Generated from Python docstrings by `scripts/generate_api_docs.py`.

## Generator Script (`scripts/generate_api_docs.py`)

- Uses `ast` module to parse bagel source at `../bagel/src/bagel/` (no runtime imports).
- Supports both NumPy-style and Google-style docstrings.
- Generates MDX with Mintlify `<ResponseField>` components.
- `--new-only` flag: only creates pages for classes without an existing MDX file.
- Re-run to regenerate all: `python scripts/generate_api_docs.py`
- Re-run for new classes only: `python scripts/generate_api_docs.py --new-only`
- The REGISTRY list at the top maps each class to its source file, output path, and import path.

## MDX Gotchas

- Angle brackets in descriptions (e.g. `<self.foo>`) break MDX parsing — they're interpreted as JSX tags. The generator escapes these to backtick-wrapped code, but manually written content needs care.
- `<ResponseField>` is used for parameters (not `<ParamField>`, which is REST-specific).
- Code blocks inside MDX must not contain unescaped `<` outside of fenced blocks.

## Bagel Source Structure (`../bagel/src/bagel/`)

- Single-file modules: `chain.py`, `state.py`, `system.py`, `energies.py`, `minimizer.py`, `mutation.py`, `callbacks.py`
- Subdirectory modules: `oracles/` (with `folding/`, `embedding/`), `analysis/`
- Core hierarchy: System → State(s) → Chain(s) → Residue(s)
- Energy terms use Oracle pattern for predictions (ESMFold, ESM-2)
