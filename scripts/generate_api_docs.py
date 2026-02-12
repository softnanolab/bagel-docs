#!/usr/bin/env python3
"""
Generate Mintlify MDX documentation pages from bagel Python source files.

Uses the `ast` module to parse source files (no runtime imports needed) and
generates one .mdx file per class using Mintlify's ResponseField components.

Usage:
    python generate_api_docs.py                 # Generate all pages
    python generate_api_docs.py --new-only      # Only generate pages for new classes
"""

import ast
import argparse
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent
BAGEL_SRC = DOCS_DIR.parent / "bagel" / "src" / "bagel"
OUTPUT_DIR = DOCS_DIR / "bagel-api"

# ── Registry: maps class names to (source_file, output_subdir, output_filename, import_path) ──

REGISTRY: list[tuple[str, str, str, str, str]] = [
    # (class_name, source_file_relative, output_subdir, mdx_filename, import_path)
    # Core
    ("Residue", "chain.py", "core", "chain", "from bagel import Residue"),
    ("Chain", "chain.py", "core", "chain-chain", "from bagel import Chain"),
    ("State", "state.py", "core", "state", "from bagel import State"),
    ("System", "system.py", "core", "system", "from bagel import System"),
    # Energies
    ("EnergyTerm", "energies.py", "energies", "energy-term", "from bagel.energies import EnergyTerm"),
    ("PTMEnergy", "energies.py", "energies", "ptm-energy", "from bagel.energies import PTMEnergy"),
    ("ChemicalPotentialEnergy", "energies.py", "energies", "chemical-potential", "from bagel.energies import ChemicalPotentialEnergy"),
    ("PLDDTEnergy", "energies.py", "energies", "plddt-energy", "from bagel.energies import PLDDTEnergy"),
    ("OverallPLDDTEnergy", "energies.py", "energies", "overall-plddt", "from bagel.energies import OverallPLDDTEnergy"),
    ("SurfaceAreaEnergy", "energies.py", "energies", "surface-area", "from bagel.energies import SurfaceAreaEnergy"),
    ("HydrophobicEnergy", "energies.py", "energies", "hydrophobic", "from bagel.energies import HydrophobicEnergy"),
    ("PAEEnergy", "energies.py", "energies", "pae-energy", "from bagel.energies import PAEEnergy"),
    ("LISEnergy", "energies.py", "energies", "lis-energy", "from bagel.energies import LISEnergy"),
    ("RingSymmetryEnergy", "energies.py", "energies", "ring-symmetry", "from bagel.energies import RingSymmetryEnergy"),
    ("SeparationEnergy", "energies.py", "energies", "separation", "from bagel.energies import SeparationEnergy"),
    ("FlexEvoBindEnergy", "energies.py", "energies", "flex-evo-bind", "from bagel.energies import FlexEvoBindEnergy"),
    ("GlobularEnergy", "energies.py", "energies", "globular", "from bagel.energies import GlobularEnergy"),
    ("TemplateMatchEnergy", "energies.py", "energies", "template-match", "from bagel.energies import TemplateMatchEnergy"),
    ("SecondaryStructureEnergy", "energies.py", "energies", "secondary-structure", "from bagel.energies import SecondaryStructureEnergy"),
    ("EmbeddingsSimilarityEnergy", "energies.py", "energies", "embeddings-similarity", "from bagel.energies import EmbeddingsSimilarityEnergy"),
    # Minimizers
    ("Minimizer", "minimizer.py", "minimizer", "minimizer", "from bagel.minimizer import Minimizer"),
    ("MonteCarloMinimizer", "minimizer.py", "minimizer", "monte-carlo", "from bagel.minimizer import MonteCarloMinimizer"),
    ("SimulatedAnnealing", "minimizer.py", "minimizer", "simulated-annealing", "from bagel.minimizer import SimulatedAnnealing"),
    ("SimulatedTempering", "minimizer.py", "minimizer", "simulated-tempering", "from bagel.minimizer import SimulatedTempering"),
    # Mutation
    ("Mutation", "mutation.py", "mutation", "mutation", "from bagel.mutation import Mutation"),
    ("MutationRecord", "mutation.py", "mutation", "mutation-record", "from bagel.mutation import MutationRecord"),
    ("MutationProtocol", "mutation.py", "mutation", "mutation-protocol", "from bagel.mutation import MutationProtocol"),
    ("Canonical", "mutation.py", "mutation", "canonical", "from bagel.mutation import Canonical"),
    ("GrandCanonical", "mutation.py", "mutation", "grand-canonical", "from bagel.mutation import GrandCanonical"),
    # Oracles
    ("Oracle", "oracles/base.py", "oracles", "oracle", "from bagel.oracles import Oracle"),
    ("OracleResult", "oracles/base.py", "oracles", "oracle-result", "from bagel.oracles import OracleResult"),
    ("OraclesResultDict", "oracles/base.py", "oracles", "oracles-result-dict", "from bagel.oracles import OraclesResultDict"),
    ("FoldingOracle", "oracles/folding/base.py", "oracles", "folding-oracle", "from bagel.oracles.folding import FoldingOracle"),
    ("FoldingResult", "oracles/folding/base.py", "oracles", "folding-result", "from bagel.oracles.folding import FoldingResult"),
    ("ESMFold", "oracles/folding/esmfold.py", "oracles", "esmfold", "from bagel.oracles.folding import ESMFold"),
    ("ESMFoldResult", "oracles/folding/esmfold.py", "oracles", "esmfold-result", "from bagel.oracles.folding import ESMFoldResult"),
    ("EmbeddingOracle", "oracles/embedding/base.py", "oracles", "embedding-oracle", "from bagel.oracles.embedding import EmbeddingOracle"),
    ("EmbeddingResult", "oracles/embedding/base.py", "oracles", "embedding-result", "from bagel.oracles.embedding import EmbeddingResult"),
    ("ESM2", "oracles/embedding/esm2.py", "oracles", "esm2", "from bagel.oracles.embedding import ESM2"),
    ("ESM2Result", "oracles/embedding/esm2.py", "oracles", "esm2-result", "from bagel.oracles.embedding import ESM2Result"),
    # Callbacks
    ("Callback", "callbacks.py", "callbacks", "callback", "from bagel.callbacks import Callback"),
    ("CallbackContext", "callbacks.py", "callbacks", "callback-context", "from bagel.callbacks import CallbackContext"),
    ("CallbackManager", "callbacks.py", "callbacks", "callback-manager", "from bagel.callbacks import CallbackManager"),
    ("DefaultLogger", "callbacks.py", "callbacks", "default-logger", "from bagel.callbacks import DefaultLogger"),
    ("FoldingLogger", "callbacks.py", "callbacks", "folding-logger", "from bagel.callbacks import FoldingLogger"),
    ("EarlyStopping", "callbacks.py", "callbacks", "early-stopping", "from bagel.callbacks import EarlyStopping"),
    ("WandBLogger", "callbacks.py", "callbacks", "wandb-logger", "from bagel.callbacks import WandBLogger"),
    # Analysis
    ("MonteCarloAnalyzer", "analysis/analyzer.py", "analysis", "analyzer", "from bagel.analysis import MonteCarloAnalyzer"),
    ("SimulatedTemperingAnalyzer", "analysis/analyzer.py", "analysis", "simulated-tempering-analyzer", "from bagel.analysis import SimulatedTemperingAnalyzer"),
]


# ── Docstring Parsing ──


@dataclass
class ParamInfo:
    name: str
    type: str
    description: str
    required: bool = True
    default: str | None = None


@dataclass
class ReturnInfo:
    type: str
    description: str


@dataclass
class MethodInfo:
    name: str
    args: list[ParamInfo]
    returns: list[ReturnInfo]
    description: str
    decorators: list[str] = field(default_factory=list)
    return_annotation: str | None = None


@dataclass
class ClassInfo:
    name: str
    docstring: str
    summary: str
    extended_description: str
    bases: list[str]
    constructor_args: list[ParamInfo]
    attributes: list[ParamInfo]
    methods: list[MethodInfo]
    examples: str = ""
    is_dataclass: bool = False
    dataclass_fields: list[ParamInfo] = field(default_factory=list)


def parse_numpy_docstring(docstring: str) -> dict:
    """Parse a NumPy-style docstring into sections."""
    if not docstring:
        return {"summary": "", "description": "", "params": [], "returns": [], "attributes": [], "examples": ""}

    lines = textwrap.dedent(docstring).strip().split("\n")
    result: dict = {"summary": "", "description": "", "params": [], "returns": [], "attributes": [], "examples": ""}

    # Extract summary (first non-empty line)
    summary_lines = []
    i = 0
    while i < len(lines) and lines[i].strip():
        summary_lines.append(lines[i].strip())
        i += 1
    result["summary"] = " ".join(summary_lines)

    # Skip blank lines after summary
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Check if there's extended description before first section
    desc_lines = []
    while i < len(lines):
        line = lines[i]
        # Check if this is a section header (next line is dashes)
        if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
            break
        desc_lines.append(line)
        i += 1
    result["description"] = "\n".join(desc_lines).strip()

    # Parse sections
    while i < len(lines):
        line = lines[i].strip()
        # Section header detection
        if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
            section_name = line.lower()
            i += 2  # skip header and dashes

            if section_name in ("parameters", "params"):
                result["params"] = _parse_param_section(lines, i)
            elif section_name in ("returns", "return"):
                result["returns"] = _parse_return_section(lines, i)
            elif section_name == "attributes":
                result["attributes"] = _parse_param_section(lines, i)
            elif section_name == "examples":
                example_lines = []
                while i < len(lines):
                    if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
                        break
                    example_lines.append(lines[i])
                    i += 1
                result["examples"] = "\n".join(example_lines).strip()
                continue
            elif section_name == "raises":
                pass  # skip raises section

            # Skip to next section
            while i < len(lines):
                if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
                    break
                i += 1
        else:
            i += 1

    return result


def parse_google_docstring(docstring: str) -> dict:
    """Parse a Google-style docstring into sections."""
    if not docstring:
        return {"summary": "", "description": "", "params": [], "returns": [], "attributes": [], "examples": ""}

    lines = textwrap.dedent(docstring).strip().split("\n")
    result: dict = {"summary": "", "description": "", "params": [], "returns": [], "attributes": [], "examples": ""}

    # Extract summary
    summary_lines = []
    i = 0
    while i < len(lines) and lines[i].strip():
        summary_lines.append(lines[i].strip())
        i += 1
    result["summary"] = " ".join(summary_lines)

    # Skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Check for extended description before first section
    desc_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^(Args|Arguments|Attributes|Returns|Raises|Examples|Notes):", line):
            break
        desc_lines.append(lines[i])
        i += 1
    result["description"] = "\n".join(desc_lines).strip()

    # Parse Google-style sections
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^(Args|Arguments|Parameters):", line):
            i += 1
            result["params"] = _parse_google_param_section(lines, i)
        elif re.match(r"^Attributes:", line):
            i += 1
            result["attributes"] = _parse_google_param_section(lines, i)
        elif re.match(r"^Returns:", line):
            i += 1
            # Simple return parsing
        elif re.match(r"^Examples:", line):
            i += 1
            example_lines = []
            while i < len(lines):
                stripped = lines[i].strip()
                if re.match(r"^(Args|Arguments|Parameters|Attributes|Returns|Raises|Notes):", stripped):
                    break
                example_lines.append(lines[i])
                i += 1
            result["examples"] = "\n".join(example_lines).strip()
            continue
        else:
            i += 1

        # Advance past section content
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r"^(Args|Arguments|Parameters|Attributes|Returns|Raises|Examples|Notes):", stripped):
                break
            i += 1

    return result


def _parse_param_section(lines: list[str], start: int) -> list[ParamInfo]:
    """Parse a NumPy-style parameter section."""
    params = []
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if we've hit a new section
        if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
            break
        if not stripped:
            i += 1
            continue

        # Parameter line: "name : type" or "name: type"
        param_match = re.match(r"^(\w+)\s*:\s*(.+)$", stripped)
        if param_match:
            name = param_match.group(1)
            type_str = param_match.group(2).strip()

            # Check for default value in type string
            default = None
            required = True
            default_match = re.match(r"(.+?),?\s*(?:default\s*[=:]\s*(.+)|optional)$", type_str, re.IGNORECASE)
            if default_match:
                type_str = default_match.group(1).strip().rstrip(",")
                default = default_match.group(2)
                if default:
                    default = default.strip()
                required = False

            # Check for "= value" in type annotation
            eq_match = re.match(r"(.+?)\s*=\s*(.+)$", type_str)
            if eq_match:
                type_str = eq_match.group(1).strip()
                default = eq_match.group(2).strip()
                required = False

            # Collect description lines
            desc_lines = []
            i += 1
            while i < len(lines):
                desc_line = lines[i]
                desc_stripped = desc_line.strip()
                if not desc_stripped:
                    i += 1
                    break
                # Check if next line is a new param
                if re.match(r"^\w+\s*:", desc_stripped) and not desc_stripped.startswith(" "):
                    # Could be a new param - check indentation
                    if len(desc_line) - len(desc_line.lstrip()) <= 4:
                        break
                if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
                    break
                desc_lines.append(desc_stripped)
                i += 1

            description = " ".join(desc_lines)
            params.append(ParamInfo(
                name=name,
                type=_clean_type(type_str),
                description=description,
                required=required,
                default=default,
            ))
        else:
            i += 1

    return params


def _parse_return_section(lines: list[str], start: int) -> list[ReturnInfo]:
    """Parse a NumPy-style returns section."""
    returns = []
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
            break
        if not stripped:
            i += 1
            continue

        # Return line: "name : type" or just "type"
        ret_match = re.match(r"^(?:(\w+)\s*:\s*)?(.+)$", stripped)
        if ret_match:
            type_str = ret_match.group(2).strip() if ret_match.group(2) else ret_match.group(1)

            desc_lines = []
            i += 1
            while i < len(lines):
                desc_line = lines[i]
                desc_stripped = desc_line.strip()
                if not desc_stripped:
                    i += 1
                    continue
                if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
                    break
                # Check if it's a new return entry (less indented)
                if re.match(r"^\w+\s*:", desc_stripped):
                    indent = len(desc_line) - len(desc_line.lstrip())
                    if indent <= 4:
                        break
                desc_lines.append(desc_stripped)
                i += 1

            description = " ".join(desc_lines)
            returns.append(ReturnInfo(type=_clean_type(type_str), description=description))
        else:
            i += 1

    return returns


def _parse_google_param_section(lines: list[str], start: int) -> list[ParamInfo]:
    """Parse a Google-style parameter section."""
    params = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if re.match(r"^(Args|Arguments|Parameters|Attributes|Returns|Raises|Examples|Notes):", stripped):
            break

        # Google style: "name (type): description" or "name: description"
        match = re.match(r"^(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$", stripped)
        if match:
            name = match.group(1)
            type_str = match.group(2) or ""
            desc = match.group(3)

            # Collect continuation lines
            i += 1
            while i < len(lines):
                cont = lines[i]
                if not cont.strip() or re.match(r"^\s{0,4}\w", cont):
                    break
                desc += " " + cont.strip()
                i += 1

            params.append(ParamInfo(name=name, type=_clean_type(type_str), description=desc.strip(), required=True))
        else:
            i += 1

    return params


def _clean_type(type_str: str) -> str:
    """Clean up type annotations for display."""
    # Remove :class:`...` RST references
    type_str = re.sub(r":class:`~?\.?([^`]+)`", r"\1", type_str)
    # Remove leading/trailing whitespace
    type_str = type_str.strip()
    # Simplify common patterns
    type_str = type_str.replace("typing.", "")
    return type_str


def _clean_rst(text: str) -> str:
    """Remove RST cross-references from text for MDX display."""
    text = re.sub(r":class:`~?\.?([^`]+)`", r"`\1`", text)
    text = re.sub(r":meth:`~?\.?([^`]+)`", r"`\1`", text)
    text = re.sub(r":func:`~?\.?([^`]+)`", r"`\1`", text)
    text = re.sub(r":attr:`~?\.?([^`]+)`", r"`\1`", text)
    text = re.sub(r":mod:`~?\.?([^`]+)`", r"`\1`", text)
    # Clean up .. math:: blocks to inline
    text = re.sub(r"\.\. math::\s*\n\s*", r"$", text)
    # Escape bare angle brackets that aren't MDX components (would break MDX parsing)
    # Match <word> patterns that aren't known MDX components
    text = re.sub(r"<(?!ResponseField|Expandable|CardGroup|Card|/)([\w.]+)>", r"`\1`", text)
    return text


# ── AST Parsing ──


def get_annotation_str(node: ast.expr | None) -> str:
    """Convert an AST annotation node to a string representation."""
    if node is None:
        return ""
    return ast.unparse(node)


def extract_class_info(tree: ast.Module, class_name: str) -> ClassInfo | None:
    """Extract class information from an AST tree."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue

        # Get bases
        bases = [ast.unparse(base) for base in node.bases]

        # Check if it's a dataclass
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            for d in node.decorator_list
        )

        # Get class docstring
        docstring = ast.get_docstring(node) or ""

        # Detect docstring style and parse
        if "Parameters\n" in docstring and "----------" in docstring:
            parsed = parse_numpy_docstring(docstring)
        elif "Args:" in docstring or "Attributes:" in docstring:
            parsed = parse_google_docstring(docstring)
        else:
            parsed = parse_numpy_docstring(docstring)  # default

        summary = parsed["summary"]
        extended_description = parsed["description"]
        examples = parsed.get("examples", "")

        # Extract dataclass fields
        dataclass_fields = []
        if is_dataclass:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fname = item.target.id
                    ftype = get_annotation_str(item.annotation)

                    # Check if it has a default
                    has_default = item.value is not None
                    default_val = None
                    if has_default:
                        default_val = ast.unparse(item.value)

                    # Check if it's init=False (from field())
                    is_init = True
                    if isinstance(item.value, ast.Call):
                        func = item.value
                        if isinstance(func.func, ast.Name) and func.func.id == "field":
                            for kw in func.keywords:
                                if kw.arg == "init" and isinstance(kw.value, ast.Constant):
                                    is_init = kw.value.value
                                if kw.arg == "default":
                                    default_val = ast.unparse(kw.value)
                                if kw.arg == "default_factory":
                                    default_val = f"{ast.unparse(kw.value)}()"

                    if is_init and not fname.startswith("_"):
                        # Look up description from docstring attributes or params
                        desc = ""
                        for attr in parsed.get("attributes", []):
                            if attr.name == fname:
                                desc = attr.description
                                break
                        if not desc:
                            for param in parsed.get("params", []):
                                if param.name == fname:
                                    desc = param.description
                                    break

                        dataclass_fields.append(ParamInfo(
                            name=fname,
                            type=_clean_type(ftype),
                            description=desc,
                            required=not has_default,
                            default=default_val,
                        ))

        # Extract __init__ parameters
        constructor_args = []
        init_method = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break

        if init_method:
            init_docstring = ast.get_docstring(init_method) or ""
            if "Parameters\n" in init_docstring and "----------" in init_docstring:
                init_parsed = parse_numpy_docstring(init_docstring)
            elif "Args:" in init_docstring:
                init_parsed = parse_google_docstring(init_docstring)
            else:
                init_parsed = parse_numpy_docstring(init_docstring)

            doc_params = {p.name: p for p in init_parsed.get("params", [])}

            for arg in init_method.args.args:
                if arg.arg == "self":
                    continue
                name = arg.arg
                type_str = get_annotation_str(arg.annotation)

                # Find default value
                # defaults are right-aligned to args (excluding self)
                non_self_args = [a for a in init_method.args.args if a.arg != "self"]
                arg_idx = non_self_args.index(arg)
                defaults = init_method.args.defaults
                default_offset = len(non_self_args) - len(defaults)
                default_val = None
                required = True
                if arg_idx >= default_offset:
                    default_node = defaults[arg_idx - default_offset]
                    default_val = ast.unparse(default_node)
                    required = False

                # Get description from docstring
                desc = ""
                if name in doc_params:
                    desc = doc_params[name].description
                    if doc_params[name].type and not type_str:
                        type_str = doc_params[name].type

                constructor_args.append(ParamInfo(
                    name=name,
                    type=_clean_type(type_str),
                    description=desc,
                    required=required,
                    default=default_val,
                ))

            # Also handle **kwargs
            if init_method.args.kwarg:
                constructor_args.append(ParamInfo(
                    name=f"**{init_method.args.kwarg.arg}",
                    type=get_annotation_str(init_method.args.kwarg.annotation) or "Any",
                    description="Additional keyword arguments.",
                    required=False,
                ))

        # Extract class-level attributes from docstring (filter out private)
        attributes = [a for a in parsed.get("attributes", []) if not a.name.startswith("_")]
        # For non-dataclass, also look at class body for annotated assignments
        if not is_dataclass:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fname = item.target.id
                    if not fname.startswith("_"):
                        ftype = get_annotation_str(item.annotation)
                        default_val = ast.unparse(item.value) if item.value else None
                        # Check if already in attributes from docstring
                        existing = [a for a in attributes if a.name == fname]
                        if not existing:
                            attributes.append(ParamInfo(
                                name=fname,
                                type=_clean_type(ftype),
                                description="",
                                required=default_val is None,
                                default=default_val,
                            ))

        # Merge class-level docstring params into constructor params when init params lack descriptions
        class_doc_params = {p.name: p for p in parsed.get("params", [])}
        for arg in constructor_args:
            if not arg.description and arg.name in class_doc_params:
                arg.description = class_doc_params[arg.name].description
                if not arg.type and class_doc_params[arg.name].type:
                    arg.type = class_doc_params[arg.name].type

        # Extract methods
        methods = []
        skip_methods = {"__init__", "__post_init__", "__copy__", "__deepcopy__",
                        "__repr__", "__str__", "__eq__", "__hash__",
                        "__len__", "__getattr__", "__dir__", "__del__"}
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name not in skip_methods:
                if item.name.startswith("_"):
                    continue

                decorators = []
                for d in item.decorator_list:
                    if isinstance(d, ast.Name):
                        decorators.append(d.id)
                    elif isinstance(d, ast.Attribute):
                        decorators.append(d.attr)

                method_docstring = ast.get_docstring(item) or ""
                if "Parameters\n" in method_docstring and "----------" in method_docstring:
                    method_parsed = parse_numpy_docstring(method_docstring)
                elif "Args:" in method_docstring:
                    method_parsed = parse_google_docstring(method_docstring)
                else:
                    method_parsed = {"summary": method_docstring.split("\n")[0] if method_docstring else "",
                                     "description": "", "params": [], "returns": []}

                method_doc_params = {p.name: p for p in method_parsed.get("params", [])}

                method_args = []
                for arg in item.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    name = arg.arg
                    type_str = get_annotation_str(arg.annotation)

                    non_self_args = [a for a in item.args.args if a.arg not in ("self", "cls")]
                    arg_idx = non_self_args.index(arg)
                    defaults = item.args.defaults
                    default_offset = len(non_self_args) - len(defaults)
                    default_val = None
                    required = True
                    if arg_idx >= default_offset:
                        default_node = defaults[arg_idx - default_offset]
                        default_val = ast.unparse(default_node)
                        required = False

                    desc = ""
                    if name in method_doc_params:
                        desc = method_doc_params[name].description
                        if method_doc_params[name].type and not type_str:
                            type_str = method_doc_params[name].type

                    method_args.append(ParamInfo(
                        name=name,
                        type=_clean_type(type_str),
                        description=desc,
                        required=required,
                        default=default_val,
                    ))

                return_annotation = get_annotation_str(item.returns)
                returns = method_parsed.get("returns", [])

                desc = method_parsed.get("summary", "")
                if method_parsed.get("description"):
                    desc = desc + "\n\n" + method_parsed["description"] if desc else method_parsed["description"]

                methods.append(MethodInfo(
                    name=item.name,
                    args=method_args,
                    returns=returns,
                    description=desc.strip(),
                    decorators=decorators,
                    return_annotation=return_annotation,
                ))

        return ClassInfo(
            name=class_name,
            docstring=docstring,
            summary=summary,
            extended_description=extended_description,
            bases=bases,
            constructor_args=constructor_args,
            attributes=attributes,
            methods=methods,
            examples=examples,
            is_dataclass=is_dataclass,
            dataclass_fields=dataclass_fields,
        )

    return None


# ── MDX Generation ──


def escape_mdx(text: str) -> str:
    """Escape characters that could break MDX rendering."""
    # Escape angle brackets that aren't part of tags
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # But restore common patterns
    text = text.replace("&lt;ResponseField", "<ResponseField")
    text = text.replace("&lt;/ResponseField", "</ResponseField")
    text = text.replace("&lt;Expandable", "<Expandable")
    text = text.replace("&lt;/Expandable", "</Expandable")
    return text


def generate_mdx(info: ClassInfo, import_path: str) -> str:
    """Generate an MDX page for a class."""
    parts = []

    # Clean RST references from summary and description
    summary = _clean_rst(info.summary)
    extended_description = _clean_rst(info.extended_description)

    # Frontmatter
    summary_escaped = summary.replace('"', '\\"')
    parts.append(f'---\ntitle: "{info.name}"\ndescription: "{summary_escaped}"\n---\n')

    # Extended description (body text, equations, etc.)
    if extended_description:
        parts.append(f"\n{extended_description}\n")

    # Parameters (from dataclass fields or constructor args) — no signature code block
    if info.is_dataclass and info.dataclass_fields:
        parts.append("\n## Parameters\n")
        for f in info.dataclass_fields:
            req = " required" if f.required else ""
            default = f' default="{f.default}"' if f.default else ""
            cleaned_desc = _clean_rst(f.description)
            desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
            parts.append(
                f'\n<ResponseField name="{f.name}" type="{f.type}"{req}{default}>'
                f"{desc}"
                f"</ResponseField>\n"
            )

    elif info.constructor_args:
        parts.append("\n## Parameters\n")
        for a in info.constructor_args:
            req = " required" if a.required else ""
            default = f' default="{a.default}"' if a.default and not a.required else ""
            cleaned_desc = _clean_rst(a.description)
            desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
            parts.append(
                f'\n<ResponseField name="{a.name}" type="{a.type}"{req}{default}>'
                f"{desc}"
                f"</ResponseField>\n"
            )

    # Attributes (from docstring, excluding dataclass fields)
    dc_field_names = {f.name for f in info.dataclass_fields}
    non_dc_attrs = [a for a in info.attributes if a.name not in dc_field_names]
    if non_dc_attrs:
        parts.append("\n## Attributes\n")
        for attr in non_dc_attrs:
            cleaned_desc = _clean_rst(attr.description)
            desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
            parts.append(
                f'\n<ResponseField name="{attr.name}" type="{attr.type}">'
                f"{desc}"
                f"</ResponseField>\n"
            )

    # Methods — no signature code blocks
    if info.methods:
        parts.append("\n## Methods\n")
        for method in info.methods:
            parts.append(f"\n### {method.name}\n")
            if method.description:
                parts.append(f"\n{_clean_rst(method.description)}\n")

            if "property" in method.decorators:
                # For properties, just show returns if available
                if method.returns:
                    parts.append("\n**Returns**\n")
                    for r in method.returns:
                        cleaned_desc = _clean_rst(r.description)
                        desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
                        parts.append(
                            f'\n<ResponseField name="return" type="{r.type}">'
                            f"{desc}"
                            f"</ResponseField>\n"
                        )
                continue

            if method.args:
                parts.append("\n**Parameters**\n")
                for a in method.args:
                    req = " required" if a.required else ""
                    default = f' default="{a.default}"' if a.default and not a.required else ""
                    cleaned_desc = _clean_rst(a.description)
                    desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
                    parts.append(
                        f'\n<ResponseField name="{a.name}" type="{a.type}"{req}{default}>'
                        f"{desc}"
                        f"</ResponseField>\n"
                    )

            if method.returns:
                parts.append("\n**Returns**\n")
                for r in method.returns:
                    cleaned_desc = _clean_rst(r.description)
                    desc = f"\n  {cleaned_desc}\n" if cleaned_desc else "\n"
                    parts.append(
                        f'\n<ResponseField name="return" type="{r.type}">'
                        f"{desc}"
                        f"</ResponseField>\n"
                    )

    # Example section
    parts.append("\n## Example\n")
    if info.examples:
        example_text = _clean_rst(info.examples)
        # Strip doctest >>> and ... prefixes if present
        example_lines = example_text.split("\n")
        cleaned_lines = []
        for line in example_lines:
            stripped = line.strip()
            if stripped.startswith(">>> "):
                cleaned_lines.append(stripped[4:])
            elif stripped.startswith("..."):
                cleaned_lines.append(stripped[4:] if stripped.startswith("... ") else stripped[3:])
            else:
                cleaned_lines.append(line)
        example_code = "\n".join(cleaned_lines).strip()
        parts.append(f"\n```python\n{example_code}\n```\n")
    else:
        parts.append('\n```python\n{/* TODO: add example */}\n```\n')

    return "".join(parts)


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="Generate Mintlify MDX docs from bagel source.")
    parser.add_argument("--new-only", action="store_true",
                        help="Only generate pages for classes that don't have an MDX file yet.")
    args = parser.parse_args()

    # Parse all needed source files
    parsed_files: dict[str, ast.Module] = {}

    generated = 0
    skipped = 0
    errors = 0

    for class_name, source_rel, subdir, mdx_name, import_path in REGISTRY:
        output_path = OUTPUT_DIR / subdir / f"{mdx_name}.mdx"

        if args.new_only and output_path.exists():
            skipped += 1
            continue

        source_path = BAGEL_SRC / source_rel

        if not source_path.exists():
            print(f"  WARNING: Source file not found: {source_path}")
            errors += 1
            continue

        # Parse source file (cache)
        key = str(source_path)
        if key not in parsed_files:
            with open(source_path) as f:
                parsed_files[key] = ast.parse(f.read())

        tree = parsed_files[key]
        info = extract_class_info(tree, class_name)

        if info is None:
            print(f"  WARNING: Class {class_name} not found in {source_path}")
            errors += 1
            continue

        mdx_content = generate_mdx(info, import_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mdx_content)
        generated += 1
        print(f"  Generated: {output_path.relative_to(DOCS_DIR)}")

    print(f"\nDone: {generated} generated, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
