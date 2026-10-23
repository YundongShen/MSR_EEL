"""Extract requirement, test, and source units across programming languages."""

from __future__ import annotations

import json
import re
from pathlib import Path

from data_pipeline.parse_instances import _extract_code_units, _split_file_diffs, apply_test_patch

import tree_sitter_languages as tsl

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".php": "php", ".rb": "ruby", ".cs": "c_sharp", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
}

FUNC_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "tsx": {"function_declaration", "method_definition", "generator_function_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "php": {"function_definition", "method_declaration"},
    "ruby": {"method", "singleton_method"},
    "c_sharp": {"method_declaration", "constructor_declaration", "local_function_statement"},
    "kotlin": {"function_declaration"},
    "scala": {"function_definition"},
}
CLASS_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "abstract_class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "abstract_class_declaration", "interface_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
    "go": {"type_declaration"},
    "rust": {"impl_item", "struct_item", "enum_item", "trait_item"},
    "c": {"struct_specifier"},
    "cpp": {"class_specifier", "struct_specifier"},
    "php": {"class_declaration", "interface_declaration", "trait_declaration"},
    "ruby": {"class", "module"},
    "c_sharp": {"class_declaration", "interface_declaration", "struct_declaration", "record_declaration"},
    "kotlin": {"class_declaration", "object_declaration"},
    "scala": {"class_definition", "object_definition", "trait_definition"},
}
SPEC_CALLEES = {"it", "test", "specify", "scenario", "should", "example", "bench"}
_parsers: dict[str, object] = {}


def language_of(path: str) -> str | None:
    return EXT_LANG.get(Path(path).suffix.lower())


def _parser(lang: str):
    if lang not in _parsers:
        _parsers[lang] = tsl.get_parser(lang)
    return _parsers[lang]


def _text(node, data: bytes) -> str:
    return data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name(node, data: bytes) -> str:
    target = node.child_by_field_name("name")
    if target is None:
        decl = node.child_by_field_name("declarator")
        while decl is not None and decl.type not in ("identifier", "field_identifier", "qualified_identifier",
                                                     "destructor_name", "operator_name", "type_identifier"):
            nxt = decl.child_by_field_name("declarator")
            if nxt is None:
                break
            decl = nxt
        target = decl
    if target is None:
        target = node.child_by_field_name("type")
    if target is None:
        for child in node.children:
            if "identifier" in child.type or child.type == "constant":
                target = child
                break
    return _text(target, data).strip()[:120] if target is not None else ""


def _walk(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def _named_function_value(node, lang: str, data: bytes):
    """const f = (...) => {...} / const f = function (...) {...}  (JS / TS)."""
    if lang not in ("javascript", "typescript", "tsx") or node.type != "variable_declarator":
        return None
    value = node.child_by_field_name("value")
    if value is not None and value.type in ("arrow_function", "function_expression", "function"):
        return _name(node, data)
    return None


def code_units(filepath: str, source: str) -> list[dict]:
    """Functions / methods / classes of one file (the fields of parse_instances._extract_code_units)."""
    lang = language_of(filepath)
    if lang is None:
        return []
    if lang == "python":
        return _extract_code_units(filepath, source)
    data = source.encode("utf-8", errors="replace")
    try:
        tree = _parser(lang).parse(data)
    except Exception:
        return []
    units = []
    for node in _walk(tree.root_node):
        kind = name = None
        if node.type in FUNC_TYPES[lang]:
            kind, name = "function", _name(node, data)
        elif node.type in CLASS_TYPES[lang]:
            if node.type in ("struct_specifier", "class_specifier") and node.child_by_field_name("body") is None:
                continue
            kind, name = "class", _name(node, data)
        else:
            arrow = _named_function_value(node, lang, data)
            if arrow is not None:
                kind, name = "function", arrow
        if kind is None or node.end_point[0] == node.start_point[0]:
            continue
        units.append({
            "filepath": filepath, "name": name, "kind": kind,
            "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
            "code": _text(node, data),
        })
    return units


def _spec_blocks(lang: str, root, data: bytes) -> list[tuple[int, int, str, str]]:
    """it()/test()-style blocks: (start_line, end_line, name, code)."""
    out = []
    for node in _walk(root):
        callee = None
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None:
                callee = _text(fn, data).split(".")[0].split("(")[0]
        elif node.type == "call" and lang == "ruby":
            fn = node.child_by_field_name("method")
            if fn is not None:
                callee = _text(fn, data)
        elif node.type in ("function_call_expression",) and lang == "php":
            fn = node.child_by_field_name("function")
            if fn is not None:
                callee = _text(fn, data)
        if callee in SPEC_CALLEES and node.end_point[0] > node.start_point[0]:
            args = node.child_by_field_name("arguments")
            label = ""
            if args is not None and args.named_child_count:
                label = _text(args.named_children[0], data).strip("'\"` ")[:120]
            out.append((node.start_point[0] + 1, node.end_point[0] + 1, label or callee, _text(node, data)))
    return out


def added_lines(file_diff: str) -> tuple[set[int], list[str]]:
    """New-side line numbers the diff adds, and the new side of each hunk."""
    added: set[int] = set()
    blocks: list[str] = []
    new_no, cur = None, None
    for line in file_diff.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            new_no = int(m.group(1))
            cur = []
            blocks.append(cur)
            continue
        if new_no is None or line.startswith("\\"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.add(new_no)
            cur.append(line[1:])
            new_no += 1
        elif line.startswith("-"):
            continue
        else:
            cur.append(line[1:] if line.startswith(" ") else line)
            new_no += 1
    return added, ["\n".join(b) for b in blocks if any(x.strip() for x in b)]


def test_units(test_files: dict[str, str], test_patch: str) -> list[dict]:
    """Test units that contain a line added by the test patch (post-patch files)."""
    post = apply_test_patch(test_files, test_patch)
    results = []
    for d in _split_file_diffs(test_patch or ""):
        if d["deleted"]:
            continue
        path = d["new"]
        added, new_blocks = added_lines(d["text"])
        if not added:
            continue
        found: list[tuple[int, int, str, str]] = []
        source = post.get(path)
        lang = language_of(path)
        patched = source is not None and (d["added"] or source != test_files.get(path))
        if patched and lang is not None:
            data = source.encode("utf-8", errors="replace")
            try:
                root = _parser(lang).parse(data).root_node
            except Exception:
                root = None
            if root is not None:
                spans = [(n.start_point[0] + 1, n.end_point[0] + 1, _name(n, data), _text(n, data))
                         for n in _walk(root) if n.type in FUNC_TYPES[lang] and n.end_point[0] > n.start_point[0]]
                spans += _spec_blocks(lang, root, data)
                hit = [s for s in spans if any(s[0] <= ln <= s[1] for ln in added)]
                found = [s for s in hit if not any(o is not s and s[0] <= o[0] and o[1] <= s[1] and (o[0], o[1]) != (s[0], s[1])
                                                   for o in hit)]
        seen = set()
        for start, _end, name, code in sorted(found):
            if (start, name) in seen:
                continue
            seen.add((start, name))
            results.append({"filepath": path, "function_name": name, "code": code})
        if not found:
            for i, block in enumerate(new_blocks):
                results.append({"filepath": path, "function_name": f"hunk_{i}", "code": block})
    return results


def process_instance(inst: dict) -> dict:
    source_units = [u for path, src in inst.get("source_files", {}).items() for u in code_units(path, src)]
    tests = test_units(inst.get("test_files", {}), inst.get("test_patch", ""))
    ids = inst.get("fail_to_pass_ids", inst.get("fail_to_pass", inst.get("FAIL_TO_PASS")))
    ids = json.loads(ids) if isinstance(ids, str) else list(ids or [])
    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "requirement": inst["problem_statement"],
        "fail_to_pass_ids": ids,
        "source_units": source_units,
        "test_functions": tests,
        "test_patch": inst.get("test_patch", ""),
    }
