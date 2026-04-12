from __future__ import annotations

import ast

_HTTP_METHOD_ATTRS = frozenset(
    {
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "options",
        "head",
        "route",
        "websocket",
        "api_route",
    }
)


def _route_from_decorator(dec: ast.expr) -> dict | None:
    if not isinstance(dec, ast.Call):
        return None
    fn = dec.func
    if not isinstance(fn, ast.Attribute):
        return None
    attr = fn.attr.lower()
    if attr not in _HTTP_METHOD_ATTRS:
        return None
    path: str | None = None
    if dec.args:
        a0 = dec.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            path = a0.value
    line = getattr(dec, "lineno", 0) or 0
    return {"method": fn.attr.upper(), "path": path, "line": line}


def scan_module(source: str, path: str) -> dict:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return {
            "path": path,
            "ok": False,
            "syntax_error": e.msg,
            "lineno": e.lineno,
            "offset": e.offset,
        }

    routes: list[dict] = []
    top_functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            top_functions.append(node.name)
            for dec in node.decorator_list:
                hint = _route_from_decorator(dec)
                if hint:
                    routes.append({**hint, "function": node.name})
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    for dec in child.decorator_list:
                        hint = _route_from_decorator(dec)
                        if hint:
                            routes.append(
                                {
                                    **hint,
                                    "function": f"{node.name}.{child.name}",
                                }
                            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)

    line_count = source.count("\n") + (1 if source else 0)
    return {
        "path": path,
        "ok": True,
        "lines": line_count,
        "routes": routes,
        "top_level_functions": top_functions,
        "classes": classes,
        "imports": sorted(set(imports)),
    }


def scan_codebase(files: dict[str, str]) -> dict:
    per_file: list[dict] = []
    routes_all: list[dict] = []
    imports_merged: set[str] = set()
    ok_n = 0
    for path in sorted(files.keys()):
        src = files[path]
        row = scan_module(src, path)
        per_file.append(row)
        if row.get("ok"):
            ok_n += 1
            routes_all.extend(
                {**r, "file": path} for r in row.get("routes", [])
            )
            imports_merged.update(row.get("imports", []))
    return {
        "files": per_file,
        "file_count": len(per_file),
        "parsed_ok_count": ok_n,
        "routes": routes_all,
        "imports": sorted(imports_merged),
    }


def merge_files_for_prompt(files: dict[str, str]) -> str:
    parts: list[str] = []
    for path in sorted(files.keys()):
        parts.append(f"# --- {path} ---\n")
        parts.append(files[path].strip())
        parts.append("\n\n")
    return "".join(parts).strip()


def format_scan_for_prompt(scan: dict) -> str:
    if "files" in scan:
        lines: list[str] = [
            f"Multi-file scan: {scan.get('file_count', 0)} path(s), "
            f"{scan.get('parsed_ok_count', 0)} parsed OK.",
            "",
        ]
        for f in scan.get("files", []):
            if not f.get("ok"):
                lines.append(
                    f"- {f.get('path')}: SYNTAX ERROR: {f.get('syntax_error')} "
                    f"(line {f.get('lineno')})"
                )
                continue
            lines.append(f"- {f.get('path')}: {f.get('lines')} lines")
            if f.get("routes"):
                for r in f["routes"]:
                    p = r.get("path") or "?"
                    lines.append(
                        f"    route {r.get('method')} {p!r} → {r.get('function')} (line {r.get('line')})"
                    )
            if f.get("top_level_functions"):
                lines.append(f"    defs: {', '.join(f['top_level_functions'])}")
            if f.get("classes"):
                lines.append(f"    classes: {', '.join(f['classes'])}")
        if scan.get("routes"):
            lines.append("")
            lines.append("All detected HTTP-style routes (heuristic):")
            for r in scan["routes"]:
                p = r.get("path") or "?"
                lines.append(
                    f"  - [{r.get('file')}] {r.get('method')} {p!r} → {r.get('function')}"
                )
        im = scan.get("imports") or []
        if im:
            lines.append("")
            lines.append(f"Imports (merged, {len(im)}): {', '.join(im[:80])}")
            if len(im) > 80:
                lines.append(f"  ... and {len(im) - 80} more")
        return "\n".join(lines)

    if not scan.get("ok"):
        return (
            f"Syntax error in {scan.get('path')}: {scan.get('syntax_error')} "
            f"(line {scan.get('lineno')})"
        )
    lines = [
        f"File {scan.get('path')}: {scan.get('lines')} lines",
    ]
    for r in scan.get("routes", []):
        p = r.get("path") or "?"
        lines.append(
            f"  route {r.get('method')} {p!r} → {r.get('function')} (line {r.get('line')})"
        )
    if scan.get("top_level_functions"):
        lines.append(f"Top-level functions: {', '.join(scan['top_level_functions'])}")
    if scan.get("classes"):
        lines.append(f"Classes: {', '.join(scan['classes'])}")
    im = scan.get("imports") or []
    if im:
        lines.append(f"Imports ({len(im)}): {', '.join(im[:60])}")
        if len(im) > 60:
            lines.append(f"  ... and {len(im) - 60} more")
    return "\n".join(lines)
