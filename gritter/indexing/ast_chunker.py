from __future__ import annotations
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
import tree_sitter_rust as tsrust

from gritter.models.chunk import CodeChunk
from gritter.utils.tokens import count_tokens, split_at_token_boundary

_PARSERS: dict[str, Parser] = {
    "python": Parser(Language(tspython.language())),
    "typescript": Parser(Language(tsts.language_typescript())),
    "rust": Parser(Language(tsrust.language())),
}

_TOP_LEVEL_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "macro_rules",
        "type_item",
        "const_item",
    },
}

_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "typescript": {"import_statement"},
    "rust": {"use_declaration"},
}

_METHOD_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "typescript": {"method_definition"},
    "rust": {"function_item"},
}

# Body container node types — the child of a class/impl that holds the methods
_BODY_TYPES: dict[str, set[str]] = {
    "python": {"block"},
    "typescript": {"class_body"},
    "rust": {"declaration_list"},
}


def chunk_file_ast(
    content: str,
    file_path: str,
    language: str,
    min_tokens: int = 50,
    max_tokens: int = 512,
    overlap_tokens: int = 20,
) -> list[CodeChunk]:
    """Parse content with tree-sitter and extract top-level definitions as chunks.

    Raises ValueError if the file has syntax errors.
    """
    source = content.encode("utf-8")
    parser = _PARSERS[language]
    tree = parser.parse(source)

    if tree.root_node.has_error:
        raise ValueError(f"Parse error in {file_path}")

    imports = _extract_imports(tree.root_node, source, language)
    top_level_types = _TOP_LEVEL_TYPES[language]

    raw_chunks: list[CodeChunk] = []
    for node in tree.root_node.children:
        if node.type not in top_level_types:
            continue

        node_text = source[node.start_byte:node.end_byte].decode("utf-8")
        symbol_name, symbol_type = _get_symbol(node, source, language)
        tokens = count_tokens(node_text)

        if tokens > max_tokens:
            sub = _split_large_node(
                node, source, file_path, language, imports,
                min_tokens, max_tokens, overlap_tokens,
            )
            raw_chunks.extend(sub)
        else:
            raw_chunks.append(CodeChunk(
                content=node_text,
                file_path=file_path,
                language=language,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                imports=imports,
            ))

    return _merge_small_chunks(raw_chunks, min_tokens)


def _get_symbol(node: Node, source: bytes, language: str) -> tuple[str | None, str | None]:
    if language == "python":
        return _get_python_symbol(node, source)
    elif language == "typescript":
        return _get_typescript_symbol(node, source)
    elif language == "rust":
        return _get_rust_symbol(node, source)
    return (None, "module")


def _get_python_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    if node.type == "function_definition":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "function")
    elif node.type == "class_definition":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "class")
    elif node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_python_symbol(child, source)
    return (None, "module")


def _get_typescript_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    if node.type in ("function_declaration", "generator_function_declaration"):
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "function")
    elif node.type == "class_declaration":
        name = node.child_by_field_name("name")
        return (_node_text(name, source), "class")
    elif node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "lexical_declaration",
                "generator_function_declaration",
            ):
                return _get_typescript_symbol(child, source)
    elif node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                name = child.child_by_field_name("name")
                return (_node_text(name, source), "function")
    return (None, "module")


def _get_rust_symbol(node: Node, source: bytes) -> tuple[str | None, str | None]:
    type_map = {
        "function_item": "function",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "class",
        "impl_item": "class",
        "macro_rules": "function",
        "type_item": "module",
        "const_item": "module",
    }
    symbol_type = type_map.get(node.type, "module")

    name = node.child_by_field_name("name")
    if name is not None:
        return (_node_text(name, source), symbol_type)

    # impl_item may not have a "name" field — extract the implemented type instead.
    # e.g. `impl fmt::Display for Point` → look for type_identifier after "for"
    if node.type == "impl_item":
        found_for = False
        for child in node.children:
            if child.type == "for":
                found_for = True
            elif found_for and child.type in ("type_identifier", "scoped_type_identifier"):
                return (_node_text(child, source), symbol_type)
        # Fall back: take the first type_identifier
        for child in node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                return (_node_text(child, source), symbol_type)

    return (None, symbol_type)


def _extract_imports(root: Node, source: bytes, language: str) -> list[str]:
    import_types = _IMPORT_TYPES[language]
    return [
        source[child.start_byte:child.end_byte].decode("utf-8")
        for child in root.children
        if child.type in import_types
    ]


def _split_large_node(
    node: Node,
    source: bytes,
    file_path: str,
    language: str,
    imports: list[str],
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[CodeChunk]:
    """Split a large node by extracting methods from its body, or fall back to token splitting."""
    method_types = _METHOD_TYPES.get(language, set())
    body_types = _BODY_TYPES.get(language, set())
    parent_name, _ = _get_symbol(node, source, language)

    methods: list[Node] = []
    for child in node.children:
        if child.type in body_types:
            for grandchild in child.children:
                if grandchild.type in method_types:
                    methods.append(grandchild)

    if not methods:
        node_text = source[node.start_byte:node.end_byte].decode("utf-8")
        parts = split_at_token_boundary(node_text, max_tokens, overlap_tokens)
        node_start = node.start_point[0] + 1
        node_end = node.end_point[0] + 1
        return [
            CodeChunk(
                content=part,
                file_path=file_path,
                language=language,
                symbol_name=parent_name,
                symbol_type="module",
                start_line=node_start,
                end_line=node_end,
                imports=imports,
            )
            for part in parts
        ]

    chunks: list[CodeChunk] = []
    for method in methods:
        method_text = source[method.start_byte:method.end_byte].decode("utf-8")
        method_name, _ = _get_symbol(method, source, language)
        qualified_name = (
            f"{parent_name}.{method_name}" if parent_name and method_name
            else method_name or parent_name
        )

        tokens = count_tokens(method_text)
        if tokens < min_tokens and chunks:
            prev = chunks[-1]
            chunks[-1] = CodeChunk(
                content=prev.content + "\n\n" + method_text,
                file_path=prev.file_path,
                language=prev.language,
                symbol_name=prev.symbol_name,
                symbol_type=prev.symbol_type,
                start_line=prev.start_line,
                end_line=method.end_point[0] + 1,
                imports=imports,
            )
        else:
            chunks.append(CodeChunk(
                content=method_text,
                file_path=file_path,
                language=language,
                symbol_name=qualified_name,
                symbol_type="method",
                start_line=method.start_point[0] + 1,
                end_line=method.end_point[0] + 1,
                imports=imports,
            ))

    return chunks


def _node_text(node: Node | None, source: bytes) -> str | None:
    if node is None:
        return None
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _merge_small_chunks(chunks: list[CodeChunk], min_tokens: int) -> list[CodeChunk]:
    """Merge consecutive chunks that are too small, but only when both are anonymous
    (symbol_name is None). Named AST symbols should not be merged across boundaries
    because they represent distinct semantic units.
    """
    if not chunks:
        return []
    merged: list[CodeChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        # Only merge if prev is small AND both chunks are anonymous module-level fragments
        can_merge = (
            count_tokens(prev.content) < min_tokens
            and prev.symbol_name is None
            and chunk.symbol_name is None
        )
        if can_merge:
            merged[-1] = CodeChunk(
                content=prev.content + "\n\n" + chunk.content,
                file_path=prev.file_path,
                language=prev.language,
                symbol_name=prev.symbol_name,
                symbol_type=prev.symbol_type,
                start_line=prev.start_line,
                end_line=chunk.end_line,
                imports=prev.imports,
            )
        else:
            merged.append(chunk)
    return merged
