# SPDX-License-Identifier: Apache-2.0
# Derived from EvalPlus0.3.1, https://github.com/evalplus/evalplus
# Modified2026-11-03: retain reachable helper definitions regardless of
# explicit return. Valid generators/mutating helpers can have no return statement.
# All code extraction, dependency pruning, imports and collision rules retained.
from typing import Optional
import tree_sitter_python
from tree_sitter import Parser,Language
from evalplus.sanitize import (code_extract,has_return_statement,CLASS_TYPE,FUNCTION_TYPE,IMPORT_TYPE,
    EXPRESSION_TYPE,ASSIGNMENT_TYPE,get_definition_name,get_deps,get_function_dependency)

def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ) and (name != entrypoint or has_return_statement(child)):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reacheable):
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return sanitized_output[:-1].decode("utf8")

def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code=extract_target_code_or_empty(code,entrypoint).strip()
    return sanitized_code if sanitized_code else code_extract(code)
