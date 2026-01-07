#!/usr/bin/env python3
"""
Tree-sitter demo script for parsing Python code.
Demonstrates queries for:
1. List all classes in a file
2. List all methods and attributes of a specific class
3. List all top-level functions
4. List all global variables
"""

import sys
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query, QueryCursor


def create_parser():
    """Create and return a Python parser."""
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser, PY_LANGUAGE


def parse_file(parser, file_path: str) -> bytes:
    """Parse a Python file and return the tree."""
    with open(file_path, "rb") as f:
        source_code = f.read()
    tree = parser.parse(source_code)
    return tree, source_code


def list_all_classes(language, tree, source_code: bytes) -> list[str]:
    """
    Query 1: List all class names in the file.
    """
    query_text = """
    (class_definition
      name: (identifier) @class_name)
    """
    query = Query(language, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    classes = []
    for node in captures.get("class_name", []):
        class_name = source_code[node.start_byte:node.end_byte].decode("utf-8")
        classes.append(class_name)

    return classes


def list_class_members(language, tree, source_code: bytes, target_class: str) -> dict:
    """
    Query 2: List all methods and attributes of a specific class.
    """
    # First, find the target class node and its body using matches() for proper grouping
    class_query_text = """
    (class_definition
      name: (identifier) @class_name
      body: (block) @class_body)
    """
    class_query = Query(language, class_query_text)
    cursor = QueryCursor(class_query)

    # Find the body node for the target class using matches() for proper grouping
    target_body = None
    for pattern_idx, captures in cursor.matches(tree.root_node):
        name_nodes = captures.get("class_name", [])
        body_nodes = captures.get("class_body", [])
        if name_nodes and body_nodes:
            class_name = source_code[name_nodes[0].start_byte:name_nodes[0].end_byte].decode("utf-8")
            if class_name == target_class:
                target_body = body_nodes[0]
                break

    if target_body is None:
        return {"methods": [], "attributes": []}

    # Query for methods (function definitions) within the class body
    method_query_text = """
    (function_definition
      name: (identifier) @method_name)
    """
    method_query = Query(language, method_query_text)
    method_cursor = QueryCursor(method_query)
    method_captures = method_cursor.captures(target_body)

    methods = []
    for node in method_captures.get("method_name", []):
        method_name = source_code[node.start_byte:node.end_byte].decode("utf-8")
        methods.append(method_name)

    # Query for class-level assignments (attributes)
    # This finds self.x = ... patterns in __init__ and other methods
    attr_query_text = """
    (assignment
      left: (attribute
        object: (identifier) @obj
        attribute: (identifier) @attr_name))
    """
    attr_query = Query(language, attr_query_text)
    attr_cursor = QueryCursor(attr_query)
    attr_captures = attr_cursor.captures(target_body)

    # Filter for self.x assignments
    attributes = set()
    obj_nodes = attr_captures.get("obj", [])
    attr_nodes = attr_captures.get("attr_name", [])
    for obj_node, attr_node in zip(obj_nodes, attr_nodes):
        obj_name = source_code[obj_node.start_byte:obj_node.end_byte].decode("utf-8")
        if obj_name == "self":
            attr_name = source_code[attr_node.start_byte:attr_node.end_byte].decode("utf-8")
            attributes.add(attr_name)

    return {
        "methods": methods,
        "attributes": sorted(attributes)
    }


def list_top_level_functions(language, tree, source_code: bytes) -> list[str]:
    """
    Query 3: List all top-level function definitions (not inside classes).
    """
    # Query for all function definitions at module level
    query_text = """
    (module
      (function_definition
        name: (identifier) @func_name))
    """
    query = Query(language, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    functions = []
    for node in captures.get("func_name", []):
        func_name = source_code[node.start_byte:node.end_byte].decode("utf-8")
        functions.append(func_name)

    return functions


def list_global_variables(language, tree, source_code: bytes) -> list[str]:
    """
    Query 4: List all global variable assignments at module level.
    """
    # Query for assignments at module level (not inside functions or classes)
    query_text = """
    (module
      (expression_statement
        (assignment
          left: (identifier) @var_name)))
    """
    query = Query(language, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    # Also check for pattern assignments (e.g., a = b = c)
    pattern_query_text = """
    (module
      (expression_statement
        (assignment
          left: (pattern_list
            (identifier) @var_name))))
    """
    pattern_query = Query(language, pattern_query_text)
    pattern_cursor = QueryCursor(pattern_query)
    pattern_captures = pattern_cursor.captures(tree.root_node)

    variables = set()
    for node in captures.get("var_name", []):
        var_name = source_code[node.start_byte:node.end_byte].decode("utf-8")
        variables.add(var_name)

    for node in pattern_captures.get("var_name", []):
        var_name = source_code[node.start_byte:node.end_byte].decode("utf-8")
        variables.add(var_name)

    return sorted(variables)


def main():
    # Default to ingest.py if no file specified
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = Path(__file__).parent.parent / "ingest.py"

    print(f"Parsing: {file_path}\n")

    # Create parser
    parser, language = create_parser()

    # Parse file
    tree, source_code = parse_file(parser, file_path)

    # Query 1: List all classes
    print("=" * 60)
    print("QUERY 1: List all classes")
    print("=" * 60)
    classes = list_all_classes(language, tree, source_code)
    print(f"Found {len(classes)} classes:\n")
    for i, class_name in enumerate(classes, 1):
        print(f"  {i}. {class_name}")

    # Query 2: List members of each class
    if classes:
        print("\n" + "=" * 60)
        print("QUERY 2: List methods and attributes for each class")
        print("=" * 60)

        for class_name in classes:
            members = list_class_members(language, tree, source_code, class_name)
            print(f"\n{class_name}:")
            print(f"  Methods ({len(members['methods'])}):")
            for method in members['methods']:
                print(f"    - {method}()")
            print(f"  Attributes ({len(members['attributes'])}):")
            for attr in members['attributes']:
                print(f"    - self.{attr}")

    # Query 3: List top-level functions
    print("\n" + "=" * 60)
    print("QUERY 3: List all top-level functions")
    print("=" * 60)
    functions = list_top_level_functions(language, tree, source_code)
    print(f"Found {len(functions)} top-level functions:\n")
    for i, func_name in enumerate(functions, 1):
        print(f"  {i}. {func_name}()")

    # Query 4: List global variables
    print("\n" + "=" * 60)
    print("QUERY 4: List all global variables")
    print("=" * 60)
    variables = list_global_variables(language, tree, source_code)
    print(f"Found {len(variables)} global variables:\n")
    for i, var_name in enumerate(variables, 1):
        print(f"  {i}. {var_name}")


if __name__ == "__main__":
    main()
