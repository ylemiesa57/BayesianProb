"""Static structural checks for the network built by ComplexCyber.py.

Like test_cpt_validity.py, this deliberately does NOT import ComplexCyber.py
(it executes several 1000-run sensitivity_analysis() calls at module level,
which takes minutes and has nothing to do with graph structure). Instead it
parses the source with `ast` and inspects the literal add_node/add_edge/
set_cpt calls directly.

Guards against a real copy-paste bug found and fixed in this file: six nodes
('(0,2)', 'ssh(1,2)', 'ssh(0,2)', 'user(2)', 'local_bof(2)', '(2,2)') were
each declared with add_node() twice. BayesianNetwork.add_node() stores nodes
in a plain dict keyed by name, so the duplicates were harmless no-ops rather
than a state-corrupting bug -- but they were dead, copy-pasted code that
made the network definition harder to read and easy to accidentally diverge
(e.g. if a future edit only updated one of the two duplicate declarations).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _calls(tree, method):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
        ):
            yield node


def _first_str_arg(call):
    if call.args and isinstance(call.args[0], ast.Constant):
        return call.args[0].value
    return None


def test_no_duplicate_add_node_calls():
    """Every node name should be declared with add_node() exactly once."""
    tree = ast.parse((REPO_ROOT / "ComplexCyber.py").read_text())
    names = [_first_str_arg(c) for c in _calls(tree, "add_node")]
    names = [n for n in names if n is not None]

    seen = set()
    duplicates = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)

    assert not duplicates, f"add_node() called more than once for: {sorted(duplicates)}"


def test_every_edge_endpoint_has_a_declared_node():
    """Every add_edge(parent, child) endpoint must appear in some add_node() call."""
    tree = ast.parse((REPO_ROOT / "ComplexCyber.py").read_text())
    declared_nodes = {_first_str_arg(c) for c in _calls(tree, "add_node")}
    declared_nodes.discard(None)

    for call in _calls(tree, "add_edge"):
        args = [a.value for a in call.args if isinstance(a, ast.Constant)]
        assert len(args) == 2, f"add_edge() call with unexpected args: {ast.dump(call)}"
        parent, child = args
        assert parent in declared_nodes, f"add_edge() references undeclared node {parent!r}"
        assert child in declared_nodes, f"add_edge() references undeclared node {child!r}"
