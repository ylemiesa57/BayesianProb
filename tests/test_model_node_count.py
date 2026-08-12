"""
Regression tests for a silent node-dropping bug in ComplexCyber.py.

Both "Top 4" sections in ComplexCyber.py declared a 4-element `nodes` list
but only ever built 3-element `top_arrs`/`infs` lists to pass into
`complex_bn.model(...)`. Since `model()` used to build its plotted DataFrame
with `dict(zip(nodes, diffs))`, the mismatched lengths meant the 4th named
node was silently dropped from the plot -- no error, no warning, just one
fewer box in the box plot than the "Top 4" comment promised.

Two checks:
1. A static, ast-based check (matching the existing test_graph_construction.py/
   test_cpt_validity.py convention of not importing ComplexCyber.py directly,
   since doing so runs several 1000-iteration simulations) that both
   `nodes = [...]` blocks in ComplexCyber.py feed exactly as many
   sensitivity_analysis()/inference() calls into complex_bn.model() as there
   are node names.
2. A live check that BN.BayesianNetwork.model() itself now rejects
   mismatched top_arrs/infs/nodes lengths instead of silently truncating via
   zip, and still behaves normally when the lengths do match.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from BN import BayesianNetwork  # noqa: E402


def _string_list(node: ast.AST):
    if isinstance(node, ast.List):
        return [elt.value for elt in node.elts if isinstance(elt, ast.Constant)]
    return None


def _model_calls_with_preceding_nodes(tree: ast.Module):
    """Yield (nodes_list, model_call) for each `complex_bn.model(...)` call in
    module body order, paired with whichever `nodes = [...]` assignment most
    recently preceded it."""
    current_nodes = None
    for stmt in tree.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "nodes"
            ):
                maybe_list = _string_list(node.value)
                if maybe_list is not None:
                    current_nodes = maybe_list
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "model"
            ):
                yield current_nodes, node


def test_complex_cyber_model_calls_use_every_selected_node():
    tree = ast.parse((REPO_ROOT / "ComplexCyber.py").read_text())
    pairs = list(_model_calls_with_preceding_nodes(tree))
    assert len(pairs) == 4, f"expected 4 complex_bn.model() calls, found {len(pairs)}"

    for nodes, call in pairs:
        assert nodes is not None, "model() call with no preceding `nodes = [...]`"
        top_arrs_arg = call.args[0]
        assert isinstance(top_arrs_arg, ast.List)
        sensitivity_calls = [
            elt
            for elt in top_arrs_arg.elts
            if isinstance(elt, ast.Call)
            and isinstance(elt.func, ast.Attribute)
            and elt.func.attr == "sensitivity_analysis"
        ]
        assert len(sensitivity_calls) == len(nodes), (
            f"nodes={nodes} has {len(nodes)} entries but only "
            f"{len(sensitivity_calls)} sensitivity_analysis() calls were "
            "passed into model() -- a node would be silently dropped"
        )


class TestModelLengthValidation:
    def setup_method(self):
        self.bn = BayesianNetwork()
        self.bn.add_node("A", ["T", "F"])
        self.bn.add_node("B", ["T", "F"])
        self.bn.add_edge("A", "B")
        self.bn.set_cpt("A", {"T": 0.6, "F": 0.4})
        self.bn.set_cpt(
            "B", {"T": {"T": 0.9, "F": 0.1}, "F": {"T": 0.2, "F": 0.8}}
        )

    def test_mismatched_node_count_raises(self):
        with pytest.raises(ValueError, match="mismatched lengths"):
            self.bn.model(
                top_arrs=[[0.1], [0.2], [0.3]],
                infs=[0.1, 0.2, 0.3],
                nodes=["n0", "n1", "n2", "n3"],  # one more node than data
                normal_or_pareto="normal",
            )

    def test_matched_node_count_still_works(self, monkeypatch):
        # Avoid popping up an interactive plot in a headless test run.
        monkeypatch.setattr("BN.go.Figure.show", lambda self: None)
        result = self.bn.model(
            top_arrs=[[0.1, 0.15], [0.2, 0.25]],
            infs=[0.1, 0.2],
            nodes=["n0", "n1"],
            normal_or_pareto="normal",
        )
        assert result == "Completed"
