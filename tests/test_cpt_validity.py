"""Static validity checks for the hardcoded CPTs in CyberBN.py and ComplexCyber.py.

BN.py's inference/sensitivity_analysis code indexes these conditional probability
tables directly with no renormalization, so a CPT whose probabilities for a given
parent-state combination don't sum to 1 silently skews every inference result that
touches it (see the vul_exec_0_1_cpt / vul_ssh_1_2_cpt typo fixed in commit 5999d32,
where an extra leading zero made a row sum to 0.172 instead of 1.0).

That fix has been manually re-verified by hand on several separate occasions since
(each time by writing a one-off throwaway script) without ever landing as an actual
test in this repo, so the check could regress silently. This makes it permanent.

This does NOT import CyberBN.py/ComplexCyber.py directly -- doing so executes their
full top-level script (network construction plus several 1000-run
sensitivity_analysis() calls), which takes minutes and has nothing to do with CPT
validity. Instead it parses the source with `ast` and evaluates only the literal
dict assigned to each `*_cpt` name, so it stays fast and has zero side effects.
Random single-parent CPTs like `{'T': a, 'F': 1-a}` aren't literal (they reference a
variable) and are skipped -- they're structurally guaranteed to sum to 1 by
construction.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CPT_FILES = ["CyberBN.py", "ComplexCyber.py"]


def _literal_cpts(path: Path):
    """Yield (var_name, literal_dict) for every top-level `*_cpt = {...}` assignment
    in `path` whose right-hand side is a pure literal (no names/expressions)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith("_cpt"):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue  # not a pure literal (e.g. references a sampled variable)
                yield target.id, value


def _rows(cpt_dict):
    """Split a parsed CPT into (row_label, {'T': .., 'F': ..}) pairs.

    Handles both a plain single-parent CPT ({'T': .53, 'F': .47}) and a
    multi-parent CPT keyed by a parent-state tuple/str
    ({('T','T','T'): {'T': .53, 'F': .47}, ...}).
    """
    if all(isinstance(v, dict) for v in cpt_dict.values()):
        yield from cpt_dict.items()
    else:
        yield None, cpt_dict


def _cases():
    for fname in CPT_FILES:
        path = REPO_ROOT / fname
        for cpt_name, cpt_dict in _literal_cpts(path):
            for row_label, row in _rows(cpt_dict):
                case_id = f"{fname}:{cpt_name}" + (f"[{row_label}]" if row_label is not None else "")
                yield pytest.param(fname, cpt_name, row_label, row, id=case_id)


@pytest.mark.parametrize("fname,cpt_name,row_label,row", list(_cases()))
def test_cpt_row_sums_to_one(fname, cpt_name, row_label, row):
    total = sum(row.values())
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"{fname}:{cpt_name}{'[' + repr(row_label) + ']' if row_label is not None else ''} "
        f"sums to {total}, not 1.0: {row}"
    )


def test_found_at_least_one_cpt_per_file():
    """Guard against the scan silently finding nothing (e.g. after a refactor)."""
    for fname in CPT_FILES:
        path = REPO_ROOT / fname
        found = list(_literal_cpts(path))
        assert found, f"no literal *_cpt assignments found in {fname} -- did the format change?"
