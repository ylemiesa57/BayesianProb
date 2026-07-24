"""Regression tests for BN.py's sensitivity_analysis()/inference() edge cases.

BN.py itself is safe to import directly (unlike CyberBN.py/ComplexCyber.py,
which run several 1000-iteration sensitivity_analysis() calls at module
level and are tested via ast parsing instead -- see test_cpt_validity.py and
test_graph_construction.py).

Guards against two related bugs found and fixed together:

1. sensitivity_analysis()'s per-state perturbation sign was computed as
   `pos_or_neg = np.random.normal(...); if pos_or_neg < 0: pos_or_neg = -1`
   followed immediately by an unconditional `pos_or_neg = 1` -- so every
   perturbation was always positive-direction regardless of the coin flip
   above it, silently biasing every sensitivity_analysis() run. Fixed by
   moving the `pos_or_neg = 1` into the missing `else` branch.

2. Fixing (1) means CPT state probabilities can now genuinely be pushed
   toward 0 in both directions, which surfaced a latent
   ZeroDivisionError in both sensitivity_analysis()'s CPT re-normalization
   and inference()'s target-probability re-normalization whenever every
   state's probability lands at exactly 0 (previously unreachable, since
   perturbations only ever pushed values up before fix #1). Both now guard
   with `if total_prob > 0` before dividing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BN import BayesianNetwork  # noqa: E402


def _toy_network():
    bn = BayesianNetwork()
    bn.add_node("A", ["T", "F"])
    bn.add_node("B", ["T", "F"])
    bn.add_edge("A", "B")
    bn.set_cpt("A", {"T": 0.6, "F": 0.4})
    bn.set_cpt(
        "B",
        {
            "T": {"T": 0.9, "F": 0.1},
            "F": {"T": 0.2, "F": 0.8},
        },
    )
    return bn


def test_negative_draw_actually_decreases_perturbed_cpt_direction():
    """The historical bug made every perturbation direction positive
    regardless of the coin-flip draw's sign, so a node's CPT probabilities
    could only ever increase (or hold at the 1.0 clamp), never decrease.

    Exercises the real BN.sensitivity_analysis() call path (not a
    reimplementation of the snippet) by monkeypatching np.random.normal and
    sample_truncated to deterministic values, then checks that a forced
    *negative* draw genuinely decreases P(B=T) relative to a forced
    *positive* draw. B's CPT is set up so that P(B=T|A=T)=0.9 >
    P(B=T|A=F)=0.2, so pushing A's P(A=T) down must push P(B=T) down too --
    that only happens if the -1 branch actually gets applied instead of
    being silently overwritten to +1.
    """
    import BN as bn_module

    def run_one_iteration(forced_normal_value):
        bn = _toy_network()
        real_normal = np.random.normal
        bn_module.np.random.normal = lambda *a, **k: forced_normal_value
        bn_module.sample_truncated = lambda *a, **k: 0.3
        try:
            # num_runs is hardcoded to 1000 inside sensitivity_analysis, but
            # every iteration is independent (cpts restored each time), so
            # the first result already reflects the forced draw.
            results = bn.sensitivity_analysis("B", "normal")
        finally:
            bn_module.np.random.normal = real_normal
        return results[0]

    negative_result = run_one_iteration(forced_normal_value=-5.0)
    positive_result = run_one_iteration(forced_normal_value=5.0)

    # Hand-computed expected values for this exact toy network with a fixed
    # gen_noise=0.3: a forced -1 perturbs A's raw CPT to {T: 0.3, F: 0.1},
    # renormalizing to {T: 0.75, F: 0.25}; a forced +1 perturbs it to
    # {T: 0.9, F: 0.7}, renormalizing to {T: 0.5625, F: 0.4375}. Propagated
    # through P(B=T|A=T)=0.9, P(B=T|A=F)=0.2 that gives P(B=T)=0.725 for the
    # negative case and 0.59375 for the positive case -- these differ (and
    # are *not* simply "negative is lower", since renormalization after a
    # uniform subtraction actually raises the majority state's share here),
    # so this is a precise fingerprint of the -1 branch genuinely running
    # rather than being silently overwritten to +1 every time.
    assert negative_result == pytest.approx(0.725), (
        f"forced-negative-draw iteration gave P(B=T)={negative_result}, "
        "expected 0.725 -- the -1 branch may be getting overwritten again"
    )
    assert positive_result == pytest.approx(0.59375), (
        f"forced-positive-draw iteration gave P(B=T)={positive_result}, "
        "expected 0.59375"
    )
    assert negative_result != positive_result


def test_sensitivity_analysis_runs_full_1000_iterations_without_crashing():
    """End-to-end run: this used to raise ZeroDivisionError once negative
    perturbations were actually possible (see module docstring, bug #2)."""
    np.random.seed(42)
    bn = _toy_network()

    results = bn.sensitivity_analysis("B", "normal")

    assert len(results) == 1000
    assert all(0.0 <= r <= 1.0 for r in results)
    assert not any(r != r for r in results)  # no NaNs


def test_inference_does_not_crash_on_a_degenerate_zero_probability_network():
    """A network where every state's probability is 0 has no well-defined
    normalized distribution; inference() should return zeros rather than
    raising ZeroDivisionError."""
    bn = _toy_network()
    bn.cpts["B"] = {"T": {"T": 0.0, "F": 0.0}, "F": {"T": 0.0, "F": 0.0}}

    result = bn.inference("B")

    assert result == {"T": 0.0, "F": 0.0}
