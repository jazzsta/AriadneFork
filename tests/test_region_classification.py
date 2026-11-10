"""Tests for the UI-facing region classification wrapper."""

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")

from ariadne_roots import quantify


def test_classify_region_maps_and_casts_fields(monkeypatch):
    """The wrapper maps sampling names to CSV names and casts numpy values."""
    calls = {}

    def fake_estimate_rounds_for_region(H, **kwargs):
        calls["graph"] = H
        calls["kwargs"] = kwargs
        return {
            "n_inside": 2,
            "mean_rounds_normalized": np.float64(0.5),
            "ci_low_normalized": np.float64(0.4),
            "ci_high_normalized": np.float64(0.6),
            "mean_alpha": np.float64(0.25),
            "alpha_ci_low": np.float64(0.2),
            "alpha_ci_high": np.float64(0.3),
        }

    monkeypatch.setattr(
        quantify, "estimate_rounds_for_region", fake_estimate_rounds_for_region
    )

    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(0, 10))
    G.add_edge(0, 1, weight=10)

    front = {0.0: [10.0, 10.0]}
    actual = (10.0, 10.0)
    save_path = "sampling.svg"

    def progress(done, total, n_inside):
        pass

    result = quantify.classify_region(
        G,
        front,
        actual,
        n_samples=7,
        region_factor=0.25,
        save_path=save_path,
        scale_factor=2.0,
        progress_callback=progress,
        verbose=True,
    )

    assert result == {
        "mean_rounds_normalized": 0.5,
        "ci_low_normalized": 0.4,
        "ci_high_normalized": 0.6,
        "local_mean_alpha": 0.25,
        "local_alpha_ci_low": 0.2,
        "local_alpha_ci_high": 0.3,
    }
    assert all(
        value is None or type(value) is float for value in result.values()
    )

    assert calls["kwargs"]["n_samples"] == 7
    assert calls["kwargs"]["region_factor"] == 0.25
    assert calls["kwargs"]["save_path"] == save_path
    assert calls["kwargs"]["scale_factor"] == 2.0
    assert calls["kwargs"]["plot"] is True
    assert calls["kwargs"]["show"] is False
    assert calls["kwargs"]["front"] is front
    assert calls["kwargs"]["actual"] is actual
    assert calls["kwargs"]["progress_callback"] is progress
    assert not calls["graph"].is_directed()


def test_classify_region_collects_trees_when_saving_html(monkeypatch):
    """The HTML path needs tree payloads for the hover panel."""
    calls = {}

    def fake_estimate_rounds_for_region(H, **kwargs):
        calls["kwargs"] = kwargs
        return {
            "n_inside": 0,
            "mean_rounds_normalized": None,
            "ci_low_normalized": None,
            "ci_high_normalized": None,
            "mean_alpha": None,
            "alpha_ci_low": None,
            "alpha_ci_high": None,
        }

    def fake_save_sampling_html(result, dest, **kwargs):
        calls["html"] = (result, dest, kwargs)

    monkeypatch.setattr(
        quantify, "estimate_rounds_for_region", fake_estimate_rounds_for_region
    )
    monkeypatch.setattr(
        quantify, "save_sampling_html", fake_save_sampling_html
    )

    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_edge(0, 1)

    quantify.classify_region(
        G,
        {},
        (0.0, 0.0),
        n_samples=1,
        save_html_path="sampling.html",
    )

    assert calls["kwargs"]["collect_trees"] is True
    assert calls["html"][1] == "sampling.html"


def test_classify_region_preserves_none_fields(monkeypatch):
    """Empty regions stay None rather than becoming NaN or zero."""

    def fake_estimate_rounds_for_region(H, **kwargs):
        return {
            "n_inside": 0,
            "mean_rounds_normalized": None,
            "ci_low_normalized": None,
            "ci_high_normalized": None,
            "mean_alpha": None,
            "alpha_ci_low": None,
            "alpha_ci_high": None,
        }

    monkeypatch.setattr(
        quantify, "estimate_rounds_for_region", fake_estimate_rounds_for_region
    )

    result = quantify.classify_region(
        nx.Graph(), {}, (0.0, 0.0), n_samples=1
    )

    assert result == {field: None for field in quantify.REGION_CLASSIFICATION_FIELDS}


def test_estimate_rounds_for_region_reuses_front_and_saves_svg(
    monkeypatch, tmp_path
):
    """Passing front/actual avoids recomputation, and save_path writes an SVG."""
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(0, 10))
    G.add_edge(0, 1, weight=10)

    front = {0.0: [10.0, 10.0], 1.0: [10.0, 20.0]}
    actual = (10.0, 10.0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Pareto front should have been reused")

    monkeypatch.setattr(quantify, "pareto_front", fail_if_called)

    save_path = tmp_path / "sampling.svg"
    result = quantify.estimate_rounds_for_region(
        G,
        n_samples=0,
        plot=True,
        show=False,
        save_path=save_path,
        front=front,
        actual=actual,
        scale_factor=2.0,
        verbose=False,
    )

    assert result["n_inside"] == 0
    assert result["front"] == front
    assert result["actual"] is actual
    assert result["trees"] == []
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_progress_callback_reports_sampling_progress():
    """The sampling routine reports done/total counts at regular intervals."""
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(0, 10))
    G.add_edge(0, 1, weight=10)

    front = {0.0: [10.0, 10.0], 1.0: [10.0, 20.0]}
    actual = (10.0, 10.0)
    calls = []

    quantify.estimate_rounds_for_region(
        G,
        n_samples=10,
        plot=False,
        show=False,
        front=front,
        actual=actual,
        verbose=False,
        progress_callback=lambda done, total, inside: calls.append(
            (done, total, inside)
        ),
    )

    assert calls
    assert calls[-1][0] == 10
    assert calls[-1][1] == 10
    assert all(total == 10 for _, total, _ in calls)
    assert [done for done, _, _ in calls] == sorted(
        done for done, _, _ in calls
    )


def test_sampling_refines_front_with_non_dominated_trees():
    """Sampled trees that are not dominated are added to the returned front."""
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(0, 10))
    G.add_node(2, pos=(5, 10))
    G.add_edge(0, 1, weight=10)
    G.add_edge(1, 2, weight=5)

    # Extreme points that the sampled trees cannot dominate, leaving room for
    # sampled points to fill in the middle of the front.
    front = {0.0: [1.0, 1000.0], 1.0: [1000.0, 1.0]}
    actual = (10.0, 10.0)

    result = quantify.estimate_rounds_for_region(
        G,
        n_samples=20,
        plot=False,
        show=False,
        front=front,
        actual=actual,
        verbose=False,
    )

    updated = result["front"]
    assert len(updated) > len(front)

    points = list(updated.values())
    for i, (w1, t1) in enumerate(points):
        for j, (w2, t2) in enumerate(points):
            if i == j:
                continue
            assert not (
                w2 <= w1 and t2 <= t1 and (w2 < w1 or t2 < t1)
            ), "updated front still contains a dominated point"


def test_save_sampling_html_writes_interactive_plot(tmp_path):
    """The HTML output uses Plotly and has no rasterized plot elements."""
    result = {
        "n_inside": 1,
        "n_eff": 1.0,
        "mean_rounds_normalized": 0.5,
        "ci_low_normalized": 0.4,
        "ci_high_normalized": 0.6,
        "mean_alpha": 0.25,
        "alpha_ci_low": 0.2,
        "alpha_ci_high": 0.3,
        "all_points": [(10.0, 10.0, 1, 0.25, True)],
        "actual": (10.0, 10.0),
        "front": {0.0: [10.0, 10.0], 1.0: [10.0, 20.0]},
        "radius": (1.0, 1.0),
        "max_rounds": 3,
        "trees": [
            {
                "p": [[0, 0], [0, 10]],
                "e": [[0, 1]],
                "b": 0,
                "a": 0.25,
                "r": 1,
            }
        ],
        "actual_tree": {
            "p": [[0, 0], [0, 10]],
            "e": [[0, 1]],
            "b": 0,
            "a": None,
            "r": None,
        },
    }

    html_path = tmp_path / "sampling.html"
    quantify.save_sampling_html(
        result,
        html_path,
        scale_factor=1.0,
        region_factor=0.5,
        unit="cm",
    )

    text = html_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text
    assert "plotly.min.js" in text
    assert "<image" not in text
    assert '"xref":"paper"' in text or '"xref": "paper"' in text
    assert "rsa-opt-marker" in text
    assert "rect.cboutline" in text
    assert "rsa-hover-tree" in text
    assert 'const trees = [{"p":[[0,0],[0,10]]' in text
    assert 'const actualTree = {"p":[[0,0],[0,10]]' in text
    assert "Actual (straightened) RSA" in text
    assert (tmp_path / "plotly.min.js").exists()


def test_collect_trees_aligns_with_sampled_points():
    """Each stored tree payload matches one sampled point for the hover panel."""
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(0, 100))
    G.add_node(2, pos=(60, 100))
    G.add_edge(0, 1, weight=100)
    G.add_edge(1, 2, weight=60)

    front = {0.0: [120.0, 300.0], 1.0: [300.0, 120.0]}
    actual = (100.0, 100.0)

    result = quantify.estimate_rounds_for_region(
        G,
        n_samples=8,
        plot=False,
        show=False,
        front=front,
        actual=actual,
        verbose=False,
        collect_trees=True,
    )

    assert result["trees"]
    assert len(result["trees"]) == len(result["all_points"])
    for tree in result["trees"]:
        assert len(tree["p"]) == len(tree["e"]) + 1
        assert tree["b"] == 0

        indices = set()
        for a, b in tree["e"]:
            indices.update((a, b))
        assert indices == set(range(len(tree["p"])))

    actual_tree = result["actual_tree"]
    assert actual_tree is not None
    assert len(actual_tree["p"]) == len(actual_tree["e"]) + 1
    assert actual_tree["a"] is None
    assert actual_tree["r"] is None
