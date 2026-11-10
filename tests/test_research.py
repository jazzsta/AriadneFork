"""Smoke tests for the refactored module boundaries."""

import importlib
from pathlib import Path

from ariadne_roots import research

main_module = importlib.import_module("ariadne_roots.main")
quantify_module = importlib.import_module("ariadne_roots.quantify")
pareto_module = importlib.import_module("ariadne_roots.pareto_functions")


def test_research_module_exposes_workflow_functions():
    """The research module is import-safe and exposes the story entry points."""
    for name in [
        "pareto_front",
        "pareto_steiner_fast",
        "random_tree",
        "random_tree_normalized",
        "generate_trees_from_tips_topdown",
        "generate_trees_from_tips_bottomup",
        "plot_rsa_and_abstraction",
        "plot_steiner_sequence",
        "plot_random_sequence",
        "plot_randomization_landscape",
        "sample_region",
    ]:
        assert callable(getattr(research, name))


def test_research_notebook_exists():
    """The Ariadne 0.3 research notebook is shipped with the project."""
    notebook = Path("Ariadne0.3_Exploration.ipynb")
    assert notebook.exists()
    assert notebook.stat().st_size > 0


def test_ui_modules_do_not_import_research_or_legacy():
    """UI/analysis modules must not depend on research or legacy code."""
    assert callable(quantify_module.get_graph_from_json)
    forbidden = (
        "from ariadne_roots.exploration",
        "from ariadne_roots.visualisation",
        "from ariadne_roots.research",
        "from ariadne_roots.legacy",
        "import ariadne_roots.exploration",
        "import ariadne_roots.visualisation",
        "import ariadne_roots.research",
        "import ariadne_roots.legacy",
    )
    for module in (main_module, quantify_module, pareto_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for entry in forbidden:
            assert entry not in source, f"{module.__name__} imports {entry}"


def test_research_does_not_import_legacy_or_ui_research_layers():
    """Research code should be self-contained apart from quantify/pareto primitives."""
    source = Path(research.__file__).read_text(encoding="utf-8")
    assert "from ariadne_roots.legacy" not in source
    assert "from ariadne_roots.exploration" not in source
    assert "from ariadne_roots.visualisation" not in source
