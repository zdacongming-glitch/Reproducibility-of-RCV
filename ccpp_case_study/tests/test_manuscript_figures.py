"""Compare all official figure panels with the original drawing routines.

Use CCPP_MANUSCRIPT_SRCV_RUN_DIR and CCPP_MANUSCRIPT_AACV_RUN_DIR to compare
actual results instead of synthetic CSVs. Supplied results are never modified.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plot_manuscript_figures as official
from ccpp_repro import aacv_plotting, plotting


def _config(method):
    name = "ccpp_aacv_full.json" if method == "AACV" else "ccpp_full.json"
    return json.loads((official.ROOT / "configs" / name).read_text(encoding="utf-8"))


def _synthetic(run_dir, method):
    """Use full configured grids, without fitting a model."""
    run_dir.mkdir(parents=True)
    config = _config(method)
    aacv = method == "AACV"
    exp = config["aacv_experiment" if aacv else "experiment"]
    radii, repeats = exp["radii"], exp["outer_repeats"]
    models = config["models"]["candidate_order"]
    candidates, summary, mismatch, frequencies = [], [], [], []
    selectors = ["clean", "aacv" if aacv else "SRCV"]
    for r in radii:
        for i, model in enumerate(models):
            row = {"radius": r, "model": model, "outer_repeats": repeats}
            prefix = "test_aacv" if aacv else "test_sr"
            row.update({f"{prefix}_mean": 1+i/10+r*(6-i), f"{prefix}_se": 0.01*(i+1)})
            candidates.append(row)
        if aacv:
            for i, selector in enumerate(selectors):
                summary.append({"radius": r, "selector": selector, "outer_repeats": repeats,
                                "heldout_best_match_rate": 0.3+i/2,
                                "heldout_score_excess_mean": r*(2-i),
                                "heldout_score_excess_se": 0.03})
        else:
            summary.append({"radius": r, "outer_repeats": repeats,
                            "clean_oracle_match_rate": 0.3, "sr_oracle_match_rate": 0.8,
                            "clean_regret_mean": r*2, "sr_regret_mean": r,
                            "clean_regret_se": 0.03, "sr_regret_se": 0.02})
    eval_radii = radii if aacv else config["mismatch"]["eval_radii"]
    for selector in selectors:
        for rv in radii:
            for re in eval_radii:
                # Include zero clean slices to check the original line conventions.
                mean = re if selector == "clean" else abs(rv-re)
                row = {"selector": selector, "r_val": rv, "r_eval": re, "outer_repeats": repeats}
                if aacv:
                    row.update(heldout_score_excess_mean=mean,
                               heldout_score_excess_mcse=0.01, heldout_best_match_rate=0.4)
                else:
                    row.update(regret_mean=mean, regret_mcse=0.01, oracle_match_rate=0.4)
                mismatch.append(row)
                for model in models:
                    frequencies.append({"selector": selector, "r_val": rv, "r_eval": re,
                                        "model": model, "frequency": 1/len(models)})
    frames = [pd.DataFrame(rows) for rows in [candidates, summary, mismatch, frequencies]]
    if aacv:
        frames = [frame.assign(analysis_variant=official.PRIMARY) for frame in frames]
        for i in [0, 1]:
            frames[i] = pd.concat([frames[i], frames[i].assign(analysis_variant="sensitivity")])
    names = (["aacv_candidate_test_scores.csv", "aacv_summary_by_radius.csv"] if aacv
             else ["candidate_test_risk_by_radius.csv", "summary_by_radius.csv"])
    subdir = config["mismatch"]["output_subdir"]
    names += [f"{subdir}/radius_mismatch_summary.csv",
              f"{subdir}/radius_mismatch_selection_frequencies.csv"]
    for frame, name in zip(frames, names):
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return run_dir


@pytest.fixture(scope="module")
def synthetic_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("manuscript-figures")
    return {m: _synthetic(root / m, m) for m in ("SRCV", "AACV")}


@pytest.fixture(scope="module")
def result_dirs(synthetic_dirs):
    return {m: Path(os.environ.get(f"CCPP_MANUSCRIPT_{m}_RUN_DIR", directory))
            for m, directory in synthetic_dirs.items()}


def _assert_axes_data(actual, expected):
    """Exact equality of plotted numbers, styles, bands, matrices and insets."""
    assert len(actual.lines) == len(expected.lines)
    for left, right in zip(actual.lines, expected.lines):
        np.testing.assert_array_equal(np.asarray(left.get_xdata(), float), np.asarray(right.get_xdata(), float))
        np.testing.assert_array_equal(np.asarray(left.get_ydata(), float), np.asarray(right.get_ydata(), float))
        for attr in ["color", "linestyle", "marker", "linewidth", "markersize"]:
            assert getattr(left, f"get_{attr}")() == getattr(right, f"get_{attr}")()
    assert len(actual.collections) == len(expected.collections)
    for left, right in zip(actual.collections, expected.collections):
        np.testing.assert_array_equal(left.get_facecolors(), right.get_facecolors())
        assert len(left.get_paths()) == len(right.get_paths())
        for lp, rp in zip(left.get_paths(), right.get_paths()):
            np.testing.assert_array_equal(lp.vertices, rp.vertices)
    assert len(actual.images) == len(expected.images)
    for left, right in zip(actual.images, expected.images):
        np.testing.assert_array_equal(left.get_array(), right.get_array())
        assert left.get_clim() == right.get_clim()
        assert left.get_cmap().name == right.get_cmap().name
    assert len(actual.child_axes) == len(expected.child_axes)
    for left, right in zip(actual.child_axes, expected.child_axes):
        _assert_axes_data(left, right)


@pytest.mark.parametrize("method", ["SRCV", "AACV"])
@pytest.mark.parametrize("family", ["heldout", "mismatch"])
@pytest.mark.parametrize("panel", list("ABCD"))
def test_panel_data_equal_original(method, family, panel, result_dirs, monkeypatch):
    directory = result_dirs[method]
    data = official.load_results(directory, method, _config(method))
    original = []
    module = aacv_plotting if method == "AACV" else plotting
    monkeypatch.setattr(module, "save_figure", lambda fig, *args: original.append(fig))
    if family == "heldout":
        if method == "AACV":
            module.draw_aacv_case_study_main(directory, directory, [], "unused", data.transition)
        else:
            module.draw_case_study_main(directory, directory, [], "unused")
    elif method == "AACV":
        module.draw_aacv_mismatch(directory, directory, [], "unused", data.slices)
    else:
        module.draw_radius_mismatch(directory, directory, [], "unused")
    fig, ax = plt.subplots()
    try:
        draw = official.draw_heldout if family == "heldout" else official.draw_mismatch
        draw(ax, data, panel)
        _assert_axes_data(ax, original[0].axes["ABCD".index(panel)])
    finally:
        plt.close(fig)
        plt.close(original[0])


def test_composites_and_standalones_have_same_data(result_dirs):
    srcv, aacv = [official.load_results(result_dirs[m], m, _config(m)) for m in ["SRCV", "AACV"]]
    figures = dict(official.make_figures(srcv, aacv))
    try:
        assert len(figures) == 10
        main = figures[official.STEMS["heldout"]]
        assert len(main.axes) == 6
        assert len(main.axes[3].child_axes) == 2
        mismatch = figures[official.STEMS["mismatch"]]
        assert len([a for a in mismatch.axes if a.images]) == 2
        for stem, (method, family, panel) in official.SINGLE_PANELS.items():
            row = int(method == "AACV")
            if family == "heldout":
                expected = figures[official.STEMS["heldout_excess"]].axes[row]
            else:
                expected = figures[official.STEMS["mismatch_excess"]].axes[2*row + int(panel == "C")]
            _assert_axes_data(figures[stem].axes[0], expected)
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_input_rejection(synthetic_dirs, tmp_path):
    data = official.load_results(synthetic_dirs["SRCV"], "SRCV", _config("SRCV"))
    with pytest.raises(ValueError, match="duplicate"):
        official._complete(pd.concat([data.candidate, data.candidate.iloc[:1]]),
                           {"radius": _config("SRCV")["experiment"]["radii"],
                            "model": _config("SRCV")["models"]["candidate_order"]},
                           ["mean", "se"], "candidate")
    bad = copy.deepcopy(_config("AACV"))
    bad["working_error"]["analysis_variants"][0]["name"] = "undeclared"
    with pytest.raises(ValueError, match="primary"):
        official.load_results(synthetic_dirs["AACV"], "AACV", bad)
    with pytest.raises(FileNotFoundError):
        official.load_results(tmp_path / "missing", "SRCV", _config("SRCV"))
    bad = copy.deepcopy(_config("SRCV"))
    bad["experiment"]["outer_repeats"] = 1
    with pytest.raises(ValueError, match="outer-repeat"):
        official.load_results(synthetic_dirs["SRCV"], "SRCV", bad)


def test_generation_manifest_and_input_integrity(synthetic_dirs, tmp_path, monkeypatch):
    # Low-resolution test renders; official CLI PNGs remain 400 dpi.
    savefig = plt.Figure.savefig

    def quick_save(self, *args, **kwargs):
        kwargs["dpi"] = 60
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", quick_save)
    paths = [p for directory in synthetic_dirs.values() for p in directory.rglob("*.csv")]
    before = {str(p): official.sha256(p) for p in paths}
    manifest = official.generate(synthetic_dirs["SRCV"], synthetic_dirs["AACV"], tmp_path / "figures")
    assert manifest["aacv_variant"] == official.PRIMARY
    assert len(manifest["output_sha256"]) == 20
    for name, digest in manifest["output_sha256"].items():
        assert official.sha256(tmp_path / "figures" / name) == digest
    assert before == {str(p): official.sha256(p) for p in paths}
    saved = json.loads((tmp_path / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    assert saved["input_sha256"] == manifest["input_sha256"]
    assert saved["script_sha256"] == official.sha256(Path(official.__file__))
