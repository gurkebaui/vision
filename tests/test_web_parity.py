"""The browser demo must agree with the Python engine.

`web/gestures.js` is a hand-port of the Python scorer.  Ports drift silently,
so this test feeds identical landmarks to both and fails if they disagree.
Skipped automatically when Node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from synth import POSES, make_state, pose

from gesturekit.gestures.static import score_all

WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

CASES = list(POSES) + ["pinch", "ok"]

SCRIPT = """
import fs from 'fs';
const {scoreAll} = await import(process.argv[1]);
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = {};
for (const [name, pts] of Object.entries(cases)) {
  out[name] = scoreAll(pts.map(([x, y, z]) => ({x, y, z})));
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js_scores(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    cases = {name: [[float(v) for v in p] for p in pose(name)] for name in CASES}
    case_file = tmp / "cases.json"
    case_file.write_text(json.dumps(cases))

    result = subprocess.run(
        [NODE, "--input-type=module", "-e", SCRIPT,
         str(WEB / "gestures.js"), str(case_file)],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=str(WEB),
    )
    if result.returncode != 0:
        pytest.fail(f"node failed: {result.stderr[:600]}")
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", CASES)
def test_same_winner(name, js_scores):
    py = score_all(make_state(pose(name)))
    js = js_scores[name]
    assert max(js, key=js.get) == max(py, key=py.get) == name


@pytest.mark.parametrize("name", CASES)
def test_scores_match_numerically(name, js_scores):
    py = score_all(make_state(pose(name)))
    js = js_scores[name]
    assert set(js) == set(py), "the two implementations know different gestures"
    for gesture, value in py.items():
        assert js[gesture] == pytest.approx(value, abs=0.01), (
            f"{name}/{gesture}: python={value:.4f} js={js[gesture]:.4f}"
        )


def test_web_assets_present():
    for filename in ("index.html", "demo.js", "gestures.js", "style.css"):
        assert (WEB / filename).is_file(), f"missing web/{filename}"


def test_demo_has_no_simulated_fallback():
    """v1's page faked detections with Math.random when the model failed."""
    source = (WEB / "demo.js").read_text()
    assert "Math.random" not in source
    assert "startDemoMode" not in source
