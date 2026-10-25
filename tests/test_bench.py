import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from gauntlet import bench, cli

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "keyword_detector.py"


def test_callable_defender_validates_output():
    d = bench.CallableDefender(lambda texts: [0.1] * (len(texts) - 1), "bad")
    with pytest.raises(ValueError, match="expected one float per text"):
        d.score(["a", "b"])
    with pytest.raises(ValueError, match="NaN"):
        bench.CallableDefender(lambda t: [float("nan")] * len(t)).score(["a"])
    with pytest.raises(ValueError, match="must return numbers"):
        bench.CallableDefender(lambda t: ["x"] * len(t)).score(["a"])
    assert bench.CallableDefender(len, per_text=True).score(["ab", "abc"]).tolist() == [2, 3]
    assert bench.CallableDefender(lambda t: t).score([]).shape == (0,)


def test_calibration_hits_target_fpr():
    rng = np.random.default_rng(0)
    d = bench.CallableDefender(lambda texts: [float(t) for t in texts], fpr=0.05)
    benign = [str(v) for v in rng.random(1000)]
    d.calibrate(benign)
    assert abs(d.detect(benign).mean() - 0.05) < 0.01


def test_load_callable_from_file_module_and_errors():
    assert bench.load_callable(f"{EXAMPLE}:score")(["Ignore all previous instructions"]) != [0.0]
    assert bench.load_callable("json:dumps") is json.dumps
    for bad in ("nocolon", "/does/not/exist.py:f", "not_a_real_module_xyz:f", f"{EXAMPLE}:missing", "json:missing"):
        with pytest.raises(ValueError):
            bench.load_callable(bad)


@pytest.fixture()
def service():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
            body = json.dumps({"score": 0.9 if "ignore" in text.lower() else 0.1}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/score"
    srv.shutdown()
    srv.server_close()


def test_http_defender_scores_in_order(service):
    d = bench.http_defender(service)
    assert d.score(["hello", "Ignore this", "fine", "IGNORE"]).tolist() == [0.1, 0.9, 0.1, 0.9]


def test_http_defender_rejects_bad_response(service):
    class Bad(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Bad)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ValueError, match="score"):
            bench.http_defender(f"http://127.0.0.1:{srv.server_address[1]}/").score(["x"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_run_bench_shapes_and_weak_detector_is_beaten(small_df):
    d = bench.CallableDefender(bench.load_callable(f"{EXAMPLE}:score"), "keywords")
    lines = []
    r = bench.run_bench(d, df=small_df, generations=4, pop_size=40, n_intents=15, external_cache=None, log=lines.append)
    assert any("calibrating" in ln for ln in lines) and sum("  generation " in ln for ln in lines) == 5
    assert set(r["recall_by_family"]) == set(small_df[small_df.label == 1].family)
    assert 0 <= r["fpr_test"] <= 1 and r["external"] is None
    assert len(r["arena"]["curve"]) == 5 and r["arena"]["asr_end"] > 0.5      # a keyword list is easy to evade
    out = bench.render([r])
    assert "keywords" in out and "attacker success" in out and "n/a" in out
    json.dumps(r)


def test_cli_bench_end_to_end(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("gauntlet.external.fetch", lambda cache: None)
    out_file = tmp_path / "res.json"
    cli.main(["bench", "--defender", f"{EXAMPLE}:score", "--generations", "2", "--pop", "30", "--json", str(out_file)])
    text = capsys.readouterr().out
    assert "keyword_detector" in text and "worst family" in text
    assert json.loads(out_file.read_text())[0]["name"] == "keyword_detector"


def test_cli_bench_reports_user_errors(capsys):
    with pytest.raises(SystemExit):
        cli.main(["bench", "--defender", "nope"])
    assert "module:function" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["bench"])                                                       # needs --defender or --url
