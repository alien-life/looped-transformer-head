"""End-to-end smoke test on the offline tiny model: data -> train (curriculum) -> depth sweep."""

import json
import random

import torch

from lth import data, evaluate, probe, train
from lth.evaluate import mcnemar, wilson


def test_statistics():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 0) == (0.0, 1.0)
    assert mcnemar([True] * 10, [True] * 10) == 1.0
    # 9 discordant pairs, all in one direction: p = 2 * 0.5**9
    assert abs(mcnemar([False] * 9 + [True], [True] * 10) - 2 * 0.5**9) < 1e-12
    assert mcnemar([True, False] * 5, [False, True] * 5) == 1.0


def test_step_plan():
    rng = random.Random(0)
    assert train.step_plan(train.TIERS["5x"], rng) == (5, [5])
    seen = set()
    for _ in range(500):
        loops, exits = train.step_plan(train.TIERS["50x"], rng)
        assert 1 <= loops <= 50 and exits[-1] == loops
        assert set(exits[:-1]) == {e for e in (5, 10, 25, 50) if e < loops}
        seen.add(loops)
    assert min(seen) == 1 and max(seen) == 50


def test_train_and_evaluate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = tmp_path / "data"
    data.main(
        [
            "--out",
            str(d),
            "--train",
            "48",
            "--val",
            "8",
            "--test",
            "12",
            "--hops",
            "1-2",
            "--ood-hops",
            "3-3",
            "--ood",
            "6",
            "--tasks",
            "lookup_chain",
        ]
    )
    assert {p.name for p in d.iterdir()} == {
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "test_ood.jsonl",
        "dataset_card.json",
    }

    common = [
        "--model",
        "tiny-random",
        "--data",
        str(d),
        "--batch-size",
        "4",
        "--eval-every",
        "4",
        "--eval-examples",
        "4",
        "--log-every",
        "2",
        "--warmup",
        "2",
        "--num-heads",
        "4",
        "--max-loops",
        "8",
    ]
    one, five = tmp_path / "1x", tmp_path / "5x"
    train.main(common + ["--out", str(one), "--tier", "1x", "--steps", "4"])
    cache = tmp_path / "hidden.pt"
    train.main(
        common
        + [
            "--out",
            str(five),
            "--tier",
            "5x",
            "--steps",
            "6",
            "--cache-hidden",
            str(cache),
        ]
    )
    assert (
        (cache / "key.json").exists()
        and (cache / "h.npy").exists()
        and not (cache / "final.npy").exists()
    )
    two = tmp_path / "2x"
    train.main(
        common
        + [
            "--out",
            str(two),
            "--tier",
            "2x",
            "--steps",
            "2",
            "--cache-hidden",
            str(cache),
        ]
    )
    assert torch.load(two / "head.pt", weights_only=True)["meta"]["train_loops"] == 2
    cont = tmp_path / "5x-continued"
    train.main(
        common
        + [
            "--out",
            str(cont),
            "--tier",
            "5x",
            "--steps",
            "2",
            "--init-from",
            str(five / "head.pt"),
        ]
    )

    ckpt = torch.load(five / "head.pt", weights_only=True)
    assert ckpt["meta"]["train_loops"] == 5 and ckpt["meta"]["steps"] == 6
    assert ckpt["config"]["gate"] == "matrix"
    assert ckpt["config"]["norm_out"] is True and ckpt["config"]["qk_norm"] is True
    assert (
        ckpt["state_dict"]["proj_out.weight"].abs().sum() > 0
    )  # the gate moved off zero
    cfg_json = json.loads((five / "config.json").read_text())
    assert cfg_json["steps"] == 6
    log = [json.loads(line) for line in open(five / "train_log.jsonl")]
    assert any("val_loss" in row for row in log) and any("loss" in row for row in log)

    # explicit scalar gate + legacy flags still work and move alpha off zero
    scalar_dir = tmp_path / "scalar"
    train.main(
        common
        + [
            "--out",
            str(scalar_dir),
            "--tier",
            "2x",
            "--steps",
            "4",
            "--gate",
            "scalar",
            "--no-norm-out",
            "--no-qk-norm",
        ]
    )
    s_ckpt = torch.load(scalar_dir / "head.pt", weights_only=True)
    assert (
        s_ckpt["config"]["gate"] == "scalar"
        and s_ckpt["state_dict"]["alpha"].abs().sum() > 0
    )
    assert "norm_out.weight" not in s_ckpt["state_dict"]
    assert "layers.0.q_norm.weight" not in s_ckpt["state_dict"]

    out = tmp_path / "report"
    code = evaluate.main(
        [
            "--model",
            "tiny-random",
            "--data",
            f"test={d / 'test.jsonl'}",
            "--data",
            f"ood={d / 'test_ood.jsonl'}",
            "--head",
            f"1x={one / 'head.pt'}",
            "--head",
            f"5x={five / 'head.pt'}",
            "--control",
            "1x",
            "--depths",
            "0,1,5,8",
            "--out",
            str(out),
            "--batch-size",
            "4",
            "--max-new-tokens",
            "6",
            "--latency-prompts",
            "2",
            "--latency-tokens",
            "4",
        ]
    )
    assert code == 0  # not --strict
    report = json.loads((out / "report.json").read_text())
    assert set(report["accuracy"]) == {"test", "ood"}
    assert set(report["accuracy"]["test"]["heads"]["5x"]) == {"1", "5", "8"}
    assert report["accuracy"]["test"]["baseline"]["n"] == 12
    criteria = {c["criterion"] for c in report["heads"]["5x"]["acceptance"]}
    assert {
        "beats frozen baseline",
        "monotonic up to trained depth",
        "looping beats single-pass control",
        "stable beyond trained depth",
        "no regression on general text",
    } <= criteria
    assert set(report["latency"]["heads"]["5x"]["depths"]) == {"1", "5", "8"}
    assert report["regression"]["5x"]["baseline_ppl"] > 1
    assert "## Acceptance" in (out / "report.md").read_text()
    assert len(list((out / "predictions").iterdir())) == 2 * (1 + 3 + 3)

    # An untrained random backbone cannot pass, so --strict must fail the run.
    assert (
        evaluate.main(
            [
                "--model",
                "tiny-random",
                "--data",
                str(d / "test.jsonl"),
                "--head",
                str(five / "head.pt"),
                "--depths",
                "0,5",
                "--out",
                str(tmp_path / "strict"),
                "--latency-prompts",
                "0",
                "--no-regression",
                "--max-new-tokens",
                "4",
                "--strict",
            ]
        )
        == 1
    )


def test_chain_supervision_and_probe_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = tmp_path / "names"
    data.main(
        [
            "--out",
            str(d),
            "--train",
            "32",
            "--val",
            "8",
            "--test",
            "12",
            "--hops",
            "1-3",
            "--tasks",
            "name_chain",
        ]
    )
    common = [
        "--model",
        "tiny-random",
        "--data",
        str(d),
        "--batch-size",
        "4",
        "--eval-every",
        "4",
        "--steps",
        "4",
        "--eval-examples",
        "4",
        "--log-every",
        "2",
        "--warmup",
        "2",
        "--num-heads",
        "4",
        "--max-loops",
        "8",
    ]
    chain, plain = tmp_path / "chain", tmp_path / "plain"
    train.main(
        common
        + [
            "--out",
            str(chain),
            "--tier",
            "3x",
            "--chain-supervision",
            "--cache-hidden",
            str(tmp_path / "h.pt"),
        ]
    )
    train.main(common + ["--out", str(plain), "--tier", "3x"])
    assert (
        torch.load(chain / "head.pt", weights_only=True)["meta"]["chain_supervision"]
        is True
    )
    assert (
        torch.load(plain / "head.pt", weights_only=True)["meta"]["chain_supervision"]
        is False
    )

    out = tmp_path / "probe"
    probe.main(
        [
            "--model",
            "tiny-random",
            "--data",
            str(d / "test.jsonl"),
            "--out",
            str(out),
            "--head",
            f"chain={chain / 'head.pt'}",
            "--head",
            f"plain={plain / 'head.pt'}",
            "--batch-size",
            "4",
        ]
    )
    results = json.loads((out / "probe.json").read_text())
    assert len(results) == 3 and "frozen backbone" in results
    rows = next(v for k, v in results.items() if k.startswith("chain"))
    assert {(r["hops"], r["loop"]) for r in rows} == {
        (h, l) for h in (1, 2, 3) for l in (1, 2, 3)
    }
    targets = {(r["hops"], r["loop"]): r["target"] for r in rows}
    assert (
        targets[3, 1] == "hop 1"
        and targets[3, 2] == "hop 2"
        and targets[3, 3] == "answer"
    )
    assert (
        targets[2, 1] == "hop 1"
        and targets[2, 2] == "answer"
        and targets[1, 1] == "answer"
    )
    three_hop = next(r for r in rows if (r["hops"], r["loop"]) == (3, 2))
    assert abs(sum(three_hop["holds"].values()) - 1) < 1e-9
    assert not next(r for r in rows if r["hops"] == 1)[
        "holds"
    ]  # single hop: no entity token to locate
    assert all(r["loop"] == 0 for r in results["frozen backbone"])

    # a head that reads an earlier backbone layer: cached and live training, then evaluation and probe
    tap, tap_live = tmp_path / "tap", tmp_path / "tap-live"
    train.main(
        common
        + [
            "--out",
            str(tap),
            "--tier",
            "2x",
            "--chain-supervision",
            "--tap-layer",
            "1",
            "--cache-hidden",
            str(tmp_path / "h-tap.pt"),
        ]
    )
    train.main(
        common
        + [
            "--out",
            str(tap_live),
            "--tier",
            "2x",
            "--tap-layer",
            "1",
            "--gate",
            "matrix",
        ]
    )
    live = torch.load(tap_live / "head.pt", weights_only=True)
    assert (
        live["config"]["gate"] == "matrix"
        and live["state_dict"]["proj_out.weight"].abs().sum() > 0
    )
    ckpt = torch.load(tap / "head.pt", weights_only=True)
    assert ckpt["config"]["tap_layer"] == 1 and ckpt["config"]["standardize"] is True
    assert (
        ckpt["state_dict"]["in_std"].min() > 0
        and ckpt["state_dict"]["in_mean"].abs().sum() > 0
    )
    import numpy as np

    read, final = (
        np.load(tmp_path / "h-tap.pt" / "h.npy"),
        np.load(tmp_path / "h-tap.pt" / "final.npy"),
    )
    assert (
        read.dtype == np.float16
        and 0 < final.shape[0] < read.shape[0]
        and np.isfinite(read).all()
    )
    import pytest

    with pytest.raises(
        SystemExit
    ):  # the plain cache was built for the final layer, not layer 1
        train.main(
            common
            + [
                "--out",
                str(tmp_path / "x"),
                "--tier",
                "2x",
                "--tap-layer",
                "1",
                "--cache-hidden",
                str(tmp_path / "h.pt"),
            ]
        )
    assert (
        evaluate.main(
            [
                "--model",
                "tiny-random",
                "--data",
                str(d / "test.jsonl"),
                "--head",
                f"tap={tap / 'head.pt'}",
                "--depths",
                "0,1,2",
                "--out",
                str(tmp_path / "tap-report"),
                "--max-new-tokens",
                "4",
                "--latency-prompts",
                "2",
                "--latency-tokens",
                "3",
            ]
        )
        == 0
    )
    probe.main(
        [
            "--model",
            "tiny-random",
            "--data",
            str(d / "test.jsonl"),
            "--out",
            str(tmp_path / "tap-probe"),
            "--head",
            f"tap={tap / 'head.pt'}",
            "--batch-size",
            "4",
        ]
    )

    # sampled-depth tiers cannot be chain-supervised
    with pytest.raises(SystemExit):
        train.main(
            common
            + ["--out", str(tmp_path / "bad"), "--tier", "50x", "--chain-supervision"]
        )


def test_hidden_cache_matches_live_states_and_survives_interrupted_builds(tmp_path):
    from lth.model import LoopedHeadModel
    from lth.tiny import build_tiny

    base, tok = build_tiny(device="cpu")
    model = LoopedHeadModel(base, num_heads=4, max_loops=4)
    enc = data.Encoder(tok)
    exs = data.generate(6, (1, 2), 0, "train", ["name_chain"])
    path = tmp_path / "cache"
    cache = train.HiddenCache(
        model, enc, exs, path, "tiny-random", "none", batch_size=4
    )
    batch = enc.train_batch(exs[:3])
    h_base, h_in = cache.batch([0, 1, 2], batch["input_ids"].shape[1])
    live, _ = model.encode(batch["input_ids"], batch["attention_mask"])
    mask = batch["attention_mask"].bool()
    assert h_in is None and torch.allclose(
        h_base.float()[mask], live[mask], atol=2e-2
    )  # fp16 on disk
    mean, std = cache.stats(chunk=7)
    flat = torch.cat(
        [
            torch.from_numpy(
                cache.h[cache.offsets[i] : cache.offsets[i + 1]].astype("float64")
            )
            for i in range(6)
        ]
    )
    assert torch.allclose(mean, flat.mean(0).float(), atol=1e-4) and torch.allclose(
        std, flat.std(0, unbiased=False).float(), atol=1e-4
    )
    # a build that never wrote key.json (interrupted) is rebuilt, not trusted
    (path / "key.json").unlink()
    (path / "h.npy").write_bytes(b"garbage")
    again = train.HiddenCache(model, enc, exs, path, "tiny-random", "none")
    assert torch.equal(again.batch([0, 1, 2], batch["input_ids"].shape[1])[0], h_base)
