import pytest
import torch

from lth.head import HeadConfig, LoopedHead


def make_head(alpha=0.5, **kw):
    torch.manual_seed(0)
    cfg = HeadConfig(
        **{"hidden_size": 32, "num_heads": 4, "num_layers": 2, "max_loops": 12, **kw}
    )
    head = LoopedHead(cfg).double()
    with torch.no_grad():
        head.alpha.fill_(alpha)
    return head


def hidden(b=2, t=7, d=32, seed=1):
    return torch.randn(
        b, t, d, dtype=torch.double, generator=torch.Generator().manual_seed(seed)
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"head_dim": 16},
        {"gate": "vector"},
        {"num_layers": 1},
        {"gate": "matrix"},
        {"gate": "matrix", "head_dim": 16},
        {"gate": "matrix", "norm_out": True, "qk_norm": True},
        {"norm_out": True, "qk_norm": True},
    ],
)
def test_exact_parity_at_init(kw):
    head = make_head(alpha=0.0, **kw)
    h = hidden()
    for loops in (0, 1, 5, 12):
        assert torch.equal(head(h, loops).h_out, h)


def test_causal():
    head = make_head()
    h = hidden()
    out = head(h, 6).h_out
    h2 = h.clone()
    h2[:, 4:] += 1.0
    out2 = head(h2, 6).h_out
    assert torch.allclose(out[:, :4], out2[:, :4], atol=1e-12)
    assert not torch.allclose(out[:, 4:], out2[:, 4:], atol=1e-6)


def test_loops_change_output():
    head = make_head()
    h = hidden()
    assert not torch.allclose(head(h, 1).h_out, head(h, 5).h_out, atol=1e-6)


def test_incremental_cache_matches_parallel():
    """Zero train/inference mismatch: token-by-token decoding == one parallel pass."""
    head = make_head()
    h = hidden(t=9)
    full = head(h, 7).h_out
    out = head(h[:, :4], 7, return_cache=True)
    pieces = [out.h_out]
    cache = out.cache
    for t in range(4, 9):
        out = head(h[:, t : t + 1], 7, cache=cache, return_cache=True)
        cache = out.cache
        pieces.append(out.h_out)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-10)
    # The cache holds one K/V entry per position, independent of the loop count.
    assert cache.k[0].shape[2] == 9 and cache.v[0].shape[2] == 9


def test_last_only_matches_full():
    head = make_head()
    h = hidden()
    assert torch.allclose(
        head(h, 8, last_only=True).h_out, head(h, 8).h_out[:, -1:], atol=1e-10
    )


def test_left_padding_invariance():
    head = make_head()
    h = hidden(b=1, t=5)
    ref = head(h, 6).h_out
    pad = torch.randn(1, 3, 32, dtype=torch.double)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    out = head(torch.cat([pad, h], dim=1), 6, mask).h_out
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 3:], ref, atol=1e-10)


def test_keys_values_computed_once():
    head = make_head()
    calls = {"k": 0, "q": 0}
    head.layers[0].k_proj.register_forward_hook(
        lambda *_: calls.__setitem__("k", calls["k"] + 1)
    )
    head.layers[0].q_proj.register_forward_hook(
        lambda *_: calls.__setitem__("q", calls["q"] + 1)
    )
    head(hidden(), 10)
    assert calls == {"k": 1, "q": 10}


def test_exits_match_shorter_runs():
    head = make_head()
    h = hidden()
    out = head(h, 10, exits=[2, 5, 10])
    assert sorted(out.exits) == [2, 5, 10]
    for e in (2, 5, 10):
        assert torch.allclose(out.exits[e], head(h, e).h_out, atol=1e-10)


def grads(head, h, **kw):
    head.zero_grad()
    out = head(h, 10, **kw)
    sum(o.pow(2).sum() for o in out.exits.values()).backward()
    return {n: p.grad.clone() for n, p in head.named_parameters() if p.grad is not None}


def test_gate_gradients_at_init():
    """At alpha == 0 only the gate receives gradient; the block wakes up once alpha moves."""
    head = make_head(alpha=0.0)
    g = grads(head, hidden())
    assert g["alpha"].abs().sum() > 0
    assert all(v.abs().sum() == 0 for n, v in g.items() if n != "alpha")


def test_checkpointing_gives_identical_gradients():
    head = make_head()
    h = hidden()
    a, b = grads(head, h), grads(head, h, use_checkpoint=True)
    assert a.keys() == b.keys()
    for n in a:
        assert torch.allclose(a[n], b[n], atol=1e-10), n


def test_truncated_bptt():
    head = make_head()
    h = hidden()
    full = grads(head, h)
    wide = grads(head, h, bptt_window=10)
    for n in full:
        assert torch.allclose(full[n], wide[n], atol=1e-10), n

    trunc = grads(head, h, bptt_window=3)
    # Truncation changes the gradient ...
    assert not torch.allclose(
        full["layers.0.q_proj.weight"], trunc["layers.0.q_proj.weight"], atol=1e-8
    )
    # ... but pass-1 keys/values still train even though pass 1 is outside the window.
    for name in (
        "layers.0.k_proj.weight",
        "layers.0.v_proj.weight",
        "layers.1.k_proj.weight",
    ):
        assert trunc[name].abs().sum() > 0, name
    # Step embeddings of detached passes get no gradient; pass 1 (K/V) and the window do.
    step = trunc["step_embed.weight"].abs().sum(dim=1)
    assert step[0] > 0 and (step[1:7] == 0).all() and (step[7:10] > 0).all()

    # Deep supervision: every exit owns a window.
    deep = grads(head, h, bptt_window=2, exits=[5, 10])
    step = deep["step_embed.weight"].abs().sum(dim=1)
    assert (step[[3, 4, 8, 9]] > 0).all() and (step[[1, 2, 5, 6, 7]] == 0).all()


def test_argument_validation():
    head = make_head()
    h = hidden()
    with pytest.raises(ValueError):
        head(h, 13)
    with pytest.raises(ValueError):
        head(h, 10, exits=[5])
    with pytest.raises(ValueError):
        head(h, 4, torch.ones(2, 3))
    with pytest.raises(ValueError):
        HeadConfig(hidden_size=30, num_heads=4).validate()


def make_tap_head(alpha=0.5):
    head = make_head(alpha=alpha, tap_layer=1, standardize=True)
    head.set_input_stats(
        torch.linspace(-1, 1, 32, dtype=torch.double),
        torch.linspace(0.5, 3, 32, dtype=torch.double),
    )
    return head


def test_tap_layer_reads_h_in_and_writes_to_h_base():
    head = make_tap_head()
    h, src = hidden(seed=1), hidden(seed=2)
    out = head(h, 5, h_in=src).h_out
    # the block reads only h_in: changing h_base shifts the output by exactly that change
    assert torch.allclose(head(h + 1.0, 5, h_in=src).h_out, out + 1.0, atol=1e-10)
    assert not torch.allclose(head(h, 5, h_in=src + 1.0).h_out, out, atol=1e-6)
    with pytest.raises(ValueError):
        head(h, 5)  # a tapped head must be given the tapped states
    assert torch.equal(
        make_tap_head(alpha=0.0)(h, 5, h_in=src).h_out, h
    )  # parity at init
    assert "in_mean" in head.state_dict() and "in_mean" not in make_head().state_dict()


def test_tap_layer_incremental_cache_matches_parallel():
    head = make_tap_head()
    h, src = hidden(t=9, seed=1), hidden(t=9, seed=2)
    full = head(h, 6, h_in=src).h_out
    out = head(h[:, :4], 6, h_in=src[:, :4], return_cache=True)
    pieces, cache = [out.h_out], out.cache
    for t in range(4, 9):
        out = head(
            h[:, t : t + 1], 6, h_in=src[:, t : t + 1], cache=cache, return_cache=True
        )
        cache = out.cache
        pieces.append(out.h_out)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-10)
    h2, src2 = h.clone(), src.clone()
    h2[:, 5:] += 1.0
    src2[:, 5:] += 1.0
    assert torch.allclose(
        head(h2, 6, h_in=src2).h_out[:, :5], full[:, :5], atol=1e-12
    )  # still causal


def test_matrix_gate():
    torch.manual_seed(0)
    head = LoopedHead(
        HeadConfig(
            hidden_size=32, num_heads=4, num_layers=2, max_loops=12, gate="matrix"
        )
    ).double()
    assert "alpha" not in dict(
        head.named_parameters()
    )  # fixed at 1: the output projection is the gate
    assert head.proj_out.weight.abs().sum() == 0
    h = hidden()
    head(h, 5).h_out.pow(2).sum().backward()
    g = {n: p.grad for n, p in head.named_parameters() if p.grad is not None}
    # at init only the output projection learns; the block wakes up once it is non-zero
    assert g["proj_out.weight"].abs().sum() > 0
    assert all(v.abs().sum() == 0 for n, v in g.items() if n != "proj_out.weight")
    with torch.no_grad():
        head.proj_out.weight.normal_(std=0.1)
    assert not torch.allclose(head(h, 5).h_out, h, atol=1e-6)
    full = head(h, 6).h_out
    out = head(h[:, :4], 6, return_cache=True)
    step = head(h[:, 4:], 6, torch.ones(2, 7), cache=out.cache).h_out
    assert torch.allclose(torch.cat([out.h_out, step], dim=1), full, atol=1e-10)


def test_norm_out_and_qk_norm_cache_and_depth_stability():
    torch.manual_seed(0)
    cfg = HeadConfig(
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        max_loops=20,
        gate="matrix",
        norm_out=True,
        qk_norm=True,
    )
    head = LoopedHead(cfg).double()
    with torch.no_grad():
        head.proj_out.weight.normal_(std=0.05)
    h = hidden(t=9)
    # incremental cache matches parallel pass
    full = head(h, 6).h_out
    out = head(h[:, :4], 6, return_cache=True)
    pieces, cache = [out.h_out], out.cache
    for t in range(4, 9):
        out = head(h[:, t : t + 1], 6, cache=cache, return_cache=True)
        cache = out.cache
        pieces.append(out.h_out)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-10)

    # norm_out keeps the emitted delta scale bounded across loop depths R=1..20
    deltas = [
        (head(h, r).h_out - h).pow(2).mean().sqrt().item() for r in (1, 5, 10, 20)
    ]
    assert max(deltas) / min(deltas) < 2.0

    # qk_norm keeps attention scores finite and non-saturated under extreme input outliers
    h_big = h.clone()
    h_big[..., 0] *= 1e4
    out_big = head(h_big, 4).h_out
    assert torch.isfinite(out_big).all()
    assert torch.allclose(
        head(h_big, 4, last_only=True).h_out, out_big[:, -1:], atol=1e-10
    )

    # use_rope=False (NoPE) matches between parallel, cached, last_only, and checkpointed paths
    nope = LoopedHead(
        HeadConfig(
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            max_loops=12,
            gate="matrix",
            norm_out=True,
            qk_norm=True,
            use_rope=False,
        )
    ).double()
    with torch.no_grad():
        nope.proj_out.weight.normal_(std=0.05)
    full_nope = nope(h, 5).h_out
    assert torch.allclose(
        nope(h, 5, last_only=True).h_out, full_nope[:, -1:], atol=1e-10
    )
    pre = nope(h[:, :5], 5, return_cache=True)
    tail = nope(h[:, 5:], 5, cache=pre.cache).h_out
    assert torch.allclose(torch.cat([pre.h_out, tail], dim=1), full_nope, atol=1e-10)
    assert torch.allclose(nope(h, 5, use_checkpoint=True).h_out, full_nope, atol=1e-10)
