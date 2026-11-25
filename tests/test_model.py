import pytest
import torch

from lth.model import LoopedHeadModel
from lth.tiny import build_tiny


@pytest.fixture()
def model():
    base, _ = build_tiny(device="cpu")
    m = LoopedHeadModel(base, num_heads=4, num_layers=2, max_loops=12)
    torch.manual_seed(0)
    with torch.no_grad():
        m.head.alpha.fill_(0.7)
    return m.eval()


def ids(b=2, t=10, seed=0):
    return torch.randint(4, 100, (b, t), generator=torch.Generator().manual_seed(seed))


def test_backbone_is_frozen_and_parity_holds(model):
    assert not any(p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert model.verify_parity() < 1e-4
    x = ids()
    ref = model.base(input_ids=x).logits
    assert torch.allclose(model(x, num_loops=0), ref, atol=1e-5)
    with torch.no_grad():
        model.head.alpha.zero_()
    assert torch.allclose(model(x, num_loops=7), ref, atol=1e-5)


def test_backbone_stays_in_eval_mode(model):
    model.train()
    assert model.head.training and not model.base.training


def test_loss_trains_only_the_head(model):
    x = ids()
    labels = x.clone()
    labels[:, :6] = -100
    out = model(x, num_loops=6, labels=labels, exits=[3, 6], bptt_window=2, use_checkpoint=True)
    assert sorted(out.exit_losses) == [3, 6]
    out.loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert model.head.alpha.grad.abs().sum() > 0

    # The masked loss equals cross-entropy over the supervised positions of full logits.
    logits = model(x, num_loops=6)
    ref = torch.nn.functional.cross_entropy(logits[:, 5:-1].reshape(-1, logits.shape[-1]), x[:, 6:].reshape(-1))
    single = model(x, num_loops=6, labels=labels)
    assert torch.allclose(single.loss, ref, atol=1e-5)


@pytest.mark.parametrize("loops", [0, 1, 6])
def test_cached_generation_matches_uncached(model, loops):
    x = ids(b=1, t=6)
    got = model.generate(x, num_loops=loops, max_new_tokens=8).tokens[0].tolist()
    seq = x.clone()
    want = []
    for _ in range(8):
        nxt = model(seq, num_loops=loops)[:, -1].argmax(-1)
        want.append(int(nxt))
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    assert got == want


def test_batched_left_padded_generation_matches_single(model):
    a, b = ids(b=1, t=9, seed=1), ids(b=1, t=5, seed=2)
    single = [model.generate(x, num_loops=5, max_new_tokens=6).tokens[0].tolist() for x in (a, b)]
    batch = torch.zeros(2, 9, dtype=torch.long)
    mask = torch.zeros(2, 9, dtype=torch.long)
    batch[0], mask[0] = a[0], 1
    batch[1, 4:], mask[1, 4:] = b[0], 1
    out = model.generate(batch, mask, num_loops=5, max_new_tokens=6).tokens.tolist()
    assert out == single


def test_eos_stops_generation(model):
    x = ids(b=1, t=6)
    first = int(model.generate(x, num_loops=3, max_new_tokens=1).tokens[0, 0])
    out = model.generate(x, num_loops=3, max_new_tokens=8, eos_ids=[first])
    assert out.tokens.shape[1] == 1
    assert model.generate(x, num_loops=3, max_new_tokens=8, eos_ids=[first], ignore_eos=True).tokens.shape[1] == 8


def test_head_checkpoint_roundtrip(model, tmp_path):
    x = ids()
    ref = model(x, num_loops=4)
    path = tmp_path / "head.pt"
    model.save_head(path, {"train_loops": 4})
    base, _ = build_tiny(device="cpu")
    other = LoopedHeadModel(base, num_heads=4, num_layers=1, max_loops=12)
    assert other.load_head(path) == {"train_loops": 4}
    assert torch.allclose(other(x, num_loops=4), ref, atol=1e-6)


def test_precomputed_backbone_states_give_the_same_loss(model):
    x = ids()
    labels = x.clone()
    labels[:, :6] = -100
    h, _ = model.encode(x)
    live = model(x, num_loops=4, labels=labels).loss
    cached = model(x, num_loops=4, labels=labels, h_base=h).loss
    assert torch.equal(live, cached)


def test_per_exit_labels(model):
    x = ids()
    final = x.clone()
    final[:, :6] = -100
    hop = torch.full_like(x, -100)
    hop[:, 6] = 17  # one supervised token at an intermediate exit
    empty = torch.full_like(x, -100)
    out = model(x, num_loops=3, labels={1: hop, 2: empty, 3: final}, exits=[1, 2, 3])
    assert sorted(out.exit_losses) == [1, 3]  # an exit with nothing to supervise is skipped
    logits = model(x, num_loops=3, exits=[1, 2, 3])
    ce = torch.nn.functional.cross_entropy
    want1 = ce(logits[1][:, 5], torch.full((2,), 17))
    want3 = ce(logits[3][:, 5:-1].reshape(-1, logits[3].shape[-1]), x[:, 6:].reshape(-1))
    assert torch.allclose(out.exit_losses[1], want1, atol=1e-5)
    assert torch.allclose(out.exit_losses[3], want3, atol=1e-5)
    assert torch.allclose(out.loss, (want1 + want3) / 2, atol=1e-5)


@pytest.fixture()
def tap_model():
    from lth.head import HeadConfig
    base, _ = build_tiny(device="cpu")
    m = LoopedHeadModel(base, HeadConfig(hidden_size=64, num_heads=4, num_layers=2, max_loops=12,
                                         tap_layer=1, standardize=True))
    torch.manual_seed(0)
    with torch.no_grad():
        m.head.alpha.fill_(0.7)
        m.head.set_input_stats(torch.randn(64) * 0.1, torch.rand(64) + 0.5)
    return m.eval()


def test_tapped_head_reads_an_earlier_layer(tap_model):
    x = ids()
    h_base, h_in, _ = tap_model.encode_states(x)
    want = tap_model.base.base_model(input_ids=x, output_hidden_states=True).hidden_states[1]
    assert torch.equal(h_in, want) and not torch.allclose(h_in, h_base)
    assert not torch.allclose(tap_model(x, num_loops=3), tap_model.base(input_ids=x).logits, atol=1e-3)
    assert torch.allclose(tap_model(x, num_loops=0), tap_model.base(input_ids=x).logits, atol=1e-5)


def test_tapped_cached_generation_matches_uncached(tap_model):
    x = ids(b=1, t=6)
    got = tap_model.generate(x, num_loops=4, max_new_tokens=8).tokens[0].tolist()
    seq, want = x.clone(), []
    for _ in range(8):
        nxt = tap_model(seq, num_loops=4)[:, -1].argmax(-1)
        want.append(int(nxt))
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    assert got == want


def test_tapped_loss_needs_final_states_only_at_supervised_positions(tap_model):
    x = ids()
    labels = x.clone()
    labels[:, :6] = -100
    h_base, h_in, _ = tap_model.encode_states(x)
    live = tap_model(x, num_loops=3, labels=labels).loss
    sparse = torch.zeros_like(h_base)
    sparse[:, 5:-1] = h_base[:, 5:-1]  # positions whose output predicts a supervised token
    assert torch.allclose(tap_model(x, num_loops=3, labels=labels, h_base=sparse, h_in=h_in).loss, live, atol=1e-6)


def test_tapped_head_checkpoint_roundtrip(tap_model, tmp_path):
    x = ids()
    ref = tap_model(x, num_loops=4)
    tap_model.save_head(tmp_path / "head.pt")
    base, _ = build_tiny(device="cpu")
    other = LoopedHeadModel.bare(base)
    other.load_head(tmp_path / "head.pt")
    assert other.head.config.tap_layer == 1
    assert torch.allclose(other(x, num_loops=4), ref, atol=1e-6)
