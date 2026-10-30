"""The planners draw exactly the rows the backbones' own samplers drew: torch.randperm, replayed in numpy."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from axis_qvla.signals import planners as pl
from axis_qvla.signals.blocks.schedule import draw_schedule
from axis_qvla.signals.program import Selection
from axis_qvla.signals.rowspace import flat_rowspace
from axis_qvla.signals.schema import Budget
from axis_qvla.signals.testing import TOY_TRAINABLE, toy_rowspace

# sha256 of torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy().astype("<i8").tobytes(),
# recorded with torch 2.7.1+cu126 (the openpi venv); (1319784, 0) and (86912, 100003) were also recorded
# with torch 2.11.0+cu128 (the LeRobot venv) and agree. These are the (n, seed) pairs the paper's plans use.
TORCH_DIGESTS = {
    (1319784, 42): "87977cd243606f101a30a7487a1b2e0211aa3a5db6270e104a7bf12b8ed94c91",
    (1319784, 43): "542ecba53c57e425de019d4963e5aac1809fdc5656407df0514f88ecc8d807f1",
    (1319784, 44): "53dd031f18607b62fabe667208cea1dad31f02b134a0aa920b4ed88d7b2d6b6b",
    (266764, 42): "4699678ac9f0ecdea4cbc56bbe67d3b6720dc81731d1569a3e7af4e970df2524",
    (670239, 42): "bf2cd02df971d6e177f0debdd068449525126d39fdee095fa9eb4e3a43708380",
    (1319784, 0): "9f11b0e30dadc21af6f03a2b806b535f8fbe89d90801c135778c69dcc65b594e",
    (1319784, 100003): "080efa2fae074d7f5d3768ce52c9906827493fec7e80d71e956e440109efa1a1",
    (1319784, 200006): "06370389e72a23fe2b67f846f8092a2c763136e0428d47460f28d74f46c5b056",
    (266764, 0): "ec80c6f626f2554a51431e778ed9182540077ed0c6cf49ce1162706db8296fc0",
    (670239, 0): "3bfcf00dd5236046d64015e10d6b06d78bed6cc6cb45fb8256bba10b5b3916e2",
    (86912, 0): "35325681a7db9f89d319bc74dab7e46ecc4f15f43c9f02996a941b5695e8110c",
    (86912, 1): "e3bd0e06580fef265a8fe19a430ba53a69ee2f627eed0f5fe46477140c4ac648",
    (86912, 2): "b2f2fd94c54863043513f170b3b821bd02cee521dea51c9968a2672ec2646335",
    (86912, 3): "80ef1840d46f591e945cfe57f2f6c69f0d700d2d69c3ee09dbae19feb4b98ecb",
    (86912, 4): "d5a02c88f916e0ab58c32820f06a9449a2db4d35fc6825b1f118de1318d613c3",
    (86912, 100003): "988835fbe41423b859b506936bbeb2525bb958a626f684f9b28301167cdaf121",
    (86912, 100004): "15e923ecf89de68ffc717186d0d81c4df1adf774edda44fa4945a29a28f565e4",
    (86912, 100005): "fce32b846c354f8d5e91e5e1e5e4e4297448f7de9ac5a4d0593658bfb35a705f",
    (86912, 100006): "98a806005bf9c4e6dfcb8e97d92225e2047908626c1a1a31a2ad8c643dc784e9",
    (86912, 100007): "ab8c0d820a78a0bfb9402f552ab42e3c358b8490f5feefb7cbac73cf87f69ec7",
    (86912, 200006): "faba411f20dd6765b2e64a2146a811c93ab05236c5c00674792465bfcdc28971",
    (86912, 200007): "2e477c052fc8df7f438ee3b22caa17d4717564c827152ae3a8910afa0dad5684",
    (86912, 200008): "d4e4a8f39c7957e8fa58f451eca3fbbb61317976b12fbdf30d7f02c0287e0013",
    (86912, 200009): "5ea55822e0dbc3e38c41123850f0172649089b2c192149fa72b04f6290dcfe6e",
    (86912, 200010): "1c1f93f4c5ca0684799e6b76fd35cc0a92a726eb0a339964df7d773454aab7fe",
}
FT_SIZE = [k for k in TORCH_DIGESTS if k[0] == 86912]
CORPUS_SIZE = [k for k in TORCH_DIGESTS if k[0] != 86912]


def _digest(perm) -> str:
    return hashlib.sha256(np.asarray(perm, dtype="<i8").tobytes()).hexdigest()


@pytest.mark.parametrize("n,seed", FT_SIZE)
def test_torch_randperm_matches_torch_at_the_finetune_size(n, seed):
    assert _digest(pl.torch_randperm(n, seed)) == TORCH_DIGESTS[(n, seed)]


@pytest.mark.parametrize("n,seed", CORPUS_SIZE)
def test_torch_randperm_matches_torch_at_the_corpus_sizes(n, seed):
    assert _digest(pl.torch_randperm(n, seed)) == TORCH_DIGESTS[(n, seed)]


def test_small_sizes_and_bad_seeds():
    assert pl.torch_randperm(0, 1).tolist() == [] and pl.torch_randperm(1, 1).tolist() == [0]
    assert sorted(pl.torch_randperm(1000, 7).tolist()) == list(range(1000))
    for bad in (-1, 2**32):
        with pytest.raises(pl.PlanError):
            pl.torch_randperm(10, bad)


@pytest.mark.parametrize("convention,seed_of", [("openpi-rowsampler", lambda s, e: s + e),
                                                ("lerobot-axisrowsampler", lambda s, e: s * 100_003 + e)])
def test_the_stream_is_successive_epoch_permutations(convention, seed_of):
    rows = np.array(TOY_TRAINABLE, dtype=np.int64)
    got = pl.PermutationPlanner(convention).stream(rows, 22, seed=3)
    want = np.concatenate([rows[pl.torch_randperm(9, seed_of(3, e))] for e in range(3)])[:22]
    np.testing.assert_array_equal(got, want)


def test_a_plan_is_the_stream_cut_into_global_batches():
    plan = pl.PermutationPlanner("openpi-rowsampler").plan(toy_rowspace(), None, None, Budget(4, 2), 42)
    np.testing.assert_array_equal(plan.ravel(), np.array(TOY_TRAINABLE)[pl.torch_randperm(9, 42)][:8])
    assert plan.dtype == np.int64 and plan.shape == (4, 2)


def test_the_schedule_planner_is_the_filtering_draw():
    blocks = (np.array([1, 2]), np.array([], dtype=np.int64), np.array([15, 16, 17]))
    row_w = (np.array([2.0, 1.5]), np.array([]), np.array([1.2, 1.0, 1.9]))
    got = pl.SchedulePlanner("drop_top").plan(toy_rowspace(), Selection(blocks, row_w, 9), None, Budget(3, 2), 0)
    want = draw_schedule(list(blocks), mode="drop_top", total_steps=3, batch=2, seed=0)
    np.testing.assert_array_equal(got, want)
    with pytest.raises(pl.PlanError):
        pl.SchedulePlanner("drop_top").plan(toy_rowspace(), None, None, Budget(3, 2), 0)


def test_unknown_conventions_are_refused():
    with pytest.raises(pl.PlanError):
        pl.PermutationPlanner("jax-shuffle")


def test_an_empty_draw_and_an_empty_selection():
    assert pl.PermutationPlanner("openpi-rowsampler").stream(np.array(TOY_TRAINABLE), 0, 1).tolist() == []
    with pytest.raises(pl.PlanError, match="no rows"):
        pl.PermutationPlanner("openpi-rowsampler").plan(toy_rowspace(), Selection((), (), 9), None, Budget(4, 2), 0)


def test_openpi_plans_drop_each_epochs_partial_batch():
    """openpi batches with drop_last=True: 9 rows at batch 2 give 8 per epoch, then the next epoch."""
    rows = np.array(TOY_TRAINABLE, dtype=np.int64)
    plan = pl.PermutationPlanner("openpi-rowsampler").plan(toy_rowspace(), None, None, Budget(5, 2), 42)
    want = np.concatenate([rows[pl.torch_randperm(9, 42)][:8], rows[pl.torch_randperm(9, 43)][:2]])
    np.testing.assert_array_equal(plan.ravel(), want)
    with pytest.raises(pl.PlanError, match="fewer than one global batch"):
        pl.PermutationPlanner("openpi-rowsampler").plan(toy_rowspace(), Selection((np.array([1]),), (np.array([1.0]),), 9),
                                                        None, Budget(2, 2), 0)


def test_lerobot_plans_refuse_a_short_batch_at_an_epoch_boundary():
    """LeRobot batches with drop_last=False: a plan may cross an epoch boundary only on a whole batch."""
    planner, rows = pl.PermutationPlanner("lerobot-axisrowsampler"), np.array(TOY_TRAINABLE, dtype=np.int64)
    with pytest.raises(pl.PlanError, match="not a multiple of the global batch"):
        planner.plan(toy_rowspace(), None, None, Budget(5, 2), 1)
    np.testing.assert_array_equal(planner.plan(toy_rowspace(), None, None, Budget(4, 2), 1).ravel(),
                                  planner.stream(rows, 8, 1))                      # one epoch: no boundary
    four = Selection((np.array([1, 2]), np.array([6, 7])), (np.ones(2), np.ones(2)), 9)
    np.testing.assert_array_equal(planner.plan(toy_rowspace(), four, None, Budget(4, 2), 1).ravel(),
                                  planner.stream(np.array([1, 2, 6, 7]), 8, 1))    # two whole epochs


@pytest.mark.parametrize("convention,seed", [("openpi-rowsampler", 0), ("lerobot-axisrowsampler", 1),
                                             ("lerobot-axisrowsampler", 2)])
def test_the_paper_finetune_plan_is_five_whole_epochs_of_torch_permutations(convention, seed):
    """No paper plan changes with the loaders' batching: 6,790 x 64 = 5 x 86,912 rows, whole batches."""
    rs = flat_rowspace(name="heldout24_ft", n_rows=86_912, source="tests")
    planner = pl.PermutationPlanner(convention)
    plan = planner.plan(rs, None, None, Budget(6790, 64), seed)
    np.testing.assert_array_equal(plan.ravel(), planner.stream(np.arange(86_912), 6790 * 64, seed))
    for e in range(5):
        epoch = plan.ravel()[e * 86_912:(e + 1) * 86_912]
        assert _digest(epoch) == TORCH_DIGESTS[(86_912, pl.CONVENTIONS[convention](seed, e))]
