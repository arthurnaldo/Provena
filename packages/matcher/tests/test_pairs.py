import numpy as np

from matcher import vote_pairs


def test_vote_pairs_finds_aligned_pair_and_range():
    # Video 0 frames 0..9 (s) match video 1 at +20.5 s; plus scattered hits to video 2 at random shifts.
    q_off = np.arange(10, dtype=float)
    rng = np.random.default_rng(0)
    q_frame = np.concatenate([np.arange(10), np.arange(10)])
    q_video = np.zeros(20, int)
    h_video = np.array([1] * 10 + [2] * 10)
    h_off = np.concatenate([q_off + 20.5, rng.uniform(0, 100, 10)])
    p = vote_pairs(q_frame, q_video, np.concatenate([q_off, q_off]), h_video, h_off, min_votes=5)
    assert p["a"].tolist() == [0] and p["b"].tolist() == [1]
    assert p["votes"][0] == 10
    assert abs(p["shift"][0] - 20.5) < 1e-9
    assert (p["a_start"][0], p["a_end"][0]) == (0, 9)
    assert (p["b_start"][0], p["b_end"][0]) == (20.5, 29.5)


def test_vote_pairs_empty():
    e = np.array([], int)
    assert len(vote_pairs(e, e, e.astype(float), e, e.astype(float))["a"]) == 0
