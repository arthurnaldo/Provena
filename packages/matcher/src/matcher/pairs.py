import numpy as np


def vote_pairs(q_frame: np.ndarray, q_video: np.ndarray, q_offset: np.ndarray,
               h_video: np.ndarray, h_offset: np.ndarray, min_votes: int = 8) -> dict[str, np.ndarray]:
    """Hough voting for many (query video, other video) pairs at once: `match` vectorised for audits.

    Inputs are parallel arrays, one entry per hit: query frame row, query video, query offset (s), hit video,
    hit offset (s). Per pair, shifts (hit − query offset) are binned to 1 s; a bin scores the distinct query
    frames in it and its ±1 neighbours; the best bin wins if it reaches `min_votes`.

    Returns parallel arrays per matched pair: a, b (query and other video), votes, shift (median over
    supporting hits), a_start/a_end and b_start/b_end (matched time range in each video).
    """
    empty = {k: np.array([], t) for k, t in [("a", np.int64), ("b", np.int64), ("votes", np.int64),
                                             ("shift", np.float64), ("a_start", np.float64), ("a_end", np.float64),
                                             ("b_start", np.float64), ("b_end", np.float64)]}
    if len(q_frame) == 0:
        return empty
    q_video, h_video = q_video.astype(np.int64), h_video.astype(np.int64)
    shift_bin = np.floor(h_offset - q_offset + 0.5).astype(np.int64)
    n_videos = int(max(q_video.max(), h_video.max())) + 1
    lo = int(shift_bin.min()) - 1
    width = int(shift_bin.max()) - lo + 2
    pair = q_video * n_videos + h_video
    key = pair * width + (shift_bin - lo)  # one int per (pair, bin); ±1 = neighbouring bin

    # Distinct query frames per (pair, bin).
    k_frame = np.unique(np.stack([key, q_frame.astype(np.int64)], 1), axis=0)[:, 0]
    bins, own = np.unique(k_frame, return_counts=True)
    score = own.copy()
    for d in (-1, 1):
        j = np.searchsorted(bins, bins + d)
        ok = (j < len(bins)) & (bins[np.minimum(j, len(bins) - 1)] == bins + d)
        score[ok] += own[j[ok]]

    # Best bin per pair (ties go to the bin with more votes of its own).
    bin_pair = bins // width
    order = np.lexsort((-own, -score, bin_pair))
    first = np.ones(len(order), bool)
    first[1:] = bin_pair[order][1:] != bin_pair[order][:-1]
    best = order[first]
    best = best[score[best] >= min_votes]
    if len(best) == 0:
        return empty
    best_pair, best_bin = bin_pair[best], bins[best] % width

    # Supporting hits: the winning bin and its neighbours.
    j = np.searchsorted(best_pair, pair)
    j = np.minimum(j, len(best_pair) - 1)
    keep = (best_pair[j] == pair) & (np.abs((key % width) - best_bin[j]) <= 1)
    idx = j[keep]
    out = {"a": best_pair // n_videos, "b": best_pair % n_videos, "votes": score[best]}
    for name, values, fn in [("a_start", q_offset, np.minimum), ("a_end", q_offset, np.maximum),
                             ("b_start", h_offset, np.minimum), ("b_end", h_offset, np.maximum)]:
        acc = np.full(len(best_pair), np.inf if fn is np.minimum else -np.inf)
        fn.at(acc, idx, values[keep])
        out[name] = acc
    shifts = (h_offset - q_offset)[keep]
    order = np.lexsort((shifts, idx))
    starts = np.searchsorted(idx[order], np.arange(len(best_pair)))
    ends = np.searchsorted(idx[order], np.arange(len(best_pair)), side="right")
    sorted_shifts = shifts[order]
    out["shift"] = (sorted_shifts[(starts + ends - 1) // 2] + sorted_shifts[(starts + ends) // 2]) / 2
    return out
