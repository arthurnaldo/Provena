// All-against-all PDQ search with Hough voting on time shift, done in one pass ("vote while searching").
//
// Frames come sorted by video (video ids 0..V-1). For each query video A, a scoreboard maps
// (other video B, 1 s shift bin) -> votes. A match (Hamming distance <= radius, different video) votes for
// its bin and both neighbours, and each A frame votes for a bin at most once, so a bin's votes are the
// distinct A frames whose shift is within about a second of it. No top-k cap and no list of matches.
// Every bin with >= min_votes is a copy, best first, blocking bins within ±2 of an accepted one (one copy
// spreads over ~4 bins), so a clip used twice in B gives two rows.
//
// Each pair is searched once: A only against later videos. The vote counts A's frames; B's count can differ
// (one A frame can match several B frames in static footage), so bins within SLACK of min_votes are re-counted
// on both sides in `support`; a pair's first copy is kept if either side reaches min_votes.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace {

constexpr size_t BLOCK = 4096;  // stored frames per pass (whole videos), stays in cache
constexpr size_t CHUNK = 512;   // query frames per pass (whole videos)
constexpr int SLACK = 4;        // candidate bins need min_votes - SLACK of A's frames

struct Bin {
    int votes = 0, own = 0;            // own: votes whose shift rounds to this bin (tie-break)
    int64_t last = -1, last_own = -1;  // last A frame that voted, so each frame votes once
};

struct Row {
    int64_t a, b;
    int votes, a_votes;  // votes: distinct frames on the side with more; a_votes: A's side
    double shift, a_start, a_end, b_start, b_end;
};

struct Data {
    const uint64_t* pdq;    // n x 4
    const int64_t* video;   // n, sorted
    const double* offset;   // n, seconds
    std::vector<size_t> start;  // video v's frames: start[v] .. start[v + 1]
    int radius, min_votes;
};

inline int distance(const uint64_t* x, const uint64_t* y) {
    return std::popcount(x[0] ^ y[0]) + std::popcount(x[1] ^ y[1]) + std::popcount(x[2] ^ y[2]) +
           std::popcount(x[3] ^ y[3]);
}

inline int bucket(double shift) { return (int)std::floor(shift + 0.5); }

inline uint64_t key(int64_t b, int bin) { return ((uint64_t)(uint32_t)b << 32) | (uint32_t)bin; }

// Video ranges [first, last) grouped so each group has about `size` frames (a long video is its own group).
std::vector<size_t> groups(const Data& d, size_t size) {
    std::vector<size_t> out{0};
    size_t videos = d.start.size() - 1;
    for (size_t v = 0; v < videos; ++v)
        if (d.start[v + 1] - d.start[out.back()] >= size || v + 1 == videos) out.push_back(v + 1);
    return out;
}

// The query side of a search: one video's frames (an audit video, or a clip being looked up).
struct Query {
    int64_t id;
    const uint64_t* pdq;
    const double* offset;
    size_t n;
};

// Votes (distinct frames on the side with more), matched stretch and median shift of one bin, from all matches
// within ±1 bin of it.
Row support(const Data& d, const Query& a, int64_t b, int bin) {
    std::vector<double> shifts;
    std::vector<char> b_seen(d.start[b + 1] - d.start[b], 0);
    int a_frames = 0, b_frames = 0;
    Row r{a.id, b, 0, 0, 0, 1e300, -1e300, 1e300, -1e300};
    for (size_t f = 0; f < a.n; ++f) {
        bool a_seen = false;
        for (size_t g = d.start[b]; g < d.start[b + 1]; ++g) {
            if (distance(a.pdq + 4 * f, d.pdq + 4 * g) > d.radius) continue;
            double s = d.offset[g] - a.offset[f];
            if (std::abs(bucket(s) - bin) > 1) continue;
            if (!a_seen) { a_seen = true; ++a_frames; }
            if (!b_seen[g - d.start[b]]) { b_seen[g - d.start[b]] = 1; ++b_frames; }
            shifts.push_back(s);
            r.a_start = std::min(r.a_start, a.offset[f]); r.a_end = std::max(r.a_end, a.offset[f]);
            r.b_start = std::min(r.b_start, d.offset[g]); r.b_end = std::max(r.b_end, d.offset[g]);
        }
    }
    r.votes = std::max(a_frames, b_frames);
    r.a_votes = a_frames;
    std::sort(shifts.begin(), shifts.end());
    size_t n = shifts.size();
    r.shift = (shifts[(n - 1) / 2] + shifts[n / 2]) / 2;
    return r;
}

// Query frame f (a unique number per query frame) matched index frame g: vote for g's bin and both neighbours,
// at most once per frame. Valid as long as all of f's matches with one video arrive in one uninterrupted loop.
inline void vote(std::unordered_map<uint64_t, Bin>& board, const Data& d, int64_t f, double f_offset, size_t g) {
    int bin = bucket(d.offset[g] - f_offset);
    for (int n = bin - 1; n <= bin + 1; ++n) {
        Bin& x = board[key(d.video[g], n)];
        if (x.last != f) { x.last = f; ++x.votes; }
    }
    Bin& own = board[key(d.video[g], bin)];
    if (own.last_own != f) { own.last_own = f; ++own.own; }
}

// Copies of query `a` from its scoreboard(s): best bins first, blocking bins within ±2 of an accepted one.
void pick(const Data& d, const Query& a, const std::vector<const std::unordered_map<uint64_t, Bin>*>& boards,
          std::vector<Row>& out) {
    std::vector<std::tuple<int64_t, int, int, int>> cand;  // (b, -votes, -own, bin)
    for (auto* board : boards)
        for (auto& [k, x] : *board)
            if (x.votes >= d.min_votes - SLACK) cand.emplace_back((int64_t)(k >> 32), -x.votes, -x.own, (int)(uint32_t)k);
    std::sort(cand.begin(), cand.end());
    std::vector<int> accepted;
    int64_t current = -1;
    for (auto& [b, votes, own, bin] : cand) {
        if (b != current) { current = b; accepted.clear(); }
        if (std::any_of(accepted.begin(), accepted.end(), [&](int s) { return std::abs(s - bin) <= 2; })) continue;
        // B's count may rescue a pair's first copy; a further copy (a repeat) must pass on A's frames,
        // otherwise static footage turns one copy into several.
        Row r = support(d, a, b, bin);
        if ((accepted.empty() ? r.votes : r.a_votes) < d.min_votes) continue;
        accepted.push_back(bin);
        out.push_back(r);
    }
}

void query_chunk(const Data& d, size_t v0, size_t v1, const std::vector<size_t>& blocks, std::vector<Row>& out) {
    std::vector<std::unordered_map<uint64_t, Bin>> boards(v1 - v0);
    for (size_t k = 0; k + 1 < blocks.size(); ++k) {
        size_t g0 = d.start[blocks[k]], g1 = d.start[blocks[k + 1]];
        // Each stored video sits inside one block, so all of frame f's matches with a video B arrive in this
        // one inner loop.
        for (size_t f = d.start[v0]; f < d.start[v1]; ++f) {
            int64_t a = d.video[f];
            for (size_t g = std::max(g0, d.start[a + 1]); g < g1; ++g)  // later videos only
                if (distance(d.pdq + 4 * f, d.pdq + 4 * g) <= d.radius) vote(boards[a - v0], d, f, d.offset[f], g);
        }
    }
    for (size_t a = v0; a < v1; ++a)
        pick(d, {(int64_t)a, d.pdq + 4 * d.start[a], d.offset + d.start[a], d.start[a + 1] - d.start[a]},
             {&boards[a - v0]}, out);
}

using Hashes = nb::ndarray<const uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Ints = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Floats = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out = std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int>, std::vector<double>,
                       std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>;

Out to_out(const std::vector<Row>& rows) {
    Out out;
    for (auto& r : rows) {
        std::get<0>(out).push_back(r.a); std::get<1>(out).push_back(r.b); std::get<2>(out).push_back(r.votes);
        std::get<3>(out).push_back(r.shift); std::get<4>(out).push_back(r.a_start); std::get<5>(out).push_back(r.a_end);
        std::get<6>(out).push_back(r.b_start); std::get<7>(out).push_back(r.b_end);
    }
    return out;
}

Out find_pairs(Hashes pdq, Ints video, Floats offset, int radius, int min_votes, int threads, int part, int of) {
    Data d{pdq.data(), video.data(), offset.data(), {}, radius, min_votes};
    size_t n = video.shape(0);
    std::vector<Row> rows;
    if (n) {
        d.start.push_back(0);
        for (size_t i = 1; i < n; ++i)
            if (d.video[i] != d.video[i - 1]) d.start.push_back(i);
        d.start.push_back(n);
        auto blocks = groups(d, BLOCK), chunks = groups(d, CHUNK);
        std::atomic<size_t> next{0};
        std::vector<std::vector<Row>> found(std::max(threads, 1));
        {
            nb::gil_scoped_release release;
            std::vector<std::thread> pool;
            for (auto& mine : found)
                pool.emplace_back([&] {
                    // Machine `part` of `of` queries chunks part, part + of, part + 2 * of, ...
                    for (size_t c; (c = part + of * next++) + 1 < chunks.size();)
                        query_chunk(d, chunks[c], chunks[c + 1], blocks, mine);
                });
            for (auto& t : pool) t.join();
        }
        for (auto& f : found) rows.insert(rows.end(), f.begin(), f.end());
    }
    return to_out(rows);
}


// One clip against an index: only the clip's frames are searched. Threads take blocks of whole index videos, so
// each video's votes land on one thread's scoreboard and the boards never overlap.
Out lookup(Hashes pdq, Ints video, Floats offset, Hashes qpdq, Floats qoffset, int radius, int min_votes, int threads) {
    Data d{pdq.data(), video.data(), offset.data(), {}, radius, min_votes};
    Query q{-1, qpdq.data(), qoffset.data(), qoffset.shape(0)};
    size_t n = video.shape(0);
    std::vector<Row> rows;
    if (n && q.n) {
        d.start.push_back(0);
        for (size_t i = 1; i < n; ++i)
            if (d.video[i] != d.video[i - 1]) d.start.push_back(i);
        d.start.push_back(n);
        auto blocks = groups(d, BLOCK);
        std::vector<std::unordered_map<uint64_t, Bin>> boards(std::max(threads, 1));
        std::atomic<size_t> next{0};
        nb::gil_scoped_release release;
        {
            std::vector<std::thread> pool;
            for (auto& board : boards)
                pool.emplace_back([&] {
                    for (size_t k; (k = next++) + 1 < blocks.size();)
                        for (size_t f = 0; f < q.n; ++f)
                            for (size_t g = d.start[blocks[k]]; g < d.start[blocks[k + 1]]; ++g)
                                if (distance(q.pdq + 4 * f, d.pdq + 4 * g) <= d.radius) vote(board, d, f, q.offset[f], g);
                });
            for (auto& t : pool) t.join();
        }
        std::vector<const std::unordered_map<uint64_t, Bin>*> all;
        for (auto& board : boards) all.push_back(&board);
        pick(d, q, all, rows);
    }
    return to_out(rows);
}

}  // namespace

NB_MODULE(_kernel, m) {
    m.def("find_pairs", &find_pairs, nb::arg("pdq"), nb::arg("video"), nb::arg("offset"), nb::arg("radius"),
          nb::arg("min_votes"), nb::arg("threads"), nb::arg("part") = 0, nb::arg("of") = 1,
          "Matched (a, b, votes, shift, a_start, a_end, b_start, b_end) with a < b, each pair searched once.");
    m.def("lookup", &lookup, nb::arg("pdq"), nb::arg("video"), nb::arg("offset"), nb::arg("query_pdq"),
          nb::arg("query_offset"), nb::arg("radius"), nb::arg("min_votes"), nb::arg("threads"),
          "Copies of one clip in an index: (a=-1, b, votes, shift, a_start, a_end, b_start, b_end).");
}
