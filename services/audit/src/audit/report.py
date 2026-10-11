"""Render an audit as one self-contained HTML page. Used on our side (IDs only) and by the customer's
`evidence` command (real file names + side-by-side frames)."""

import html
import json
from pathlib import Path

CSS = """
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4; --accent:#b45309; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524;
  --line:#44403c; --accent:#f59e0b; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1000px; margin:0 auto; padding:24px 16px; }
h1 { font-size:24px; margin:0 0 4px; } .muted { color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:20px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card b { display:block; font-size:22px; color:var(--accent); }
.group { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:12px 0; }
table { width:100%; border-collapse:collapse; font-size:13px; } td, th { text-align:left; padding:4px 6px;
  border-bottom:1px solid var(--line); vertical-align:top; } code { font-size:12px; word-break:break-all; }
img { max-width:100%; border-radius:6px; margin-top:6px; }
"""


def _t(s: float) -> str:
    m, s = divmod(float(s), 60)
    return f"{int(m)}:{s:04.1f}"


def _h(hours: float) -> str:
    return f"{hours:,.1f} h" if hours >= 1 else f"{hours * 60:.1f} min"


def render(report: dict, names: dict[str, str] | None = None, images: dict[tuple[str, str], str] | None = None,
           max_groups: int = 200) -> str:
    names, images = names or {}, images or {}
    s = report["summary"]
    label = lambda i: html.escape(names.get(i, i))
    lo, hi = s["duplicate_cost"]
    rate_lo, rate_hi = s["rate_per_hour"]
    out = [f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
           f"<title>Duplicate Footage Audit</title><style>{CSS}</style><main>",
           "<h1>Duplicate footage audit</h1>",
           "<p class='muted'>Same footage appearing more than once, found by comparing the video content "
           "(not file names or hashes). Only fingerprints were analysed.</p>",
           "<div class='cards'>",
           f"<div class='card'><b>{s['videos']:,}</b>videos, {_h(s['hours'])}</div>",
           f"<div class='card'><b>{s['duplicate_share']:.1%}</b>of footage is duplicate</div>",
           f"<div class='card'><b>{_h(s['duplicate_hours'])}</b>duplicate footage in "
           f"{s['videos_with_duplicate_footage']:,} videos</div>",
           f"<div class='card'><b>${lo:,}–${hi:,}</b>at ${rate_lo:,.0f}–${rate_hi:,.0f} per hour</div>",
           "</div>",
           "<p class='muted'>Duplicate time counts only the lesser copy of each match (shorter video), "
           f"with overlaps merged, so the best copy is never counted. A match needs at least {s['min_votes']} "
           "frames (4 per second) agreeing on the same time alignment.</p>",
           f"<h2>{s['groups']:,} groups of matching videos</h2>"]
    for gi, g in enumerate(report["groups"][:max_groups]):
        out.append(f"<div class='group'><b>Group {gi + 1}</b> · {len(g['videos'])} videos<table>"
                   "<tr><th>Video</th><th>Length</th><th>Duplicate time</th></tr>")
        for v in g["videos"]:
            out.append(f"<tr><td><code>{label(v['id'])}</code></td><td>{_t(v['duration'])}</td>"
                       f"<td>{_t(v['duplicate_seconds'])}</td></tr>")
        out.append("</table><table><tr><th>Match</th><th>Frames agreeing</th></tr>")
        for pr in g["pairs"][:20]:
            a, b = pr["a"], pr["b"]
            img = images.get((a, b))
            out.append(f"<tr><td><code>{label(a)}</code> {_t(pr['a_range'][0])}–{_t(pr['a_range'][1])}<br>"
                       f"= <code>{label(b)}</code> {_t(pr['b_range'][0])}–{_t(pr['b_range'][1])}"
                       + (f"<br><img src='{html.escape(img)}' loading='lazy' alt='side by side'>" if img else "")
                       + f"</td><td>{pr['votes']}</td></tr>")
        if len(g["pairs"]) > 20:
            out.append(f"<tr><td class='muted' colspan=2>+ {len(g['pairs']) - 20} more matches</td></tr>")
        out.append("</table></div>")
    if len(report["groups"]) > max_groups:
        out.append(f"<p class='muted'>+ {len(report['groups']) - max_groups:,} more groups in report.json</p>")
    out.append("</main>")
    return "\n".join(out)


def write(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "report.html").write_text(render(report))
