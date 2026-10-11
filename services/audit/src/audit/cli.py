"""provena-audit: duplicate-footage audit of a video library. Paths can be local, s3://... or gs://...

Customer (one machine):
    provena-audit fingerprint ./videos            # -> provena_send/ (send to us), provena_private/ (keep)
Customer (many machines, e.g. 50 tasks of a cloud batch job, each with its own index):
    provena-audit fingerprint s3://lib/videos --send s3://lib/provena_send --private s3://lib/provena_private \\
        --part $INDEX --of 50                     # part/of default to the job runner's task index/count
Us:
    provena-audit audit provena_send/             # -> provena_report/report.json + report.html
Customer:
    provena-audit evidence provena_report/report.json ./videos   # real names + side-by-side frames, locally
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="provena-audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fingerprint", help="fingerprint every video in a folder or bucket (runs on your machines)")
    f.add_argument("source")
    f.add_argument("--send", default="provena_send", help="where fingerprints go (send this to Provena)")
    f.add_argument("--private", default="provena_private", help="where file names go (keep this)")
    f.add_argument("--workers", type=int, default=None, help="parallel videos (default: CPU count)")
    f.add_argument("--part", type=int, default=None, help="this machine's index, 0-based")
    f.add_argument("--of", type=int, default=None, help="number of machines")

    a = sub.add_parser("audit", help="find duplicate footage in a fingerprint folder")
    a.add_argument("send")
    a.add_argument("--out", type=Path, default=Path("provena_report"))
    a.add_argument("--min-votes", type=int, default=8)
    a.add_argument("--rate", type=float, nargs=2, default=(60.0, 240.0), metavar=("LOW", "HIGH"),
                   help="price per hour of footage, for the cost estimate")

    e = sub.add_parser("evidence", help="render the report with real file names and side-by-side frames")
    e.add_argument("report", type=Path)
    e.add_argument("source")
    e.add_argument("--private", default="provena_private")
    e.add_argument("--out", type=Path, default=Path("provena_evidence"))
    e.add_argument("--limit", type=int, default=100, help="max side-by-side images")

    args = p.parse_args()
    if args.cmd == "fingerprint":
        from audit.client import fingerprint_folder, part_from_env
        env_part, env_of = part_from_env()
        part = args.part if args.part is not None else (env_part or 0)
        of = args.of if args.of is not None else (env_of or 1)
        if not 0 <= part < of:
            p.error(f"--part must be in 0..{of - 1} (got {part}); set --of to the number of machines")
        fingerprint_folder(args.source, args.send, args.private, args.workers, part, of)
    elif args.cmd == "audit":
        from audit import report
        from audit.audit import run_audit
        rep = run_audit(args.send, args.min_votes, tuple(args.rate))
        report.write(rep, args.out)
        s = rep["summary"]
        print(f"{s['videos']:,} videos, {s['hours']:.1f} h: {s['duplicate_hours']:.2f} h duplicate "
              f"({s['duplicate_share']:.1%}) in {s['groups']:,} groups -> {args.out / 'report.html'}")
    else:
        from audit.evidence import build_evidence
        page = build_evidence(args.report, args.source, args.private, args.out, args.limit)
        print(f"evidence: {page}")


if __name__ == "__main__":
    main()
