"""provena-audit: find duplicate footage in a video library. Paths can be local, s3://... or gs://...

Customer:
    provena-audit fingerprint ./videos              # -> provena_send/ (send to us), provena_private/ (keep)
    provena-audit fingerprint s3://lib/videos --send s3://lib/send --private s3://lib/private --part 3 --of 50
Us:
    provena-audit audit provena_send/               # -> pairs.csv
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="provena-audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fingerprint", help="fingerprint every video in a folder or bucket")
    f.add_argument("source")
    f.add_argument("--send", default="provena_send", help="where fingerprints go (send this to Provena)")
    f.add_argument("--private", default="provena_private", help="where file names go (keep this)")
    f.add_argument("--part", type=int, default=0, help="this machine's index, 0-based")
    f.add_argument("--of", type=int, default=1, help="number of machines")
    f.add_argument("--workers", type=int, default=None, help="videos at once (default: CPU count)")

    a = sub.add_parser("audit", help="find duplicate footage in a fingerprint folder")
    a.add_argument("send")
    a.add_argument("--out", default="pairs.csv")
    a.add_argument("--min-votes", type=int, default=8)

    args = p.parse_args()
    if args.cmd == "fingerprint":
        if not 0 <= args.part < args.of:
            p.error(f"--part must be in 0..{args.of - 1}")
        from provena_audit.fingerprint import fingerprint_folder
        fingerprint_folder(args.source, args.send, args.private, args.part, args.of, args.workers)
    else:
        from provena_audit.audit import run_audit
        run_audit(args.send, args.out, args.min_votes)


if __name__ == "__main__":
    main()
