"""provena-audit: find duplicate footage in a video library. Paths can be local, s3://... or gs://...

Customer:
    provena-audit fingerprint ./videos              # -> provena_send/ (send to us), provena_private/ (keep)
    provena-audit fingerprint s3://lib/videos --send s3://lib/send --private s3://lib/private --part 3 --of 50
Us:
    provena-audit audit provena_send/               # -> pairs.csv
    provena-audit audit provena_send/ --part 0 --of 10 --out pairs_0.csv    # one of 10 machines
    provena-audit combine provena_send/ pairs_*.csv # join the parts -> pairs.csv
    provena-audit lookup clip.mp4 --index provena_send/   # where does this clip appear?
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
    a.add_argument("--part", type=int, default=0, help="this machine's index, 0-based")
    a.add_argument("--of", type=int, default=1, help="number of machines")

    c = sub.add_parser("combine", help="join the part files of a split audit")
    c.add_argument("send")
    c.add_argument("parts", nargs="+")
    c.add_argument("--out", default="pairs.csv")

    lk = sub.add_parser("lookup", help="find where one or more clips appear in a fingerprint folder")
    lk.add_argument("clips", nargs="+")
    lk.add_argument("--index", required=True, help="fingerprint folder to search (send/)")
    lk.add_argument("--private", default=None, help="names folder of the index, to print file names")
    lk.add_argument("--min-votes", type=int, default=8)

    args = p.parse_args()
    if args.cmd in ("fingerprint", "audit") and not 0 <= args.part < args.of:
        p.error(f"--part must be in 0..{args.of - 1}")
    if args.cmd == "fingerprint":
        from provena_audit.fingerprint import fingerprint_folder
        fingerprint_folder(args.source, args.send, args.private, args.part, args.of, args.workers)
    elif args.cmd == "audit":
        from provena_audit.audit import run_audit
        run_audit(args.send, args.out, args.min_votes, args.part, args.of)
    elif args.cmd == "combine":
        from provena_audit.audit import combine
        combine(args.send, args.parts, args.out)
    else:
        from provena_audit import shards
        from provena_audit.audit import Index
        from provena_audit.storage import Location
        index = Index(args.index)
        names = dict(shards.read_names(Location(args.private))) if args.private else {}
        for clip in args.clips:
            hits = index.search(clip, args.min_votes)
            print(f"{clip}: {len(hits)} match{'es' if len(hits) != 1 else ''}")
            for h in hits:
                print(f"  {names.get(h['id'], h['id'])}  {h['b_start']:.2f}-{h['b_end']:.2f}s  "
                      f"(clip {h['clip_start']:.2f}-{h['clip_end']:.2f}s, {h['votes']} votes)")


if __name__ == "__main__":
    main()
