# provena-audit

Find duplicate and near-duplicate footage in a video library: re-uploads, re-encodes, resized, trimmed and
time-shifted cuts, light recolours and captions. Gives which videos share footage and exactly which seconds.

Fingerprinting runs where the videos are. Only small fingerprints leave your machines (about 0.5 MB per hour of
footage: hashes and timings under random IDs, no images, audio or file names); the file names stay in a private
folder you keep.

## Install

```
pipx install provena-audit                  # or: pip install provena-audit
pip install "provena-audit[cloud]"          # adds s3:// and gs:// sources
```

Python 3.11+. Video decoding is built in (PyAV), so no separate ffmpeg install is needed.

## Use

```
provena-audit fingerprint ./videos          # -> provena_send/ (fingerprints), provena_private/ (file names, keep)
provena-audit audit provena_send/           # -> pairs.csv, and prints the duplicate hours
```

`pairs.csv` has one row per copy: video IDs `a` and `b`, `votes` (distinct frames that agree), `shift` (seconds,
b minus a) and the matched range in each video. Map IDs back to files with `provena_private/names_*.csv`.

Large libraries, many machines (each gets its own index; all can share one `--send` / `--private` location, local
or `s3://` / `gs://`):

```
provena-audit fingerprint s3://lib/videos --send s3://lib/send --private s3://lib/private --part 3 --of 50
provena-audit audit provena_send/ --part 0 --of 10 --out pairs_0.csv      # one of 10 machines
provena-audit combine provena_send/ pairs_*.csv                           # join the parts -> pairs.csv
```

Fingerprinting resumes where it stopped; split audits give exactly the same pairs as one run.

Where does a clip appear? Search one or more clips against a fingerprint folder (the index loads once; each search
takes well under a second on a laptop for ~1M indexed clips):

```
provena-audit lookup clip.mp4 other.mp4 --index provena_send/ --private provena_private/
```

## How it works

1. A PDQ hash (Meta's open perceptual hash) of the full-size frame every 0.25 s.
2. Every frame is compared with every frame of every other video (C++, all cores; each pair of videos once).
3. Matching frames vote on the time shift between the two videos (1 s bins, each frame votes once per bin and its
   neighbours). 8 or more distinct frames on one shift is a copy; a clip used twice in another video gives two rows.

Measured on OpenVid-1M (1.02M clips, ~2,050 h): fingerprinted and audited in ~6 h on cloud spot machines for
~$10-12; 7.6% duplicate or near-duplicate footage.

Not caught yet: crops, mirrored copies and heavy overlays (planned: SSCD embeddings alongside PDQ).

## Test

`uv run pytest -s` downloads 20 known duplicate pairs from OpenVid-1M, fingerprints and audits them (also split
over 3 parts + combine) and checks all are found.

## Licence

MIT
