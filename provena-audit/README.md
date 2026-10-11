# provena-audit

Find duplicate footage in a video library: re-uploads, re-encodes, trimmed and shifted cuts. Only small
fingerprints (about 0.5 MB per hour of footage, no images or file names) leave your machines.

```
pipx install provena-audit            # or: pip install "provena-audit[cloud]" for s3:// and gs://
provena-audit fingerprint ./videos    # -> provena_send/ (fingerprints), provena_private/ (file names, keep)
provena-audit audit provena_send/     # -> pairs.csv: which videos share footage, and where
```

Many machines: give each `--part i --of N`; they can all write to the same `--send` / `--private` location.

How it works: a PDQ hash every 0.25 s (decoded in-process with PyAV, no ffmpeg install needed); every frame is compared with every other frame (C++, all cores);
frames that match and agree on one time shift vote, and 8+ distinct frames on one shift is a copy.

Test (downloads 20 known duplicate pairs from OpenVid-1M): `uv run pytest -s`
