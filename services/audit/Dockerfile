# provena-audit client image. Build from the repo root:
#   docker build -f services/audit/Dockerfile -t provena-audit .
# Run (one task of a batch job; part/of come from the job runner or --part/--of):
#   docker run provena-audit fingerprint s3://bucket/videos --send s3://bucket/provena_send --private s3://bucket/provena_private
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY packages/core packages/core
COPY packages/fingerprint packages/fingerprint
COPY packages/matcher packages/matcher
COPY packages/store packages/store
COPY services/audit services/audit
# pdqhash compiles from source: g++ only for the build, then removed.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && pip install --no-cache-dir ./packages/core ./packages/fingerprint ./packages/matcher ./packages/store \
        "./services/audit[cloud]" \
    && apt-get purge -y g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /src
WORKDIR /work
ENTRYPOINT ["provena-audit"]
