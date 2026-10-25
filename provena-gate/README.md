# provena-gate

Checks video data deliveries before payment: finds duplicate footage inside a delivery, footage the buyer already
owns, footage already sold to another buyer (opt-in, fingerprints only) and footage from public AI datasets, then
decides what to pay within fixed rules.

Built at Cal Hacks 13.0 (Oct 23-25, 2026) on top of this repo's matching engine (`../provena-audit`, built before
the event) and its fingerprint index of public datasets.

Run: `cd provena-gate && uv sync && uv run pytest`
