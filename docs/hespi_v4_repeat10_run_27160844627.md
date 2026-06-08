# Hespi v4 Eval10 Result

GitHub Actions run: `27160844627`

## What worked

- all 10 real images were shared across the experiment;
- layout and OCR ran once;
- all three DeepSeek repeats parsed 10 of 10 records;
- prompt SHA-256 values were identical across repeats;
- pairwise field agreement was 97.6%;
- no thinking tokens were used.

## Quality result

Operational stability did not translate into extraction accuracy:

- mean exact match: 12.5%;
- mean token F1: 15.3%;
- consensus coverage: about 30%;
- 4 of the original 11 auto-accepted field units were not exact matches.

The main failure mode was stable copying of incorrect OCR. Examples included truncated or wrong catalogue numbers and misspelled scientific names. Direct OCR support therefore means that a prediction is traceable, not that it is correct.

## Decision

Do not scale this configuration to 50 records yet.

The scale gate now also requires:

- consensus exact match of at least 20%;
- consensus token F1 of at least 25%;
- auto-accept exact match of at least 95%.

The next extraction experiment should improve targeted catalogue-number OCR and scientific-name reconciliation before another eval10 comparison.
