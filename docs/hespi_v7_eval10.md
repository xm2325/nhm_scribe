# Hespi v7 Eval10

Hespi v6 recovered the full `BM000625315` value in a targeted OCR crop, but DeepSeek returned the truncated whole-sheet reading `BM0006253`.

## Conservative resolver

The v7 resolver changes a catalogue number only when:

- the LLM produced a non-empty structured identifier;
- exactly one OCR ensemble candidate starts with the complete LLM value;
- the candidate adds no more than four characters;
- the candidate has a one-to-four-letter prefix followed only by digits.

It does not fill empty predictions and does not choose between two valid extensions.

Every decision is saved with:

- original catalogue number;
- resolved catalogue number;
- resolution status;
- correction flag;
- all OCR candidates.

The pipeline also writes a separate `*_reconciled.csv` prediction file so raw LLM output and final pipeline output are not confused.

## Run

Use **Actions -> Hespi v7 eval10 conservative catalog resolver -> Run workflow**, or update:

```text
.github/run-hespi-v7-eval10.trigger
```

Ordinary code pushes do not run this experiment.
