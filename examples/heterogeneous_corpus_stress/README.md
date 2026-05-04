# Heterogeneous Corpus Stress

This fixture is intentionally small. It is not a benchmark. It exists to smoke
test how Refmark behaves when a corpus is less tidy than a clean Markdown manual:

- Markdown guide content;
- plain-text wiki notes;
- RST-style policy notes;
- table-like records;
- duplicate support;
- release-note/query-magnet content that should be inspected separately.

Build and inspect:

```bash
python -m refmark.cli build-index examples/heterogeneous_corpus_stress/corpus \
  -o tmp/heterogeneous_stress/index.json \
  --source local

python -m refmark.cli inspect-index tmp/heterogeneous_stress/index.json \
  -o tmp/heterogeneous_stress/smells.json
```

Evaluate the tiny fixture:

```bash
python -m refmark.cli eval-index tmp/heterogeneous_stress/index.json \
  examples/heterogeneous_corpus_stress/eval.jsonl \
  --strategy rerank \
  --top-k 5 \
  -o tmp/heterogeneous_stress/eval.json
```

What this fixture should expose:

- query magnets, especially release-note style pages;
- duplicate or near-duplicate support regions;
- uneven region quality across source formats;
- refs that remain useful even when the original files have different styles.

The fixture deliberately avoids DOCX/PDF binaries. For those formats, Refmark
currently resolves to extracted text unless page/layout provenance is supplied.
