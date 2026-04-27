# Browser Page Search

This example is the smallest browser-facing product shape for Refmark:

```text
current page -> narrow refmark regions -> semantic find box -> jump/highlight
```

It is deliberately smaller than a full documentation search engine. `Ctrl+F`
finds exact words. Site search finds pages. Refmark page search finds the
answer region inside the page.

Open the demo HTML directly:

```text
examples/browser_page_search/semantic_find_demo.html
```

Try queries such as:

- `when can we deploy`
- `how do rollback plans work`
- `when do service tokens rotate`
- `how long are audit logs kept`

For a generated corpus index, build the normal portable index and export a
browser payload:

```bash
python -m refmark.cli build-index docs -o docs.refmark-index.json --source openrouter --view-cache .refmark/views.jsonl
python -m refmark.cli export-browser-index docs.refmark-index.json -o docs.refmark-browser.json
```

Then load `refmark/browser_search.js` and call:

```html
<script src="refmark/browser_search.js"></script>
<script>
  const index = await fetch("docs.refmark-browser.json").then((res) => res.json());
  RefmarkSearch.attachPageSearch(index, { topK: 5 });
</script>
```

If the page already contains elements with matching anchors, search results can
jump and highlight immediately:

```html
<section data-refmark-ref="security:P02">
  <h2>Audit Retention</h2>
  <p>Audit logs are retained for 180 days by default.</p>
</section>
```

The first version is BM25-only and has no runtime dependency, API key, embedding
model, or server. A learned resolver or ONNX/WebAssembly layer can be added
later as a second-stage scorer over the local BM25 candidates.

## Comparison Ladder

For rigorous browser/docs-search testing, compare the same Refmark-labeled
queries against:

- native page find: exact-match baseline for the current page
- raw BM25 over page or corpus text
- Refmark BM25 over text plus generated retrieval views
- static search libraries such as Lunr, MiniSearch, FlexSearch, or Pagefind
- hosted/server search such as Meilisearch, Typesense, or Elasticsearch
- semantic embeddings, where runtime dependency and cache size are acceptable

Refmark labels keep the comparison grounded: measure whether each system lands
on the right page, the right region, or any anchor inside the target range, then
track overcitation and undercitation from the returned context.
