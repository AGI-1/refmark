# Document Extraction And Provenance

Refmark can map `.txt`, `.md`, `.rst`, `.html`, `.docx`, and `.pdf` inputs into
addressable regions. For DOCX and PDF, the current provenance contract is
intentionally modest.

## Current Contract

- Refs resolve to **extracted text regions**, not guaranteed original-layout
  coordinates.
- Region records include `source_path`, `start_line`, `end_line`, `hash`,
  neighbor refs, and optional `parent_region_id`.
- PDF extraction uses `pypdf`.
- DOCX extraction uses plain OOXML paragraph extraction.
- Scanned PDFs, complex tables, headers/footers, comments, tracked changes, and
  layout-heavy pages may require a stronger upstream parser or OCR.

This means a ref such as `contract:P12` is suitable for evidence evaluation,
review highlighting over extracted text, and RAG metadata. It is not yet a
promise that Refmark can draw a box on the original PDF page.

## Recommended Use

For internal RAG and review pipelines:

1. Extract or normalize documents with your preferred parser.
2. Store the extracted text artifact if layout provenance matters.
3. Build a Refmark manifest over that extracted text.
4. Keep `source_path`, extraction metadata, and parser version next to the
   manifest.

For publishable demos:

- Say "PDF/DOCX extracted text support" rather than "PDF/DOCX layout citation".
- Include extraction warnings in reports.
- Prefer side-by-side HTML reports for human review.

## Parent/Section Metadata

Markdown-style headings are currently used as lightweight parent regions when
they are present in the manifest. Child regions under the same heading receive
`parent_region_id`, which can be used for section-aware expansion and coarse
document navigation.

For PDF/DOCX, heading hierarchy is not yet reliable unless the extracted text
already preserves clear heading regions. Treat `parent_region_id` as optional
metadata, not a guaranteed field.
