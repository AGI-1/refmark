(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.RefmarkSearch = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TOKEN_RE = /[A-Za-z0-9_]+/g;

  function tokenize(text) {
    const matches = String(text || "").match(TOKEN_RE);
    return matches ? matches.map((item) => item.toLowerCase()) : [];
  }

  function search(index, query, options) {
    const opts = Object.assign({ topK: 5, expandBefore: 0, expandAfter: 0, includeExcluded: false }, options || {});
    const terms = Array.from(new Set(tokenize(query)));
    const scores = new Map();
    const k1 = Number(index.settings && index.settings.k1) || 1.5;
    const b = Number(index.settings && index.settings.b) || 0.75;
    const avgLen = Math.max(Number(index.avg_len) || 1, 0.000001);
    const lengths = index.lengths || [];
    const postings = index.postings || {};

    for (const token of terms) {
      const entry = postings[token];
      if (!entry) continue;
      const idf = Number(entry.idf) || 0;
      for (const row of entry.p || []) {
        const regionIndex = row[0];
        if (!opts.includeExcluded && index.regions[regionIndex] && index.regions[regionIndex].search_excluded) {
          continue;
        }
        const tf = row[1];
        const length = Number(lengths[regionIndex]) || 0;
        const norm = k1 * (1.0 - b + b * (length / avgLen));
        const score = idf * ((tf * (k1 + 1.0)) / (tf + norm));
        scores.set(regionIndex, (scores.get(regionIndex) || 0) + score);
      }
    }

    return Array.from(scores.entries())
      .filter((item) => item[1] > 0)
      .sort((a, bScore) => {
        if (bScore[1] !== a[1]) return bScore[1] - a[1];
        return stableRef(index, a[0]).localeCompare(stableRef(index, bScore[0]));
      })
      .slice(0, opts.topK)
      .map((item, rankIndex) => hitFor(index, item[0], item[1], rankIndex + 1, opts));
  }

  function hitFor(index, regionIndex, score, rank, opts) {
    const region = index.regions[regionIndex];
    return {
      rank,
      score: Number(score.toFixed(6)),
      doc_id: region.doc_id,
      region_id: region.region_id,
      stable_ref: region.stable_ref,
      summary: region.summary || "",
      text: region.text || "",
      source_path: region.source_path || null,
      context_refs: contextRefs(index.regions, regionIndex, opts.expandBefore, opts.expandAfter),
      roles: region.roles || [],
      search_excluded: Boolean(region.search_excluded),
      search_exclusion_reason: region.search_exclusion_reason || null,
    };
  }

  function contextRefs(regions, regionIndex, before, after) {
    const region = regions[regionIndex];
    let start = regionIndex;
    let remainingBefore = Math.max(0, before || 0);
    while (start > 0 && remainingBefore > 0 && regions[start - 1].doc_id === region.doc_id) {
      start -= 1;
      remainingBefore -= 1;
    }
    let end = regionIndex + 1;
    let remainingAfter = Math.max(0, after || 0);
    while (end < regions.length && remainingAfter > 0 && regions[end].doc_id === region.doc_id) {
      end += 1;
      remainingAfter -= 1;
    }
    return regions.slice(start, end).map((item) => item.stable_ref);
  }

  function stableRef(index, regionIndex) {
    return (index.regions[regionIndex] && index.regions[regionIndex].stable_ref) || "";
  }

  function findRegionElement(stableRefValue) {
    if (!stableRefValue) return null;
    const escaped = cssEscape(stableRefValue);
    return (
      document.querySelector(`[data-refmark-ref="${escaped}"]`) ||
      document.getElementById(stableRefValue) ||
      document.getElementById(stableRefValue.replace(/[^A-Za-z0-9_-]+/g, "-"))
    );
  }

  function jumpTo(stableRefValue, options) {
    const opts = Object.assign({ className: "refmark-active-hit", behavior: "smooth" }, options || {});
    const element = findRegionElement(stableRefValue);
    if (!element) return false;
    clearHighlights(opts.className);
    element.classList.add(opts.className);
    element.scrollIntoView({ behavior: opts.behavior, block: "center" });
    return true;
  }

  function clearHighlights(className) {
    if (typeof document === "undefined") return;
    for (const element of document.querySelectorAll(`.${cssEscape(className || "refmark-active-hit")}`)) {
      element.classList.remove(className || "refmark-active-hit");
    }
  }

  function attachPageSearch(index, options) {
    if (typeof document === "undefined") {
      throw new Error("attachPageSearch requires a browser document.");
    }
    const opts = Object.assign({ topK: 5, placeholder: "Ask this page...", mount: null }, options || {});
    const mount = opts.mount || document.body;
    const panel = document.createElement("form");
    panel.className = "refmark-search-panel";
    panel.innerHTML = [
      '<input class="refmark-search-input" type="search" autocomplete="off">',
      '<ol class="refmark-search-results"></ol>',
    ].join("");
    const input = panel.querySelector(".refmark-search-input");
    const results = panel.querySelector(".refmark-search-results");
    input.placeholder = opts.placeholder;
    mount.prepend(panel);

    function render() {
      const hits = search(index, input.value, { topK: opts.topK, expandAfter: opts.expandAfter || 0 });
      results.innerHTML = "";
      for (const hit of hits) {
        const row = document.createElement("li");
        row.className = "refmark-search-result";
        row.innerHTML = `<button type="button"><strong>${escapeHtml(hit.stable_ref)}</strong><span>${escapeHtml(hit.summary || hit.text)}</span></button>`;
        row.querySelector("button").addEventListener("click", () => jumpTo(hit.stable_ref, opts));
        results.appendChild(row);
      }
    }

    input.addEventListener("input", render);
    panel.addEventListener("submit", (event) => {
      event.preventDefault();
      const first = search(index, input.value, { topK: 1 })[0];
      if (first) jumpTo(first.stable_ref, opts);
    });
    return { panel, input, results, search: (query, searchOptions) => search(index, query, searchOptions) };
  }

  function cssEscape(value) {
    if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  return {
    tokenize,
    search,
    jumpTo,
    attachPageSearch,
  };
});
