#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const { performance } = require("perf_hooks");

const RefmarkSearch = require("../../refmark/browser_search.js");

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rawPortable = readJson(args.rawPortable);
  const refmarkPortable = readJson(args.refmarkPortable);
  const rawBrowser = args.rawBrowser ? readJson(args.rawBrowser) : null;
  const refmarkBrowser = readJson(args.refmarkBrowser);
  const questions = readQuestions(args.questionCache, refmarkPortable);
  const topKs = splitInts(args.topKs || "1,3,5,10");
  const engines = new Set(splitWords(args.engines || "refmark-bm25,minisearch,lunr,flexsearch"));
  const variants = [];

  if (engines.has("refmark-bm25") && rawBrowser) {
    variants.push(makeRefmarkBrowserVariant("raw_refmark_bm25", rawBrowser));
  }
  if (engines.has("refmark-bm25")) {
    variants.push(makeRefmarkBrowserVariant("refmark_bm25", refmarkBrowser));
  }
  if (engines.has("minisearch")) {
    const mini = optionalRequire("minisearch");
    if (mini) {
      variants.push(makeMiniSearchVariant("raw_minisearch", rawPortable, false, mini));
      variants.push(makeMiniSearchVariant("refmark_minisearch", refmarkPortable, true, mini));
    }
  }
  if (engines.has("lunr")) {
    const lunr = optionalRequire("lunr");
    if (lunr) {
      variants.push(makeLunrVariant("raw_lunr", rawPortable, false, lunr));
      variants.push(makeLunrVariant("refmark_lunr", refmarkPortable, true, lunr));
    }
  }
  if (engines.has("flexsearch")) {
    const flex = optionalRequire("flexsearch");
    if (flex) {
      variants.push(makeFlexSearchVariant("raw_flexsearch", rawPortable, false, flex));
      variants.push(makeFlexSearchVariant("refmark_flexsearch", refmarkPortable, true, flex));
    }
  }

  const reports = {};
  for (const variant of variants) {
    reports[variant.name] = evaluateVariant(variant, questions, topKs, Number(args.warmup || 10));
  }
  const report = {
    schema: "refmark.static_search_benchmark.v1",
    generated_at: new Date().toISOString(),
    inputs: {
      raw_portable: args.rawPortable,
      refmark_portable: args.refmarkPortable,
      raw_browser: args.rawBrowser || null,
      refmark_browser: args.refmarkBrowser,
      question_cache: args.questionCache,
    },
    settings: {
      top_ks: topKs,
      engines: Array.from(engines),
      warmup: Number(args.warmup || 10),
    },
    methodology: methodologyNotes(),
    questions: questions.length,
    regions: refmarkPortable.regions.length,
    documents: new Set(refmarkPortable.regions.map((region) => region.doc_id)).size,
    reports,
  };

  if (args.output) {
    fs.mkdirSync(path.dirname(args.output), { recursive: true });
    fs.writeFileSync(args.output, JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify(report, null, 2));
}

function makeRefmarkBrowserVariant(name, index) {
  return {
    name,
    engine: "refmark-browser-bm25",
    license: "MIT",
    deployment: "static_browser",
    server_required: false,
    api_key_required: false,
    offline: true,
    build() {
      return {
        build_ms: 0,
        raw_size_bytes: byteSize(index),
        gzip_size_bytes: gzipSize(index),
        search(query, topK) {
          return RefmarkSearch.search(index, query, { topK }).map(hitFromRefmark);
        },
      };
    },
  };
}

function makeMiniSearchVariant(name, portable, enriched, MiniSearch) {
  return {
    name,
    engine: "minisearch",
    license: "MIT",
    deployment: "static_browser",
    server_required: false,
    api_key_required: false,
    offline: true,
    build() {
      const docs = docsFromPortable(portable, enriched);
      const started = performance.now();
      const index = new MiniSearch({
        fields: ["text", "summary", "questions", "keywords"],
        storeFields: ["stable_ref", "doc_id"],
        idField: "stable_ref",
      });
      index.addAll(docs);
      return {
        build_ms: performance.now() - started,
        raw_size_bytes: byteSize(index.toJSON()),
        gzip_size_bytes: gzipSize(index.toJSON()),
        search(query, topK) {
          return index.search(query, { limit: topK }).slice(0, topK).map(hitFromStored);
        },
      };
    },
  };
}

function makeLunrVariant(name, portable, enriched, lunr) {
  return {
    name,
    engine: "lunr",
    license: "MIT",
    deployment: "static_browser",
    server_required: false,
    api_key_required: false,
    offline: true,
    build() {
      const docs = docsFromPortable(portable, enriched);
      const byRef = new Map(docs.map((doc) => [doc.stable_ref, doc]));
      const started = performance.now();
      const index = lunr(function build() {
        this.ref("stable_ref");
        this.field("text");
        this.field("summary", { boost: 1.5 });
        this.field("questions", { boost: 1.4 });
        this.field("keywords", { boost: 1.2 });
        for (const doc of docs) this.add(doc);
      });
      return {
        build_ms: performance.now() - started,
        raw_size_bytes: byteSize(index.toJSON()),
        gzip_size_bytes: gzipSize(index.toJSON()),
        search(query, topK) {
          return searchLunr(index, lunr, query).slice(0, topK).map((row) => {
            const doc = byRef.get(row.ref);
            return { stable_ref: row.ref, doc_id: doc.doc_id, score: row.score };
          });
        },
      };
    },
  };
}

function makeFlexSearchVariant(name, portable, enriched, flex) {
  return {
    name,
    engine: "flexsearch",
    license: "Apache-2.0",
    deployment: "static_browser",
    server_required: false,
    api_key_required: false,
    offline: true,
    build() {
      const docs = docsFromPortable(portable, enriched);
      const Index = flex.Index || flex.default?.Index;
      if (!Index) throw new Error("FlexSearch Index API not found.");
      const started = performance.now();
      const index = new Index({
        tokenize: "forward",
        resolution: 9,
      });
      for (let i = 0; i < docs.length; i += 1) {
        index.add(i, [docs[i].text, docs[i].summary, docs[i].questions, docs[i].keywords].join("\n"));
      }
      return {
        build_ms: performance.now() - started,
        raw_size_bytes: null,
        gzip_size_bytes: null,
        search(query, topK) {
          return index.search(query, topK).map((docIndex, rankIndex) => ({
            stable_ref: docs[docIndex].stable_ref,
            doc_id: docs[docIndex].doc_id,
            score: 1.0 / (rankIndex + 1),
          }));
        },
      };
    },
  };
}

function evaluateVariant(variant, questions, topKs, warmupCount) {
  const buildStarted = performance.now();
  const instance = variant.build();
  const buildMs = instance.build_ms ?? (performance.now() - buildStarted);
  for (const question of questions.slice(0, warmupCount)) {
    instance.search(question.query, Math.max(...topKs));
  }
  const refHits = Object.fromEntries(topKs.map((k) => [k, 0]));
  const docHits = Object.fromEntries(topKs.map((k) => [k, 0]));
  let reciprocal = 0;
  let docReciprocal = 0;
  const latencies = [];
  const misses = [];
  const maxK = Math.max(...topKs);
  for (const question of questions) {
    const started = performance.now();
    const hits = instance.search(question.query, maxK).slice(0, maxK);
    latencies.push(performance.now() - started);
    const goldRefs = new Set(question.gold_refs);
    let refRank = null;
    let docRank = null;
    for (let i = 0; i < hits.length; i += 1) {
      if (refRank === null && goldRefs.has(hits[i].stable_ref)) refRank = i + 1;
      if (docRank === null && hits[i].doc_id === question.doc_id) docRank = i + 1;
    }
    if (refRank !== null) reciprocal += 1 / refRank;
    if (docRank !== null) docReciprocal += 1 / docRank;
    if (refRank === null) {
      misses.push({ query: question.query, gold: question.gold_refs, top_refs: hits.slice(0, 3).map((hit) => hit.stable_ref) });
    }
    for (const k of topKs) {
      if (refRank !== null && refRank <= k) refHits[k] += 1;
      if (docRank !== null && docRank <= k) docHits[k] += 1;
    }
  }
  const total = Math.max(questions.length, 1);
  return {
    engine: variant.engine,
    license: variant.license,
    deployment: variant.deployment,
    server_required: variant.server_required,
    api_key_required: variant.api_key_required,
    offline: variant.offline,
    build_ms: round(buildMs),
    raw_size_bytes: instance.raw_size_bytes,
    gzip_size_bytes: instance.gzip_size_bytes,
    anchor_hit_at_k: rateMap(refHits, total),
    article_hit_at_k: rateMap(docHits, total),
    anchor_mrr: round(reciprocal / total, 4),
    article_mrr: round(docReciprocal / total, 4),
    latency_ms: summarizeLatencies(latencies),
    sample_misses: misses.slice(0, 8),
  };
}

function methodologyNotes() {
  return {
    corpus: "All engines receive the same Refmark regions and are evaluated on the same Refmark-labeled question cache.",
    raw_vs_refmark: "raw_* variants index source region text only; refmark_* variants index source text plus generated retrieval views.",
    quality_metrics: "anchor metrics require the exact gold stable_ref; article metrics count any region in the gold document.",
    refmark_bm25: "Refmark BM25 uses a pre-exported browser postings payload, so build_ms is 0 for the query-time artifact.",
    minisearch: "MiniSearch uses default query ranking; prefix/fuzzy expansion was avoided because it produced noisy localization on this corpus.",
    lunr: "Lunr queries are punctuation-sanitized before using its normal parser, preventing user questions from being interpreted as query syntax.",
    flexsearch: "FlexSearch uses a single concatenated field index because multi-field result merging lacks a directly comparable global BM25-style score.",
    deployment: "Server/API/offline flags describe the benchmarked static adapters, not every possible deployment of each engine.",
  };
}

function searchLunr(index, lunr, query) {
  const sanitized = String(query).replace(/[^A-Za-z0-9_\s]+/g, " ").replace(/\s+/g, " ").trim();
  if (sanitized) {
    try {
      return index.search(sanitized);
    } catch (_error) {
      // Fall through to the programmatic API for any remaining parser edge case.
    }
  }
  const tokens = Array.from(new Set(lunr.tokenizer(sanitized || query).map((token) => token.toString()).filter(Boolean)));
  if (!tokens.length) return [];
  return index.query((builder) => {
    for (const token of tokens) {
      builder.term(token, {
        fields: ["text", "summary", "questions", "keywords"],
        presence: lunr.Query.presence.OPTIONAL,
        wildcard: lunr.Query.wildcard.TRAILING,
      });
    }
  });
}

function docsFromPortable(portable, enriched) {
  return portable.regions.map((region) => {
    const view = region.view || {};
    return {
      stable_ref: region.stable_ref || `${region.doc_id}:${region.region_id}`,
      doc_id: region.doc_id,
      text: region.text || "",
      summary: enriched ? view.summary || "" : "",
      questions: enriched ? (view.questions || []).join("\n") : "",
      keywords: enriched ? (view.keywords || []).join("\n") : "",
    };
  });
}

function readQuestions(cachePath, portable) {
  const refs = new Set(portable.regions.map((region) => region.stable_ref || `${region.doc_id}:${region.region_id}`));
  return fs.readFileSync(cachePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => (row.gold_refs || []).every((ref) => refs.has(ref)))
    .sort((a, b) => `${a.stable_ref}\0${a.query}`.localeCompare(`${b.stable_ref}\0${b.query}`));
}

function hitFromRefmark(hit) {
  return { stable_ref: hit.stable_ref, doc_id: hit.doc_id, score: hit.score };
}

function hitFromStored(hit) {
  return { stable_ref: hit.stable_ref || hit.id, doc_id: hit.doc_id, score: hit.score };
}

function optionalRequire(name) {
  try {
    return require(name);
  } catch (error) {
    if (error && error.code === "MODULE_NOT_FOUND") {
      console.error(`[SKIP] ${name} is not installed. Run npm.cmd install in examples/static_search_benchmark.`);
      return null;
    }
    throw error;
  }
}

function parseArgs(argv) {
  const args = {
    rawPortable: "examples/portable_search_index/output/full_2m_raw_index.json",
    refmarkPortable: "examples/portable_search_index/output/full_2m_openrouter_index.json",
    rawBrowser: "examples/portable_search_index/output/full_2m_raw_browser_index.json",
    refmarkBrowser: "examples/portable_search_index/output/full_2m_browser_index.json",
    questionCache: "examples/portable_search_index/output/eval_question_cache_online_scaling.jsonl",
    output: "examples/static_search_benchmark/output/static_search_benchmark.json",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (!item.startsWith("--")) continue;
    const key = item.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function splitInts(value) {
  return String(value).split(",").filter(Boolean).map((item) => Number(item.trim()));
}

function splitWords(value) {
  return String(value).split(",").map((item) => item.trim()).filter(Boolean);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function byteSize(value) {
  return Buffer.byteLength(JSON.stringify(value));
}

function gzipSize(value) {
  return zlib.gzipSync(Buffer.from(JSON.stringify(value)), { level: 9 }).length;
}

function rateMap(counts, total) {
  return Object.fromEntries(Object.entries(counts).map(([k, count]) => [k, round(count / total, 4)]));
}

function summarizeLatencies(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    avg: round(sorted.reduce((sum, value) => sum + value, 0) / Math.max(sorted.length, 1)),
    p50: round(percentile(sorted, 0.50)),
    p95: round(percentile(sorted, 0.95)),
    max: round(sorted[sorted.length - 1] || 0),
  };
}

function percentile(sorted, q) {
  if (!sorted.length) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1));
  return sorted[index];
}

function round(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

main();
