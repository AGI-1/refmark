#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");

const RefmarkSearch = require("../../refmark/browser_search.js");

const root = path.resolve(__dirname, "../..");
const dataPath = path.join(root, "examples/bgb_browser_search/output/bgb_demo_data.js");
const queryPath = path.join(root, "examples/bgb_browser_search/break_queries.json");
const outputPath = path.join(root, "examples/bgb_browser_search/output/bgb_break_suite.json");

global.window = {};
eval(fs.readFileSync(dataPath, "utf8"));
const index = global.window.BGB_REFMARK_INDEX;
const queries = JSON.parse(fs.readFileSync(queryPath, "utf8"));

const rows = queries.map((item) => evaluate(item));
const expectedRows = rows.filter((row) => row.category === "expected");
const report = {
  schema: "refmark.bgb_break_suite.v1",
  generated_at: new Date().toISOString(),
  queries: rows.length,
  expected_queries: expectedRows.length,
  expected_hit_at_1: rate(expectedRows, (row) => row.rank === 1),
  expected_hit_at_3: rate(expectedRows, (row) => row.rank !== null && row.rank <= 3),
  rows,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));

function evaluate(item) {
  const started = performance.now();
  const hits = RefmarkSearch.search(index, item.query, { topK: 5 });
  const latencyMs = performance.now() - started;
  const expected = item.expected_prefixes || [];
  let rank = null;
  if (expected.length) {
    for (let i = 0; i < hits.length; i += 1) {
      if (expected.some((prefix) => hits[i].stable_ref.startsWith(prefix))) {
        rank = i + 1;
        break;
      }
    }
  }
  return {
    category: item.category,
    query: item.query,
    note: item.note || "",
    expected_prefixes: expected,
    rank,
    latency_ms: Math.round(latencyMs * 1000) / 1000,
    top_hits: hits.map((hit) => ({
      stable_ref: hit.stable_ref,
      summary: hit.summary,
      score: hit.score,
    })),
  };
}

function rate(rows, predicate) {
  if (!rows.length) return null;
  return Math.round((rows.filter(predicate).length / rows.length) * 10000) / 10000;
}
