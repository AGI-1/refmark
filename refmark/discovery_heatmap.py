"""HTML maps for discovery cluster manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from refmark.discovery import DiscoveryManifest
from refmark.pipeline import RegionRecord
from refmark.search_index import approx_tokens


@dataclass(frozen=True)
class DiscoveryMapItem:
    cluster_id: str
    name: str
    strategy: str
    source: str
    refs: list[str]
    terms: list[str]
    tokens: int
    regions: int
    sample_titles: list[str]
    parent_id: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_discovery_map_html(
    records: Iterable[RegionRecord],
    discovery: DiscoveryManifest,
    *,
    title: str = "Refmark Discovery Cluster Map",
) -> str:
    """Render discovery clusters as an inspectable drill-down treemap."""

    items = discovery_map_items(records, discovery)
    summary = {
        "title": title,
        "regions": discovery.regions,
        "tokens": discovery.corpus_tokens,
        "clusters": len(items),
        "strategy": _dominant(item.strategy for item in items) or "unknown",
        "source": discovery.source,
        "model": discovery.model,
        "terms": _top_terms(items, 18),
    }
    return (
        _HTML_TEMPLATE.replace("__DATA__", _json_for_script([item.to_dict() for item in items]))
        .replace("__SUMMARY__", _json_for_script(summary))
        .replace("__TITLE__", _escape_html(title))
    )


def write_discovery_map_html(
    records: Iterable[RegionRecord],
    discovery: DiscoveryManifest,
    path: str | Path,
    *,
    title: str = "Refmark Discovery Cluster Map",
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_discovery_map_html(records, discovery, title=title), encoding="utf-8")


def discovery_map_items(records: Iterable[RegionRecord], discovery: DiscoveryManifest) -> list[DiscoveryMapItem]:
    by_ref = {f"{record.doc_id}:{record.region_id}": record for record in records}
    assigned: set[str] = set()
    items: list[DiscoveryMapItem] = []
    for cluster in discovery.clusters:
        refs = [ref for ref in cluster.refs if ref in by_ref]
        if not refs:
            continue
        assigned.update(refs)
        rows = [by_ref[ref] for ref in refs]
        items.append(
            DiscoveryMapItem(
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                strategy=cluster.strategy,
                source=cluster.source,
                refs=refs,
                terms=cluster.terms,
                tokens=sum(approx_tokens(record.text) for record in rows),
                regions=len(rows),
                sample_titles=[_title(record) for record in rows[:8]],
                parent_id=cluster.parent_id,
                blocks=[_block(record) for record in rows],
                notes=cluster.notes,
            )
        )
    unassigned = [record for ref, record in by_ref.items() if ref not in assigned]
    if unassigned:
        items.append(
            DiscoveryMapItem(
                cluster_id="unclustered",
                name="unclustered",
                strategy="fallback",
                source="manifest",
                refs=[f"{record.doc_id}:{record.region_id}" for record in unassigned],
                terms=[],
                tokens=sum(approx_tokens(record.text) for record in unassigned),
                regions=len(unassigned),
                sample_titles=[_title(record) for record in unassigned[:8]],
                parent_id=None,
                blocks=[_block(record) for record in unassigned],
                notes="Refs not covered by the discovery cluster manifest.",
            )
        )
    return items


def _block(record: RegionRecord) -> dict[str, Any]:
    return {
        "ref": f"{record.doc_id}:{record.region_id}",
        "title": _title(record),
        "text": record.text[:1200],
        "tokens": approx_tokens(record.text),
        "source_path": record.source_path,
    }


def _title(record: RegionRecord) -> str:
    lines = record.text.splitlines()
    for index, line in enumerate(lines[:-1]):
        cleaned = line.strip()
        next_line = lines[index + 1].strip()
        if _is_rst_adornment(cleaned) and next_line and not _is_rst_adornment(next_line):
            return next_line[:120]
        if cleaned and not _is_rst_adornment(cleaned) and _is_rst_adornment(next_line):
            return cleaned.lstrip("#").strip()[:120]
    for line in lines:
        cleaned = line.strip().lstrip("#").strip()
        if cleaned and not _is_rst_adornment(cleaned):
            return cleaned[:120]
    return f"{record.doc_id}:{record.region_id}"


def _is_rst_adornment(value: str) -> bool:
    return bool(re.fullmatch(r"([=\-~`:#\"'^_*+])\1{2,}", value.strip()))


def _top_terms(items: list[DiscoveryMapItem], limit: int) -> list[dict[str, Any]]:
    counts = Counter(term for item in items for term in item.terms[:8])
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def _dominant(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    return counts.most_common(1)[0][0] if counts else ""


def _json_for_script(value: Any) -> str:
    """Serialize JSON safely inside an inline script tag."""

    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f8fa;--text:#1f2328;--muted:#57606a;--line:#d0d7de}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif}
header{padding:18px 22px 8px}h1{margin:0;font-size:24px}.sub{margin-top:6px;color:var(--muted);line-height:1.35}
.toolbar{padding:8px 22px 6px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar input{font:inherit;font-size:13px;padding:5px 8px;width:310px}.toolbar select{font:inherit;font-size:13px;padding:5px 8px}.button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}
.breadcrumbsbar{padding:0 22px 12px;color:var(--muted);font-size:13px}
.wrap{display:flex;gap:16px;padding:0 18px 20px}.canvas{position:relative;width:1260px;height:780px;background:#fff;border:1px solid var(--line);box-shadow:0 2px 8px #0001;overflow:hidden;flex:0 0 auto}
aside{width:480px;max-height:780px;overflow:auto;background:#fff;border:1px solid var(--line);box-shadow:0 2px 8px #0001;padding:14px 16px}
.tile{position:absolute;border:1px solid rgba(31,35,40,.22);overflow:hidden;padding:7px;cursor:pointer;color:#111;background:#2fb344}
.tile:hover,.tile.selected{outline:3px solid #0969da;z-index:10}.tile.search-dim{opacity:.22}.tile.search-hit{outline:3px solid #8250df;z-index:9}
.name{font-weight:700;line-height:1.12;text-shadow:0 1px 0 #fff8}.tiny{padding:2px}.tile-badges{margin-top:5px;display:flex;flex-wrap:wrap;gap:3px}.tile-badge{font-size:10px;line-height:1;border:1px solid #0002;background:#fff9;border-radius:999px;padding:2px 5px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tile-samples{margin-top:5px;font-size:10px;line-height:1.18;color:#111d}.tile-samples div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
h2{font-size:16px;margin:12px 0 8px}.small{font-size:12px;color:var(--muted);line-height:1.4}.detail{background:#f6f8fa;border:1px solid var(--line);padding:10px;margin-top:8px}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 6px;margin:2px;background:#fff;font-size:11px}.crumb{border:0;background:transparent;color:#0969da;cursor:pointer;padding:0;font:inherit}.crumb:hover{text-decoration:underline}.block-text{white-space:pre-wrap;max-height:300px;overflow:auto;background:#fff;border:1px solid var(--line);padding:8px;color:#24292f}
ol{padding-left:20px}li{margin:4px 0}
</style>
</head>
<body>
<header><h1 id="title"></h1><div class="sub">Area is proportional to source tokens. Click a cluster to drill into its member blocks, then use breadcrumbs to return.</div></header>
<div class="toolbar"><input id="search" type="search" placeholder="Find cluster, term, ref, title..."><select id="layout"><option value="ordered">Ordered layout</option><option value="balanced">Balanced layout</option></select><button id="clear" class="button">Clear</button><span id="stats" class="small"></span></div>
<div id="breadcrumbs" class="breadcrumbsbar"></div>
<div class="wrap"><div id="canvas" class="canvas"></div><aside><h2 id="panelTitle">Overview</h2><div id="detail" class="detail small"></div></aside></div>
<script>
const DATA=__DATA__;
const SUMMARY=__SUMMARY__;
const canvas=document.getElementById('canvas'), detail=document.getElementById('detail'), search=document.getElementById('search'), layout=document.getElementById('layout'), stats=document.getElementById('stats'), breadcrumbs=document.getElementById('breadcrumbs'), panelTitle=document.getElementById('panelTitle');
const CHILDREN=new Map();
for(const item of DATA){if(item.parent_id){if(!CHILDREN.has(item.parent_id))CHILDREN.set(item.parent_id,[]); CHILDREN.get(item.parent_id).push(item)}}
let stack=[]; let view={level:'clusters',parent:null,cluster:null}; let selected=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function textOf(item){return [item.name,item.cluster_id,item.strategy,item.source,item.ref,item.title,...(item.refs||[]),...(item.terms||[]),...(item.sample_titles||[])].join(' ').toLowerCase()}
function matches(item){const q=search.value.trim().toLowerCase(); return !q||textOf(item).includes(q)}
function topClusters(){return DATA.filter(item=>!item.parent_id)}
function currentItems(filter=true){let rows=view.level==='blocks'?(view.cluster?.blocks||[]):(view.parent?CHILDREN.get(view.parent.cluster_id)||[]:topClusters()); return filter?rows.filter(matches):rows}
function color(item){const rows=currentItems(false); const avg=rows.reduce((s,i)=>s+(i.tokens||1),0)/Math.max(1,rows.length); const ratio=Math.min(2.2,(item.tokens||1)/Math.max(1,avg)); const base=view.level==='blocks'?142:130; return `hsl(${base-Math.min(72,ratio*30)} 60% 53%)`}
function balancedTreemap(items,x,y,w,h,vertical=true){items=items.filter(i=>(i.tokens||1)>0); if(!items.length)return []; if(items.length===1)return [{item:items[0],x,y,w,h}]; items=[...items].sort((a,b)=>(b.tokens||1)-(a.tokens||1)); return splitTreemap(items,x,y,w,h,vertical)}
function orderedTreemap(items,x,y,w,h,vertical=true){items=items.filter(i=>(i.tokens||1)>0); if(!items.length)return []; if(items.length===1)return [{item:items[0],x,y,w,h}]; return splitTreemap(items,x,y,w,h,vertical)}
function splitTreemap(items,x,y,w,h,vertical=true){if(items.length<=1)return items.length?[{item:items[0],x,y,w,h}]:[]; const total=items.reduce((s,i)=>s+(i.tokens||1),0); let acc=0,split=1,best=Infinity; for(let i=0;i<items.length;i++){const next=acc+(items[i].tokens||1),d=Math.abs(total/2-next); if(d<best){best=d;split=i+1;acc=next}else break} if(split>=items.length)split=Math.max(1,Math.floor(items.length/2)); const left=items.slice(0,split),right=items.slice(split); const lv=left.reduce((s,i)=>s+(i.tokens||1),0); if(vertical){const w1=w*lv/total; return splitTreemap(left,x,y,w1,h,!vertical).concat(splitTreemap(right,x+w1,y,w-w1,h,!vertical))} const h1=h*lv/total; return splitTreemap(left,x,y,w,h1,!vertical).concat(splitTreemap(right,x,y+h1,w,h-h1,!vertical))}
function layoutRects(items){return layout.value==='balanced'?balancedTreemap(items,0,0,1260,780):orderedTreemap(items,0,0,1260,780)}
function displayName(item){let name=String(item.name||item.title||item.ref||''); if(view.parent?.name){const prefix=view.parent.name+' / '; if(name.startsWith(prefix))name=name.slice(prefix.length)} return name}
function titleSize(w,h){const m=Math.min(w,h); if(m<28)return 8; if(m<42)return 9; if(m<58)return 11; if(m<82)return 13; if(m<120)return 15; return 18}
function tileHtml(item,w,h){if(w<=18||h<=14)return ''; const title=esc(displayName(item)); let html=`<div class="name" style="font-size:${titleSize(w,h)}px">${title}</div>`; if(w>160&&h>110&&item.terms?.length){html+=`<div class="tile-badges">${item.terms.slice(0,w>260?8:5).map(t=>`<span class="tile-badge">${esc(t)}</span>`).join('')}</div>`} if(w>220&&h>170&&item.sample_titles?.length){html+=`<div class="tile-samples">${item.sample_titles.slice(0,4).map(t=>`<div>${esc(t)}</div>`).join('')}</div>`} return html}
function render(){canvas.innerHTML=''; const all=currentItems(false), visible=currentItems(true), q=search.value.trim(); stats.textContent=`${visible.length}/${all.length} ${view.level==='blocks'?'blocks':'clusters'} visible`+(q?` for "${q}"`:''); renderCrumbs(); for(const r of layoutRects(all)){const item=r.item, hit=matches(item), id=item.cluster_id||item.ref; const el=document.createElement('div'); el.className='tile'+(r.w<56||r.h<34?' tiny':'')+(q&&!hit?' search-dim':'')+(q&&hit?' search-hit':'')+(selected===id?' selected':''); el.style.left=r.x+'px'; el.style.top=r.y+'px'; el.style.width=r.w+'px'; el.style.height=r.h+'px'; el.style.background=color(item); el.innerHTML=tileHtml(item,r.w,r.h); el.onclick=()=>{selected=id; view.level==='clusters'?drill(item):showBlock(item); render()}; canvas.appendChild(el)}}
function drill(item){const kids=CHILDREN.get(item.cluster_id)||[]; if(kids.length){stack.push(item); view={level:'clusters',parent:item,cluster:null}; panelTitle.textContent='Cluster'; detail.innerHTML=clusterDetail(item,kids); return} stack.push(item); view={level:'blocks',parent:null,cluster:item}; panelTitle.textContent='Cluster'; detail.innerHTML=clusterDetail(item,[])+`<h2>Contained Blocks</h2><ol>${(item.blocks||[]).slice(0,160).map(b=>`<li><code>${esc(b.ref)}</code> ${esc(b.title)}</li>`).join('')}</ol>`}
function clusterDetail(item,kids){const childHtml=kids.length?`<h2>Subclusters</h2><ol>${kids.slice(0,40).map(k=>`<li><button class="crumb" onclick="selected='${esc(k.cluster_id)}';drillById('${esc(k.cluster_id)}');render()">${esc(k.name)}</button> <span class="small">${(k.refs||[]).length} refs</span><br>${(k.terms||[]).slice(0,6).map(t=>`<span class="pill">${esc(t)}</span>`).join('')}</li>`).join('')}</ol>`:''; const samples=(item.sample_titles||[]).length?`<h2>Sample Topics</h2><ol>${item.sample_titles.slice(0,8).map(t=>`<li>${esc(t)}</li>`).join('')}</ol>`:''; return `<div class="name">${esc(item.name)}</div><div>${(item.terms||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join('')}</div><div class="small">${kids.length?`${kids.length} child clusters · `:''}${(item.refs||[]).length} refs · ${(item.tokens||0)} tokens</div>${item.notes?`<p>${esc(item.notes)}</p>`:''}${childHtml}${samples}`}
function drillById(id){const item=DATA.find(row=>row.cluster_id===id); if(item)drill(item)}
function showBlock(item){panelTitle.textContent='Block'; detail.innerHTML=`<div class="name">${esc(item.title)}</div><div><code>${esc(item.ref)}</code></div>${item.source_path?`<div class="small">${esc(item.source_path)}</div>`:''}<div class="block-text">${esc(item.text)}</div>`}
function gotoCrumb(index){if(index<0){stack=[]; view={level:'clusters',parent:null,cluster:null}; selected=null; render(); return} stack=stack.slice(0,index+1); const item=stack[stack.length-1]; view={level:'clusters',parent:item,cluster:null}; selected=item.cluster_id; render()}
function renderCrumbs(){if(!stack.length&&view.level==='clusters'){breadcrumbs.innerHTML='Overview'; panelTitle.textContent='Overview'; if(!selected) detail.innerHTML=`<div>${SUMMARY.clusters} clusters &middot; strategy <code>${esc(SUMMARY.strategy)}</code></div><h2>Common Terms</h2>${SUMMARY.terms.map(t=>`<span class="pill">${esc(t.term)}</span>`).join('')}`; return} const parts=[`<button class="crumb" onclick="gotoCrumb(-1)">Overview</button>`]; stack.forEach((item,i)=>parts.push(`<button class="crumb" onclick="gotoCrumb(${i})">${esc(item.name)}</button>`)); breadcrumbs.innerHTML=parts.join(' / ')}
document.getElementById('title').textContent=SUMMARY.title;
search.oninput=render; layout.onchange=render; document.getElementById('clear').onclick=()=>{search.value='';render();search.focus()}; render();
</script>
</body>
</html>
"""
