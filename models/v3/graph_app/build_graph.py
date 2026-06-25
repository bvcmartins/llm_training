#!/usr/bin/env python3
"""Build a self-contained interactive code graph for models/v3/src/.

Parses every .py in SRC_DIR with the `ast` module and emits a single
`index.html` that needs no server-side code: an interactive node graph on the
left (drag / zoom / click, via vis-network) and a syntax-highlighted source
panel on the right (highlight.js) that shows the FULL code body of whichever
node you click.

Node kinds : module | class | function | method
Edge kinds : contains (module→def, class→method)
             imports  (module→module, from first-party imports)
             calls    (function/method → function/class it references)

Run:  python build_graph.py   →   writes ./index.html next to this script.
Then serve the folder (see serve.sh) for remote/LAN access.
"""

from __future__ import annotations

import ast
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
OUT = HERE / "index.html"

KIND_COLOR = {
    "module":   "#4f83cc",   # blue
    "class":    "#5cb85c",   # green
    "function": "#e8913a",   # orange
    "method":   "#f0c674",   # light orange
}
EDGE_STYLE = {
    "contains": {"color": "#bbbbbb", "dashes": False, "arrow": True},
    "imports":  {"color": "#4f83cc", "dashes": False, "arrow": True},
    "calls":    {"color": "#e8913a", "dashes": True,  "arrow": True},
}


def src_segment(lines: list[str], node: ast.AST) -> str:
    """Full source for a def/class node, including any decorators."""
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(start, min(d.lineno for d in node.decorator_list))
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(lines[start - 1:end])


def callee_name(call: ast.Call) -> str | None:
    """Simple name of a call target: foo(...) -> 'foo', a.b.foo(...) -> 'foo'."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def build():
    py_files = sorted(p for p in SRC_DIR.glob("*.py") if p.name != "__init__.py")
    module_names = {p.stem for p in py_files}

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    name_index: dict[str, list[str]] = {}   # simple name -> [node_id,...]

    def add_node(nid, label, kind, module, source, signature=""):
        nodes[nid] = {
            "id": nid, "label": label, "kind": kind, "module": module,
            "source": source, "signature": signature,
        }
        name_index.setdefault(label.split(".")[-1], []).append(nid)

    # --- pass 1: collect nodes -------------------------------------------
    file_calls: list[tuple[str, set[str]]] = []   # (node_id, called simple names)

    for path in py_files:
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        mod = path.stem
        tree = ast.parse(text, filename=path.name)

        mod_id = f"mod:{mod}"
        add_node(mod_id, mod + ".py", "module", mod, text)

        for top in tree.body:
            if isinstance(top, ast.FunctionDef):
                nid = f"{mod}:{top.name}"
                source = src_segment(lines, top)
                add_node(nid, top.name, "function", mod, source)
                edges.append({"from": mod_id, "to": nid, "type": "contains"})
                file_calls.append((nid, _calls_in(top)))

            elif isinstance(top, ast.ClassDef):
                cid = f"{mod}:{top.name}"
                source = src_segment(lines, top)
                add_node(cid, top.name, "class", mod, source)
                edges.append({"from": mod_id, "to": cid, "type": "contains"})
                file_calls.append((cid, _calls_in(top, recurse=False)))

                for item in top.body:
                    if isinstance(item, ast.FunctionDef):
                        mid = f"{mod}:{top.name}.{item.name}"
                        msrc = src_segment(lines, item)
                        add_node(mid, f"{top.name}.{item.name}", "method", mod, msrc)
                        edges.append({"from": cid, "to": mid, "type": "contains"})
                        file_calls.append((mid, _calls_in(item)))

        # --- import edges (first-party only) ---
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module in module_names:
                edges.append({"from": mod_id, "to": f"mod:{n.module}", "type": "imports"})
            elif isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.name in module_names:
                        edges.append({"from": mod_id, "to": f"mod:{alias.name}", "type": "imports"})

    # --- pass 2: call / instantiation edges ------------------------------
    seen = {(e["from"], e["to"], e["type"]) for e in edges}
    for nid, names in file_calls:
        for name in names:
            for target in name_index.get(name, []):
                if target == nid:
                    continue
                if nodes[target]["kind"] == "module":
                    continue
                key = (nid, target, "calls")
                if key not in seen:
                    seen.add(key)
                    edges.append({"from": nid, "to": target, "type": "calls"})

    return nodes, edges


def _calls_in(node: ast.AST, recurse: bool = True) -> set[str]:
    """Simple names of everything called within `node`'s body."""
    names: set[str] = set()
    walker = ast.walk(node) if recurse else node.body
    pool = ast.walk(node) if recurse else _shallow(node)
    for n in pool:
        if isinstance(n, ast.Call):
            cn = callee_name(n)
            if cn:
                names.add(cn)
    return names


def _shallow(node):
    out = []
    for item in getattr(node, "body", []):
        for sub in ast.walk(item):
            if not isinstance(sub, ast.FunctionDef):
                out.append(sub)
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_TMPL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v3 src — code graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
  :root { --bg:#1e1e1e; --panel:#252526; --fg:#d4d4d4; --border:#3c3c3c; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  #app { display:flex; height:100vh; }
  #left { flex:1 1 58%; position:relative; border-right:1px solid var(--border); }
  #graph { position:absolute; inset:0; }
  #right { flex:1 1 42%; display:flex; flex-direction:column; min-width:340px; }
  #bar { padding:8px 12px; background:var(--panel); border-bottom:1px solid var(--border); display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  #title { font-weight:600; font-size:14px; }
  #kind { font-size:11px; padding:1px 7px; border-radius:10px; color:#111; }
  #code { flex:1; overflow:auto; margin:0; }
  #code pre { margin:0; }
  #code code { font-size:12.5px; line-height:1.5; }
  .legend { position:absolute; top:8px; left:8px; background:rgba(37,37,38,.92); border:1px solid var(--border);
            border-radius:6px; padding:8px 10px; font-size:11px; z-index:5; }
  .legend .row { display:flex; align-items:center; gap:6px; margin:2px 0; }
  .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
  .ctrls { position:absolute; top:8px; right:8px; z-index:5; display:flex; gap:6px; }
  .ctrls input { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:5px; padding:4px 7px; font-size:12px; }
  .ctrls button { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:5px; padding:4px 9px; font-size:12px; cursor:pointer; }
  .hint { color:#888; padding:18px; font-size:13px; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="graph"></div>
    <div class="legend" id="legend"></div>
    <div class="ctrls">
      <input id="search" placeholder="find node…" autocomplete="off">
      <button id="toggle">hierarchical</button>
      <button id="fit">fit</button>
    </div>
  </div>
  <div id="right">
    <div id="bar"><span id="kind"></span><span id="title">click a node</span></div>
    <div id="code"><div class="hint">Click any node to see its full source here. Edges: grey=contains, blue=imports, orange-dashed=calls.</div></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const KIND_COLOR = __KIND_COLOR__;
const EDGE_STYLE = __EDGE_STYLE__;

const nodes = new vis.DataSet(DATA.nodes.map(n => ({
  id: n.id, label: n.label, group: n.kind,
  color: { background: KIND_COLOR[n.kind], border: "#222",
           highlight: { background: KIND_COLOR[n.kind], border: "#fff" } },
  shape: n.kind === "module" ? "box" : (n.kind === "class" ? "ellipse" : "dot"),
  font: { color: "#eee", size: n.kind === "module" ? 16 : 13 },
  size: n.kind === "module" ? 22 : (n.kind === "class" ? 18 : 12),
})));
const edges = new vis.DataSet(DATA.edges.map((e,i) => {
  const s = EDGE_STYLE[e.type];
  return { id:i, from:e.from, to:e.to, arrows: s.arrow ? "to" : "",
           dashes:s.dashes, color:{color:s.color, opacity:0.7}, width: e.type==="imports"?2:1 };
}));
const srcById = {}; DATA.nodes.forEach(n => srcById[n.id] = n);

const container = document.getElementById("graph");
const baseOpts = {
  physics: { stabilization: true, barnesHut: { gravitationalConstant:-7000, springLength:120, springConstant:0.04 } },
  interaction: { hover:true, tooltipDelay:120, navigationButtons:false },
  nodes: { borderWidth:1.5 },
};
const network = new vis.Network(container, { nodes, edges }, baseOpts);

function showNode(id) {
  const n = srcById[id]; if (!n) return;
  const kindEl = document.getElementById("kind");
  kindEl.textContent = n.kind;
  kindEl.style.background = KIND_COLOR[n.kind];
  document.getElementById("title").textContent = n.module + " · " + n.label;
  const code = document.getElementById("code");
  const pre = document.createElement("pre");
  const c = document.createElement("code");
  c.className = "language-python";
  c.textContent = n.source;
  pre.appendChild(c); code.innerHTML = ""; code.appendChild(pre);
  hljs.highlightElement(c);
  code.scrollTop = 0;
}
network.on("click", p => { if (p.nodes.length) showNode(p.nodes[0]); });

// legend
const legend = document.getElementById("legend");
legend.innerHTML = "<b>nodes</b>" + Object.entries(KIND_COLOR).map(([k,c]) =>
  `<div class="row"><span class="dot" style="background:${c}"></span>${k}</div>`).join("")
  + '<b>edges</b><div class="row"><span style="color:#bbb">━</span> contains</div>'
  + '<div class="row"><span style="color:#4f83cc">━</span> imports</div>'
  + '<div class="row"><span style="color:#e8913a">┅</span> calls</div>';

// search
document.getElementById("search").addEventListener("input", e => {
  const q = e.target.value.toLowerCase().trim();
  if (!q) return;
  const hit = DATA.nodes.find(n => (n.module+" "+n.label).toLowerCase().includes(q));
  if (hit) { network.focus(hit.id, { scale:1.1, animation:true }); network.selectNodes([hit.id]); showNode(hit.id); }
});
// layout toggle
let hier = false;
document.getElementById("toggle").addEventListener("click", () => {
  hier = !hier;
  network.setOptions(hier
    ? { layout:{ hierarchical:{ enabled:true, direction:"LR", sortMethod:"directed", levelSeparation:220, nodeSpacing:60 } }, physics:false }
    : { layout:{ hierarchical:{ enabled:false } }, physics:baseOpts.physics });
});
document.getElementById("fit").addEventListener("click", () => network.fit({ animation:true }));
</script>
</body>
</html>
"""


def render(nodes, edges):
    data = {"nodes": list(nodes.values()), "edges": edges}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = (HTML_TMPL
           .replace("__DATA__", blob)
           .replace("__KIND_COLOR__", json.dumps(KIND_COLOR))
           .replace("__EDGE_STYLE__", json.dumps(EDGE_STYLE)))
    return out


if __name__ == "__main__":
    nodes, edges = build()
    OUT.write_text(render(nodes, edges), encoding="utf-8")
    kinds = {}
    for n in nodes.values():
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    etypes = {}
    for e in edges:
        etypes[e["type"]] = etypes.get(e["type"], 0) + 1
    print(f"wrote {OUT}")
    print(f"  nodes: {len(nodes)}  {kinds}")
    print(f"  edges: {len(edges)}  {etypes}")
