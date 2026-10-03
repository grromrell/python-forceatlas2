import os
import json
import math
import numpy as np
from forceatlas import ForceAtlas2

def make_clusters(clusters=3, size=15, p_in=0.6, p_out=0.03, seed=42):
    """3 community clusters with dense internal and sparse external connections."""
    np.random.seed(seed)
    N = clusters * size
    adj = np.zeros((N, N), dtype=float)
    groups = []
    for i in range(N):
        groups.append(i // size)
        for j in range(i + 1, N):
            c_i = i // size
            c_j = j // size
            p = p_in if c_i == c_j else p_out
            if np.random.rand() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj, groups

def make_barbell(n_bell=16, n_bridge=6):
    """Two dense clusters connected by a linear bridge."""
    N = 2 * n_bell + n_bridge
    adj = np.zeros((N, N), dtype=float)
    groups = [0] * n_bell + [1] * n_bridge + [2] * n_bell
    # Bell 1 (clique-like)
    for i in range(n_bell):
        for j in range(i + 1, n_bell):
            if np.random.rand() < 0.8:
                adj[i, j] = adj[j, i] = 1.0
    # Bridge
    prev = n_bell - 1
    for k in range(n_bridge):
        curr = n_bell + k
        adj[prev, curr] = adj[curr, prev] = 1.0
        prev = curr
    # Connect bridge end to Bell 2 start
    bell2_start = n_bell + n_bridge
    adj[prev, bell2_start] = adj[bell2_start, prev] = 1.0
    # Bell 2 (clique-like)
    for i in range(bell2_start, N):
        for j in range(i + 1, N):
            if np.random.rand() < 0.8:
                adj[i, j] = adj[j, i] = 1.0
    return adj, groups

def make_star(spokes=24):
    """Central hub node with spokes and outer ring connections."""
    N = spokes + 1
    adj = np.zeros((N, N), dtype=float)
    groups = [0] + [1] * spokes
    # Central hub (node 0) connects to all spokes
    for i in range(1, N):
        adj[0, i] = adj[i, 0] = 1.0
        # Connect spoke to neighbor spoke with some probability
        next_spoke = 1 + (i % spokes)
        if np.random.rand() < 0.3:
            adj[i, next_spoke] = adj[next_spoke, i] = 1.0
    return adj, groups

def record_layout_history(adj, iterations=80, **kwargs):
    """Runs FA2 step-by-step and records normalized node coordinates at each step."""
    fa = ForceAtlas2(adj, iterations=1, **kwargs)
    history = []
    edges = [(int(fa.edges_src[e]), int(fa.edges_dst[e])) for e in range(len(fa.edges_src))]
    
    for it in range(iterations):
        fa.go_algo()
        pos = fa.pos.copy()
        # Normalize to [-1, 1] preserving aspect ratio
        min_xy = pos.min(axis=0)
        max_xy = pos.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        span = max((max_xy - min_xy).max(), 1e-6) / 1.8
        norm_pos = (pos - center) / span
        history.append(norm_pos.tolist())
        
    return edges, history

def render_svg(history_frame, edges, groups, width=600, height=600, title=""):
    """Renders a single static frame as clean standalone SVG."""
    palette = ["#38bdf8", "#f43f5e", "#a855f7", "#10b981", "#f59e0b"]
    nodes = history_frame
    pad = 40
    w_avail = width - 2 * pad
    h_avail = height - 2 * pad
    
    # Scale from [-1, 1] to [pad, pad + w_avail]
    def to_svg(pt):
        x = pad + (pt[0] + 1.0) * 0.5 * w_avail
        y = pad + (pt[1] + 1.0) * 0.5 * h_avail
        return x, y

    lines = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:12px; font-family:sans-serif;">',
        f'<text x="20" y="30" fill="#94a3b8" font-size="16" font-weight="bold">{title}</text>',
        '<g stroke="#334155" stroke-width="1.2" stroke-opacity="0.6">'
    ]
    for u, v in edges:
        x1, y1 = to_svg(nodes[u])
        x2, y2 = to_svg(nodes[v])
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />')
    lines.append('</g>')
    
    lines.append('<g>')
    for i, pt in enumerate(nodes):
        x, y = to_svg(pt)
        color = palette[groups[i] % len(palette)]
        r = 7.0 if groups[i] == 0 and len(set(groups)) > 1 else 5.0
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')
    lines.append('</g>')
    lines.append('</svg>')
    return "\n".join(lines)

def build_interactive_html(dataset, filepath):
    """Builds a self-contained HTML visualizer with timeline playback and graph switcher."""
    data_json = json.dumps(dataset)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForceAtlas2 Network Layout Visualizer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 24px; }}
  h1 {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; color: #38bdf8; letter-spacing: -0.5px; }}
  p.sub {{ font-size: 13px; color: #94a3b8; margin-bottom: 20px; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: flex; flex-direction: column; align-items: center; width: 680px; max-width: 100%; }}
  .toolbar {{ display: flex; gap: 12px; align-items: center; width: 100%; margin-bottom: 14px; justify-content: space-between; }}
  select, button {{ background: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 7px 14px; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.15s ease; }}
  select:hover, button:hover {{ background: #334155; border-color: #475569; }}
  button.active {{ background: #38bdf8; color: #0f172a; font-weight: 600; border-color: #38bdf8; }}
  canvas {{ background: #0f172a; border-radius: 10px; width: 640px; height: 500px; display: block; }}
  .controls {{ display: flex; align-items: center; gap: 14px; width: 100%; margin-top: 14px; }}
  input[type=range] {{ flex: 1; accent-color: #38bdf8; cursor: pointer; }}
  .badge {{ font-size: 12px; font-weight: 600; background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 6px; min-width: 90px; text-align: center; border: 1px solid #334155; }}
  .stats {{ display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: #64748b; width: 100%; justify-content: flex-end; }}
  .stats span strong {{ color: #cbd5e1; }}
</style>
</head>
<body>
<h1>ForceAtlas2 Layout Engine</h1>
<p class="sub">Native C & Vectorized NumPy Simulation</p>

<div class="card">
  <div class="toolbar">
    <select id="graphSelect">
      <option value="clusters">Community Clusters (3 Groups)</option>
      <option value="barbell">Barbell Network</option>
      <option value="star">Star / Hub-and-Spoke</option>
    </select>
    <div style="display:flex; gap:8px;">
      <button id="btnPlay">▶ Play</button>
      <button id="btnReset">↺ Reset</button>
    </div>
  </div>

  <canvas id="cv" width="640" height="500"></canvas>

  <div class="controls">
    <span class="badge" id="iterBadge">Iter: 0</span>
    <input type="range" id="iterSlider" min="0" max="79" value="0">
  </div>

  <div class="stats">
    <span>Nodes: <strong id="lblNodes">0</strong></span>
    <span>Edges: <strong id="lblEdges">0</strong></span>
    <span>Engine: <strong style="color:#10b981;">Native C (pthreads)</strong></span>
  </div>
</div>

<script>
const DATA = {data_json};
const palette = ["#38bdf8", "#f43f5e", "#a855f7", "#10b981", "#f59e0b", "#ec4899"];

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const select = document.getElementById("graphSelect");
const slider = document.getElementById("iterSlider");
const btnPlay = document.getElementById("btnPlay");
const btnReset = document.getElementById("btnReset");
const iterBadge = document.getElementById("iterBadge");
const lblNodes = document.getElementById("lblNodes");
const lblEdges = document.getElementById("lblEdges");

let currentKey = "clusters";
let frameIdx = 0;
let isPlaying = false;
let animTimer = null;

function renderFrame() {{
  const g = DATA[currentKey];
  const frame = g.history[frameIdx];
  const edges = g.edges;
  const groups = g.groups;
  const w = cv.width;
  const h = cv.height;
  const pad = 40;

  ctx.clearRect(0, 0, w, h);

  // Draw edges
  ctx.strokeStyle = "rgba(71, 85, 105, 0.45)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let e = 0; e < edges.length; e++) {{
    const u = edges[e][0];
    const v = edges[e][1];
    const x1 = pad + (frame[u][0] + 1) * 0.5 * (w - 2 * pad);
    const y1 = pad + (frame[u][1] + 1) * 0.5 * (h - 2 * pad);
    const x2 = pad + (frame[v][0] + 1) * 0.5 * (w - 2 * pad);
    const y2 = pad + (frame[v][1] + 1) * 0.5 * (h - 2 * pad);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }}
  ctx.stroke();

  // Draw nodes
  for (let i = 0; i < frame.length; i++) {{
    const x = pad + (frame[i][0] + 1) * 0.5 * (w - 2 * pad);
    const y = pad + (frame[i][1] + 1) * 0.5 * (h - 2 * pad);
    const grp = groups[i] || 0;
    const r = (grp === 0 && currentKey === "star") ? 7.5 : 5.0;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = palette[grp % palette.length];
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }}

  iterBadge.innerText = `Iter: ${{frameIdx}}`;
  slider.value = frameIdx;
}}

function updateMeta() {{
  const g = DATA[currentKey];
  slider.max = g.history.length - 1;
  lblNodes.innerText = g.history[0].length;
  lblEdges.innerText = g.edges.length;
}}

function step() {{
  const maxFrames = DATA[currentKey].history.length;
  if (frameIdx < maxFrames - 1) {{
    frameIdx++;
    renderFrame();
  }} else {{
    pause();
  }}
}}

function play() {{
  if (isPlaying) return;
  isPlaying = true;
  btnPlay.innerText = "⏸ Pause";
  btnPlay.classList.add("active");
  animTimer = setInterval(step, 40);
}}

function pause() {{
  isPlaying = false;
  btnPlay.innerText = "▶ Play";
  btnPlay.classList.remove("active");
  if (animTimer) clearInterval(animTimer);
}}

btnPlay.onclick = () => isPlaying ? pause() : play();
btnReset.onclick = () => {{ pause(); frameIdx = 0; renderFrame(); }};

slider.oninput = (e) => {{
  pause();
  frameIdx = parseInt(e.target.value, 10);
  renderFrame();
}};

select.onchange = (e) => {{
  pause();
  currentKey = e.target.value;
  frameIdx = 0;
  updateMeta();
  renderFrame();
}};

// Initialize
updateMeta();
renderFrame();
setTimeout(play, 300);
</script>
</body>
</html>
"""
    with open(filepath, "w") as f:
        f.write(html_content)

def run_visual_suite():
    print("=" * 60)
    print("ForceAtlas2 Visual Verification Suite")
    print("=" * 60)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = {}

    graphs = [
        ("clusters", "3 Community Clusters", make_clusters(), {"gravity": 1.2, "scaling_ratio": 3.0}),
        ("barbell", "Barbell Graph", make_barbell(), {"gravity": 1.0, "scaling_ratio": 2.5}),
        ("star", "Star Hub Network", make_star(), {"gravity": 1.5, "scaling_ratio": 3.5, "outbound_attraction_distribution": True}),
    ]

    for key, title, (adj, groups), params in graphs:
        print(f"-> Simulating {title} (nodes: {len(adj)}, edges: {int(adj.sum()//2)})...")
        edges, history = record_layout_history(adj, iterations=80, **params)
        dataset[key] = {"edges": edges, "history": history, "groups": groups}
        
        # Verify sanity of positions (no NaN, non-zero spread)
        final_frame = np.array(history[-1])
        assert not np.isnan(final_frame).any(), f"NaNs detected in {title}"
        assert not np.isinf(final_frame).any(), f"Infs detected in {title}"
        spread = final_frame.std()
        assert spread > 0.1, f"Nodes collapsed to a single point in {title}"

        # Save static SVG of final state
        svg_content = render_svg(history[-1], edges, groups, title=f"{title} (FA2 Layout)")
        svg_path = os.path.join(out_dir, f"demo_{key}.svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
        print(f"   [Saved] {os.path.basename(svg_path)}")

    # Save interactive HTML
    html_path = os.path.join(out_dir, "layout_demo.html")
    build_interactive_html(dataset, html_path)
    print(f"-> [Saved Interactive Visualizer] {html_path}")
    print("=" * 60)
    print("ALL VISUAL TESTS VERIFIED SUCCESSFULLY.")
    print(f"Open in browser: file://{html_path}")
    print("=" * 60)
    return html_path

# Pytest discovery entrypoint
def test_visual_layout_generation():
    html_path = run_visual_suite()
    assert os.path.exists(html_path)
    assert os.path.getsize(html_path) > 1000

if __name__ == "__main__":
    run_visual_suite()
