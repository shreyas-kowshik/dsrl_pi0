#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve a diagnostics dashboard for DSRL-Pi0 experiments.

Parses a training .sh script to locate the experiment output directory,
discovers all diagnostic .mp4 files, and serves an interactive HTML
dashboard on a local port.

Usage:
    python -m examples.visualize_diagnostics \
        --sh_path examples/scripts/run_cartpole_residual_sac.sh \
        --port 8501

    # Or point directly at a diagnostics directory:
    python -m examples.visualize_diagnostics \
        --diagnostics_dir /data/user_data/skowshik/dsrl_exp/logs/cartpole-rsac/cartpole-rsac_2025_.../diagnostics \
        --port 8501
"""

import argparse
import html
import http.server
import json
import os
import re
import socketserver
import urllib.parse
from pathlib import Path


# ---------------------------------------------------------------------------
# .sh file parsing: extract EXP directory and prefix
# ---------------------------------------------------------------------------

def _collect_shell_vars(content: str) -> dict:
    """Collect simple variable assignments (VAR=value) from shell script."""
    variables = {}
    for line in content.splitlines():
        line = line.strip()
        # Match: VAR=value  or  export VAR=value (no ${ default} form)
        m = re.match(r'(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)=([^\s].*)?$', line)
        if m and not m.group(2, ):
            continue
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip('"').strip("'")
            # Skip lines that are the ${VAR:-default} self-reference pattern
            if val.startswith('${' + key + ':-'):
                val = val[len('${' + key + ':-'):-1]
            variables[key] = val
    return variables


def _resolve_shell_vars(value: str, variables: dict) -> str:
    """Substitute $VAR and ${VAR} references using collected variables."""
    def _replace(m):
        var_name = m.group(1) or m.group(2)
        return variables.get(var_name, m.group(0))
    # Match ${VAR} or $VAR (word-boundary)
    return re.sub(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)', _replace, value)


def parse_sh_for_exp(sh_path: str) -> str:
    """Extract the EXP variable value from a training .sh script."""
    with open(sh_path) as f:
        content = f.read()

    variables = _collect_shell_vars(content)

    # Match patterns like: export EXP=${EXP:-/some/path}  or  export EXP=/some/path
    m = re.search(r'export\s+EXP=\$\{EXP:-([^}]+)\}', content)
    if m:
        raw = m.group(1).strip()
    else:
        m = re.search(r'export\s+EXP=([^\s]+)', content)
        if m:
            raw = m.group(1).strip().strip('"').strip("'")
        else:
            return ""

    return _resolve_shell_vars(raw, variables)


def parse_sh_for_prefix(sh_path: str) -> str:
    """Extract the --prefix argument from the .sh script."""
    with open(sh_path) as f:
        content = f.read()
    m = re.search(r'--prefix\s+([^\s\\]+)', content)
    if m:
        return m.group(1).strip()
    return ""


def find_diagnostics_dirs(exp_base: str, prefix: str) -> list:
    """Find all diagnostics/ directories under EXP that match the prefix."""
    results = []
    if not os.path.isdir(exp_base):
        return results
    for entry in sorted(os.listdir(exp_base)):
        full = os.path.join(exp_base, entry)
        if not os.path.isdir(full):
            continue
        if prefix and not entry.startswith(prefix):
            continue
        diag_dir = os.path.join(full, 'diagnostics')
        if os.path.isdir(diag_dir):
            results.append(diag_dir)
    return results


# ---------------------------------------------------------------------------
# Diagnostics discovery
# ---------------------------------------------------------------------------

PLOT_META = {
    'Q01_multistep_consistency': ('Q-Function', 'Q01: Multi-step Consistency', False),
    'Q02_q_vs_qtarg':           ('Q-Function', 'Q02: Q vs Q-target Scatter', True),
    'Q03_td_error':             ('Q-Function', 'Q03: TD-Error Trajectory', False),
    'Q04_q_base':               ('Q-Function', 'Q04: Q(s, base_action)', False),
    'Q05_q_exec':               ('Q-Function', 'Q05: Q(s, exec_action)', False),
    'Q06_grad_norm_exec':       ('Q-Function', 'Q06: Grad Norm at Exec', False),
    'Q07_grad_norm_base':       ('Q-Function', 'Q07: Grad Norm at Base', False),
    'Q08_td_histogram':         ('Q-Function', 'Q08: TD-Error Histogram', True),
    'Q09_q_variance_base':      ('Q-Function', 'Q09: Q Variance (Base)', False),
    'Q10_q_variance_exec':      ('Q-Function', 'Q10: Q Variance (Exec)', False),
    'E01_delta_q':              ('Actor / Residual', 'E01: Delta-Q', False),
    'E02_action_traces':        ('Actor / Residual', 'E02: Action Traces', False),
    'L01_umap_scatter':         ('Landscape (UMAP)', 'L01: UMAP Scatter', False),
    'L02_grad_ascent_base':     ('Landscape (UMAP)', 'L02: Grad Ascent (Base)', False),
    'L03_grad_ascent_exec':     ('Landscape (UMAP)', 'L03: Grad Ascent (Exec)', False),
    'L04_grad_line_base':       ('Landscape (UMAP)', 'L04: Grad Line (Base)', False),
    'L05_grad_line_exec':       ('Landscape (UMAP)', 'L05: Grad Line (Exec)', False),
}


def discover_diagnostics(diag_dir: str, step_filter: int = None) -> dict:
    """Walk a diagnostics/ directory and return structured metadata.

    Args:
        diag_dir: Path to the diagnostics/ directory.
        step_filter: If set, only include this specific step number.

    Returns:
        {
          "root": "/abs/path/to/diagnostics",
          "steps": [
            {
              "name": "step_0",
              "aggregate": ["Q01_multistep_consistency.mp4", ...],
              "trajectories": {
                "traj_0": ["Q03_td_error.mp4", ...],
                ...
              }
            },
            ...
          ]
        }
    """
    steps = []
    if not os.path.isdir(diag_dir):
        return {"root": diag_dir, "steps": steps}

    for step_name in sorted(os.listdir(diag_dir)):
        step_path = os.path.join(diag_dir, step_name)
        if not os.path.isdir(step_path) or not step_name.startswith('step_'):
            continue
        if step_filter is not None and step_name != f'step_{step_filter}':
            continue
        step_info = {"name": step_name, "aggregate": [], "trajectories": {}}

        agg_path = os.path.join(step_path, 'aggregate')
        if os.path.isdir(agg_path):
            step_info["aggregate"] = sorted(
                f for f in os.listdir(agg_path) if f.endswith('.mp4')
            )

        for traj_name in sorted(os.listdir(step_path)):
            traj_path = os.path.join(step_path, traj_name)
            if not os.path.isdir(traj_path) or not traj_name.startswith('traj_'):
                continue
            step_info["trajectories"][traj_name] = sorted(
                f for f in os.listdir(traj_path) if f.endswith('.mp4')
            )

        steps.append(step_info)
    return {"root": diag_dir, "steps": steps}


# ---------------------------------------------------------------------------
# HTML dashboard generation
# ---------------------------------------------------------------------------

def _video_card(video_url: str, plot_key: str) -> str:
    """Generate HTML for a single video card."""
    cat, title, _ = PLOT_META.get(plot_key, ('Other', plot_key, False))
    return f'''
    <div class="card">
      <div class="card-title">{html.escape(title)}</div>
      <video controls loop muted preload="metadata">
        <source src="{html.escape(video_url)}" type="video/mp4">
      </video>
    </div>'''


def build_dashboard_html(all_experiments: list) -> str:
    """Build the full HTML dashboard page.

    Args:
        all_experiments: list of (exp_label, discovery_dict) tuples.
    """
    sections = []

    for exp_label, disc in all_experiments:
        for step in disc["steps"]:
            step_name = step["name"]
            step_num = step_name.replace("step_", "")

            # --- Aggregate ---
            if step["aggregate"]:
                cards = []
                for fname in step["aggregate"]:
                    key = fname.replace('.mp4', '')
                    url = f'/video?root={urllib.parse.quote(disc["root"])}&path={step_name}/aggregate/{fname}'
                    cards.append(_video_card(url, key))
                sections.append(f'''
                <div class="section">
                  <h2>{html.escape(exp_label)} &mdash; Step {step_num} &mdash; Aggregate</h2>
                  <div class="grid">{"".join(cards)}</div>
                </div>''')

            # --- Per-trajectory (collapsible) ---
            for traj_name in sorted(step["trajectories"].keys(),
                                     key=lambda t: int(t.split('_')[1])):
                files = step["trajectories"][traj_name]
                if not files:
                    continue
                traj_num = traj_name.replace("traj_", "")

                # Group by category
                by_cat = {}
                for fname in files:
                    key = fname.replace('.mp4', '')
                    cat = PLOT_META.get(key, ('Other', key, False))[0]
                    by_cat.setdefault(cat, []).append((fname, key))

                cards = []
                for cat in ['Q-Function', 'Actor / Residual', 'Landscape (UMAP)', 'Other']:
                    for fname, key in by_cat.get(cat, []):
                        url = f'/video?root={urllib.parse.quote(disc["root"])}&path={step_name}/{traj_name}/{fname}'
                        cards.append(_video_card(url, key))

                sections.append(f'''
                <details class="section">
                  <summary>
                    <h2 style="display:inline">{html.escape(exp_label)} &mdash; Step {step_num} &mdash; Trajectory {traj_num}</h2>
                  </summary>
                  <div class="grid">{"".join(cards)}</div>
                </details>''')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSRL-Pi0 Diagnostics Dashboard</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d37;
    --text: #e0e0e0;
    --accent: #6c8cff;
    --muted: #888;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 1.5rem;
  }}
  h1 {{
    text-align: center;
    margin-bottom: 0.5rem;
    font-size: 1.6rem;
    color: var(--accent);
  }}
  .subtitle {{
    text-align: center;
    color: var(--muted);
    margin-bottom: 2rem;
    font-size: 0.9rem;
  }}
  .section {{
    margin-bottom: 2rem;
  }}
  .section > h2, details > summary > h2 {{
    font-size: 1.15rem;
    margin-bottom: 0.75rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
  }}
  details > summary {{
    cursor: pointer;
    padding: 0.5rem 0;
    list-style: none;
  }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details > summary::before {{
    content: "\\25B6";
    display: inline-block;
    margin-right: 0.5rem;
    transition: transform 0.2s;
    color: var(--accent);
  }}
  details[open] > summary::before {{
    transform: rotate(90deg);
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 1rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .card-title {{
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
  }}
  .card video {{
    width: 100%;
    display: block;
  }}
  /* Controls */
  .controls {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }}
  .controls button {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.4rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
  }}
  .controls button:hover, .controls button.active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }}
</style>
</head>
<body>
<h1>DSRL-Pi0 Diagnostics Dashboard</h1>
<p class="subtitle">Animated diagnostic visualizations from evaluation runs</p>

<div class="controls">
  <button onclick="toggleAll(true)">Expand All Trajectories</button>
  <button onclick="toggleAll(false)">Collapse All Trajectories</button>
  <button onclick="playAll()">Play All Videos</button>
  <button onclick="pauseAll()">Pause All Videos</button>
</div>

{"".join(sections)}

<script>
function toggleAll(open) {{
  document.querySelectorAll('details').forEach(d => d.open = open);
}}
function playAll() {{
  document.querySelectorAll('video').forEach(v => v.play());
}}
function pauseAll() {{
  document.querySelectorAll('video').forEach(v => v.pause());
}}
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class DiagnosticsHandler(http.server.BaseHTTPRequestHandler):
    """Serves the dashboard HTML and streams .mp4 files."""

    dashboard_html = ""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/' or parsed.path == '':
            self._serve_html()
        elif parsed.path == '/video':
            self._serve_video(parsed.query)
        else:
            self.send_error(404)

    def _serve_html(self):
        data = self.dashboard_html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, query_string):
        params = urllib.parse.parse_qs(query_string)
        root = params.get('root', [''])[0]
        rel_path = params.get('path', [''])[0]

        # Security: ensure path doesn't escape the root
        full_path = os.path.normpath(os.path.join(root, rel_path))
        if not full_path.startswith(os.path.normpath(root)):
            self.send_error(403, 'Path traversal blocked')
            return
        if not os.path.isfile(full_path):
            self.send_error(404, f'File not found: {rel_path}')
            return

        file_size = os.path.getsize(full_path)

        # Support Range requests for video seeking
        range_header = self.headers.get('Range')
        if range_header:
            m = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                length = end - start + 1
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Content-Length', str(length))
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(full_path, 'rb') as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        self.send_response(200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Content-Length', str(file_size))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        with open(full_path, 'rb') as f:
            while True:
                chunk = f.read(1 << 16)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def log_message(self, format, *args):
        """Suppress noisy per-request logs; keep errors."""
        if args and '200' not in str(args[0]) and '206' not in str(args[0]):
            super().log_message(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Serve DSRL-Pi0 diagnostics dashboard.')
    parser.add_argument(
        '--sh_path', type=str, default=None,
        help='Path to a training .sh script to auto-discover diagnostics.')
    parser.add_argument(
        '--diagnostics_dir', type=str, default=None,
        help='Direct path to a diagnostics/ directory (overrides --sh_path).')
    parser.add_argument(
        '--step', type=int, default=None,
        help='Only show diagnostics for this training step (e.g. --step 5000).')
    parser.add_argument(
        '--port', type=int, default=8501,
        help='Port to serve the dashboard on (default: 8501).')
    args = parser.parse_args()

    # --- Resolve diagnostics directories ---
    experiments = []  # list of (label, discovery_dict)

    if args.diagnostics_dir:
        diag_dir = os.path.abspath(args.diagnostics_dir)
        disc = discover_diagnostics(diag_dir, step_filter=args.step)
        label = os.path.basename(os.path.dirname(diag_dir))
        experiments.append((label, disc))
        print(f'[Dashboard] Using direct diagnostics dir: {diag_dir}')

    elif args.sh_path:
        sh_path = os.path.abspath(args.sh_path)
        if not os.path.isfile(sh_path):
            print(f'Error: .sh file not found: {sh_path}')
            return

        exp_base = parse_sh_for_exp(sh_path)
        prefix = parse_sh_for_prefix(sh_path)
        print(f'[Dashboard] Parsed from {os.path.basename(sh_path)}:')
        print(f'            EXP    = {exp_base}')
        print(f'            prefix = {prefix}')

        if not exp_base or not os.path.isdir(exp_base):
            print(f'Error: EXP directory not found: {exp_base}')
            print('Tip: use --diagnostics_dir to point directly at a diagnostics/ folder.')
            return

        diag_dirs = find_diagnostics_dirs(exp_base, prefix)
        if not diag_dirs:
            print(f'Error: No diagnostics/ directories found under {exp_base} '
                  f'matching prefix "{prefix}".')
            print('Tip: use --diagnostics_dir to point directly at a diagnostics/ folder.')
            return

        for d in diag_dirs:
            disc = discover_diagnostics(d, step_filter=args.step)
            label = os.path.basename(os.path.dirname(d))
            experiments.append((label, disc))
            print(f'  Found: {d}')
    else:
        print('Error: provide either --sh_path or --diagnostics_dir.')
        return

    total_videos = sum(
        len(s["aggregate"]) + sum(len(v) for v in s["trajectories"].values())
        for _, disc in experiments for s in disc["steps"]
    )
    print(f'[Dashboard] Discovered {total_videos} diagnostic videos '
          f'across {sum(len(d["steps"]) for _, d in experiments)} step(s).')

    # --- Build and serve ---
    dashboard = build_dashboard_html(experiments)
    DiagnosticsHandler.dashboard_html = dashboard

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), DiagnosticsHandler) as httpd:
        print(f'\n[Dashboard] Serving at http://0.0.0.0:{args.port}')
        print(f'[Dashboard] Open http://localhost:{args.port} in your browser.')
        print(f'[Dashboard] Press Ctrl+C to stop.\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n[Dashboard] Shutting down.')


if __name__ == '__main__':
    main()
