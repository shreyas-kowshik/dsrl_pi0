#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve a rollout video dashboard for DSRL-Pi0 base-policy evaluations.

Parses a training / evaluation .sh script to locate the experiment output
directory, discovers all rollout .mp4 files and the companion summary.csv,
and serves an interactive HTML dashboard on a local port.

Usage:
    python -m examples.visualize_rollouts \
        --sh_path examples/scripts/evaluate/evaluate_libero_pro_base_vision_pre_trained.sh \
        --port 8502

    # Or point directly at an output directory:
    python -m examples.visualize_rollouts \
        --output_dir /data/user_data/skowshik/libero-pro-base-eval/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_init_vision_full_data_trained/ \
        --port 8502
"""

import argparse
import csv
import html
import http.server
import os
import re
import socketserver
import urllib.parse


# ---------------------------------------------------------------------------
# .sh file parsing: extract OUTPUT_DIR
# ---------------------------------------------------------------------------

def _collect_shell_vars(content: str) -> dict:
    """Collect simple variable assignments (VAR=value) from shell script."""
    variables = {}
    for line in content.splitlines():
        line = line.strip()
        m = re.match(r'(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)=([^\s].*)?$', line)
        if m and not m.group(2):
            continue
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip('"').strip("'")
            if val.startswith('${' + key + ':-'):
                val = val[len('${' + key + ':-'):-1]
            variables[key] = val
    return variables


def _resolve_shell_vars(value: str, variables: dict) -> str:
    """Substitute $VAR and ${VAR} references using collected variables."""
    def _replace(m):
        var_name = m.group(1) or m.group(2)
        return variables.get(var_name, m.group(0))
    return re.sub(
        r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)',
        _replace, value,
    )


def parse_sh_for_output_dir(sh_path: str) -> str:
    """Extract the OUTPUT_DIR variable value from an evaluation .sh script."""
    with open(sh_path) as f:
        content = f.read()

    variables = _collect_shell_vars(content)

    # Try OUTPUT_DIR first, then fall back to EXP
    for var_name in ('OUTPUT_DIR', 'EXP'):
        m = re.search(
            rf'(?:export\s+)?{var_name}=\$\{{{var_name}:-([^}}]+)\}}', content)
        if m:
            raw = m.group(1).strip()
            return _resolve_shell_vars(raw, variables)
        m = re.search(rf'(?:export\s+)?{var_name}=([^\s]+)', content)
        if m:
            raw = m.group(1).strip().strip('"').strip("'")
            return _resolve_shell_vars(raw, variables)
    return ""


# ---------------------------------------------------------------------------
# Rollout discovery
# ---------------------------------------------------------------------------

def load_summary(output_dir: str) -> dict:
    """Load summary.csv into a dict keyed by rollout_id (int).

    Returns:
        {rollout_id: {"is_success": bool, "episode_return": float, "episode_len": int}}
    """
    csv_path = os.path.join(output_dir, 'summary.csv')
    summary = {}
    if not os.path.isfile(csv_path):
        return summary
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rid = int(row['rollout_id'])
            except (ValueError, TypeError):
                continue  # skip AGGREGATE or malformed rows
            summary[rid] = {
                'is_success': row.get('is_success', '0') == '1',
                'episode_return': float(row.get('episode_return', 0)),
                'episode_len': int(row.get('episode_len', 0)),
            }
    return summary


def discover_rollouts(output_dir: str) -> dict:
    """Walk an output directory and return structured metadata.

    Returns:
        {
          "root": "/abs/path/to/output_dir",
          "label": "experiment_name",
          "videos": [
            {"filename": "rollout_0000.mp4", "id": 0,
             "is_success": False, "episode_return": 0.0, "episode_len": 500},
            ...
          ],
          "success_rate": 0.0,
          "total": 50
        }
    """
    video_dir = os.path.join(output_dir, 'videos')

    # If output_dir itself contains .mp4 files (user pointed directly at
    # a videos/ folder), treat it as the video directory and look for
    # summary.csv in the parent.
    if not os.path.isdir(video_dir):
        has_videos = any(
            f.endswith(('.mp4', '.webm', '.avi'))
            for f in os.listdir(output_dir)
        )
        if has_videos:
            video_dir = output_dir
            summary = load_summary(os.path.dirname(output_dir))
        else:
            summary = load_summary(output_dir)
    else:
        summary = load_summary(output_dir)

    videos = []
    if os.path.isdir(video_dir):
        for fname in sorted(os.listdir(video_dir)):
            if not fname.endswith(('.mp4', '.webm', '.avi')):
                continue
            m = re.match(r'rollout_(\d+)', fname)
            rid = int(m.group(1)) if m else len(videos)
            info = summary.get(rid, {})
            videos.append({
                'filename': fname,
                'id': rid,
                'is_success': info.get('is_success', None),
                'episode_return': info.get('episode_return', None),
                'episode_len': info.get('episode_len', None),
            })

    # Resolve the root so /video routes point at the right place.
    # root should be the *parent* of the video files directory.
    root = os.path.abspath(output_dir)
    if video_dir == output_dir:
        # User pointed at videos/ directly — root is its parent so
        # the URL path "videos/<file>" still resolves correctly.
        root = os.path.dirname(root)

    n_success = sum(1 for v in videos if v['is_success'])
    total = len(videos)
    return {
        'root': root,
        'label': os.path.basename(os.path.normpath(output_dir)),
        'videos': videos,
        'success_rate': n_success / total if total else 0.0,
        'total': total,
    }


# ---------------------------------------------------------------------------
# HTML dashboard generation
# ---------------------------------------------------------------------------

def _success_badge(is_success):
    if is_success is None:
        return ''
    if is_success:
        return '<span class="badge success">SUCCESS</span>'
    return '<span class="badge failure">FAIL</span>'


def _video_card(video_url: str, video_info: dict) -> str:
    """Generate HTML for a single rollout video card."""
    badge = _success_badge(video_info['is_success'])
    meta_parts = []
    if video_info['episode_return'] is not None:
        meta_parts.append(f'R={video_info["episode_return"]:.1f}')
    if video_info['episode_len'] is not None:
        meta_parts.append(f'L={video_info["episode_len"]}')
    meta_str = ' &middot; '.join(meta_parts)
    meta_line = f'<div class="card-meta">{meta_str}</div>' if meta_str else ''

    return f'''
    <div class="card {'card-success' if video_info['is_success'] else 'card-failure' if video_info['is_success'] is not None else ''}">
      <div class="card-header">
        <span class="card-title">Rollout {video_info["id"]:04d}</span>
        {badge}
      </div>
      {meta_line}
      <video controls loop muted preload="metadata">
        <source src="{html.escape(video_url)}" type="video/mp4">
      </video>
    </div>'''


def build_dashboard_html(experiments: list) -> str:
    """Build the full HTML dashboard page.

    Args:
        experiments: list of discovery dicts from discover_rollouts().
    """
    sections = []
    total_videos = 0

    for exp in experiments:
        total_videos += exp['total']
        n_success = sum(1 for v in exp['videos'] if v['is_success'])
        rate_pct = exp['success_rate'] * 100

        cards = []
        for v in exp['videos']:
            url = (f'/video?root={urllib.parse.quote(exp["root"])}'
                   f'&path=videos/{v["filename"]}')
            cards.append(_video_card(url, v))

        sections.append(f'''
        <div class="section">
          <div class="section-header">
            <h2>{html.escape(exp["label"])}</h2>
            <div class="stats">
              <span class="stat">{exp["total"]} rollouts</span>
              <span class="stat">{n_success}/{exp["total"]} success ({rate_pct:.0f}%)</span>
            </div>
          </div>
          <div class="grid" id="grid-{html.escape(exp['label'])}">
            {"".join(cards)}
          </div>
        </div>''')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rollout Video Dashboard</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d37;
    --text: #e0e0e0;
    --accent: #6c8cff;
    --muted: #888;
    --success: #34d399;
    --failure: #f87171;
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
    margin-bottom: 2.5rem;
  }}
  .section-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.5rem;
  }}
  .section-header h2 {{
    font-size: 1.1rem;
    color: var(--text);
    word-break: break-all;
  }}
  .stats {{
    display: flex;
    gap: 1rem;
  }}
  .stat {{
    font-size: 0.85rem;
    color: var(--muted);
  }}
  /* Controls */
  .controls {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }}
  .controls button, .controls select, .controls input {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.4rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
  }}
  .controls input {{
    width: 220px;
  }}
  .controls button:hover {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }}
  /* Grid */
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
  .card-success {{ border-left: 3px solid var(--success); }}
  .card-failure {{ border-left: 3px solid var(--failure); }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  .card-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
  }}
  .card-meta {{
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .badge {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
  }}
  .badge.success {{ background: var(--success); color: #000; }}
  .badge.failure {{ background: var(--failure); color: #000; }}
  .card video {{
    width: 100%;
    display: block;
  }}
  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<h1>Rollout Video Dashboard</h1>
<p class="subtitle">{total_videos} rollout videos from evaluation runs</p>

<div class="controls">
  <input type="text" id="search" placeholder="Filter by rollout ID..." oninput="filterVideos()">
  <select id="filter-status" onchange="filterVideos()">
    <option value="all">All</option>
    <option value="success">Success Only</option>
    <option value="failure">Failure Only</option>
  </select>
  <select id="cols" onchange="changeCols()">
    <option value="repeat(auto-fill, minmax(380px, 1fr))">Auto</option>
    <option value="1fr">1 Column</option>
    <option value="1fr 1fr">2 Columns</option>
    <option value="1fr 1fr 1fr">3 Columns</option>
    <option value="1fr 1fr 1fr 1fr">4 Columns</option>
  </select>
  <button onclick="playAll()">Play All</button>
  <button onclick="pauseAll()">Pause All</button>
</div>

{"".join(sections)}

<script>
function filterVideos() {{
  const q = document.getElementById('search').value.toLowerCase();
  const status = document.getElementById('filter-status').value;
  document.querySelectorAll('.card').forEach(c => {{
    const title = c.querySelector('.card-title').textContent.toLowerCase();
    const matchText = !q || title.includes(q);
    let matchStatus = true;
    if (status === 'success') matchStatus = c.classList.contains('card-success');
    else if (status === 'failure') matchStatus = c.classList.contains('card-failure');
    c.classList.toggle('hidden', !(matchText && matchStatus));
  }});
}}
function changeCols() {{
  const v = document.getElementById('cols').value;
  document.querySelectorAll('.grid').forEach(g => g.style.gridTemplateColumns = v);
}}
function playAll() {{
  document.querySelectorAll('.card:not(.hidden) video').forEach(v => v.play());
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

class RolloutHandler(http.server.BaseHTTPRequestHandler):
    """Serves the dashboard HTML and streams .mp4 files."""

    dashboard_html = ""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ('/', ''):
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
                self.send_header('Content-Range',
                                 f'bytes {start}-{end}/{file_size}')
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
        description='Serve rollout video dashboard for base-policy evaluations.')
    parser.add_argument(
        '--sh_path', type=str, default=None,
        help='Path to an evaluation .sh script to auto-discover the output dir.')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Direct path to an output directory containing videos/ and '
             'summary.csv (overrides --sh_path).')
    parser.add_argument(
        '--port', type=int, default=8502,
        help='Port to serve the dashboard on (default: 8502).')
    args = parser.parse_args()

    # --- Resolve output directories ---
    experiments = []

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        if not os.path.isdir(out_dir):
            print(f'Error: directory not found: {out_dir}')
            return
        disc = discover_rollouts(out_dir)
        experiments.append(disc)
        print(f'[Dashboard] Using output dir: {out_dir}')

    elif args.sh_path:
        sh_path = os.path.abspath(args.sh_path)
        if not os.path.isfile(sh_path):
            print(f'Error: .sh file not found: {sh_path}')
            return

        out_dir = parse_sh_for_output_dir(sh_path)
        print(f'[Dashboard] Parsed from {os.path.basename(sh_path)}:')
        print(f'            OUTPUT_DIR = {out_dir}')

        if not out_dir or not os.path.isdir(out_dir):
            print(f'Error: output directory not found: {out_dir}')
            print('Tip: use --output_dir to point directly at the directory.')
            return

        disc = discover_rollouts(out_dir)
        experiments.append(disc)
    else:
        print('Error: provide either --sh_path or --output_dir.')
        return

    total_videos = sum(e['total'] for e in experiments)
    total_success = sum(
        sum(1 for v in e['videos'] if v['is_success']) for e in experiments
    )
    print(f'[Dashboard] Discovered {total_videos} rollout videos, '
          f'{total_success}/{total_videos} successful.')

    # --- Build and serve ---
    dashboard = build_dashboard_html(experiments)
    RolloutHandler.dashboard_html = dashboard

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), RolloutHandler) as httpd:
        print(f'\n[Dashboard] Serving at http://0.0.0.0:{args.port}')
        print(f'[Dashboard] Open http://localhost:{args.port} in your browser.')
        print(f'[Dashboard] Press Ctrl+C to stop.\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n[Dashboard] Shutting down.')


if __name__ == '__main__':
    main()
