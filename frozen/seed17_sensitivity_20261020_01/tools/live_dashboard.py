"""Read-only localhost dashboard for existing experiment logs and figures.

No training, evaluation, plotting, shell execution or file mutation happens in
the HTTP server. The trainer refreshes PNGs; the browser polls every 5 seconds.
"""
from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit

if __name__ == "__main__" and __package__ in (None, ""):
    # Direct file execution puts tools/, rather than the project, on sys.path.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priority_mec.reporting import read_csv_rows


FILE_SUFFIXES = {".png", ".svg", ".pdf", ".json", ".csv", ".jsonl", ".txt"}
CONTENT_TYPES = {".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf",
                 ".json": "application/json; charset=utf-8", ".csv": "text/csv; charset=utf-8",
                 ".jsonl": "application/x-ndjson; charset=utf-8", ".txt": "text/plain; charset=utf-8"}


def safe_path(root: Path, request_path: str):
    """Resolve a URL to a regular file under root, including symlink checks."""
    parsed = urlsplit(request_path)
    decoded = unquote(parsed.path)
    if any(char in decoded for char in ("\\", "\x00", ":")):
        raise ValueError("Invalid artifact path")
    relative = decoded.lstrip("/")
    if any(part == ".." for part in relative.split("/")):
        raise ValueError("Path traversal rejected")
    root = Path(root).resolve()
    candidate = (root/relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("Artifact lies outside dashboard root")
    if not candidate.is_file() or candidate.suffix.lower() not in FILE_SUFFIXES:
        raise FileNotFoundError(relative)
    return candidate


def _safe_value(value):
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def metrics_snapshot(root: Path):
    """Fresh JSON-safe state, without trusting arbitrary scripts or HTML files."""
    root = Path(root).resolve()
    directories = [root]
    manifest = _read_json(root/"experiment_manifest.json")
    groups = manifest.get("groups", []) if isinstance(manifest, dict) else []
    labels = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        identifier = group.get("id", group.get("name", ""))
        path = (root/str(group.get("run_dir", group.get("path", identifier)))).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_dir() and path not in directories:
            directories.append(path)
            labels[path] = group.get("label", identifier)
    runs = []
    for directory in directories:
        episodes = read_csv_rows(directory/"episodes.csv")
        validation = read_csv_rows(directory/"validation.csv")
        status = _read_json(directory/"status.json")
        summary = _read_json(directory/"summary.json")
        if isinstance(summary, dict):
            summary = summary.get("summary", summary)
        evaluations = {}
        for name in ("evaluation_best", "evaluation_last", "evaluation"):
            result = _read_json(directory/name/"summary.json")
            if result:
                evaluations[name] = result.get("summary", result)
        relative = directory.relative_to(root).as_posix()
        if directory == root and groups and not any((episodes, validation, status, summary)):
            continue
        runs.append({"run": relative, "label": labels.get(directory, directory.name),
                     "completed_episodes": len(episodes), "latest_episode": episodes[-1] if episodes else None,
                     "validation_points": len(validation), "latest_validation": validation[-1] if validation else None,
                     "status": status, "summary": summary, "evaluations": evaluations})
    images = []
    for directory in directories:
        for path in sorted((directory/"plots").glob("*.png")):
            # Reuse the serving guard; do not advertise external symlink targets.
            try:
                safe_path(root, "/"+path.relative_to(root).as_posix())
            except (ValueError, FileNotFoundError, OSError):
                continue
            images.append({"path": path.relative_to(root).as_posix(), "mtime_ns": path.stat().st_mtime_ns})
    for path in sorted((root/"comparison").glob("*/*.png")):
        try:
            safe_path(root, "/"+path.relative_to(root).as_posix())
        except (ValueError, FileNotFoundError, OSError):
            continue
        images.append({"path": path.relative_to(root).as_posix(), "mtime_ns": path.stat().st_mtime_ns})
    return _safe_value({"root_name": root.name, "refresh_seconds": 5, "runs": runs, "images": images})


INDEX_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAV MEC live experiment</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1b2935;font:15px system-ui,sans-serif}
header{padding:20px 26px;background:#173247;color:white}h1{margin:0 0 5px;font-size:24px}
main{padding:18px 24px}#state{font-size:13px;opacity:.85}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:15px}
article,figure{margin:0;background:white;border:1px solid #dce3e9;border-radius:7px;padding:15px}
h2{font-size:18px;margin:0 0 10px}.metric{display:flex;justify-content:space-between;gap:12px;padding:4px 0}
.metric span:first-child{color:#546574}pre{white-space:pre-wrap;word-break:break-word;font-size:11px;color:#506270}
#figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,650px),1fr));gap:18px;margin-top:20px}
img{display:block;width:100%;height:auto}figcaption{font-size:13px;overflow-wrap:anywhere}a{color:#24638b}
.empty{padding:24px;color:#53697a}footer{padding:20px 24px;font-size:12px;color:#53697a}
</style></head><body><header><h1 id="title">UAV MEC live experiment</h1>
<div id="state">Connecting to existing logs...</div></header><main><div id="runs" class="cards"></div>
<div id="figures"></div></main><footer>Read-only localhost view · refresh every 5 seconds · training curves show completed episodes and a trailing 25-episode mean. Baselines have no training curves.</footer>
<script>
const runsNode=document.getElementById('runs'), figuresNode=document.getElementById('figures');
const images=new Map();
function number(v){if(v===null||v===undefined)return 'N/A';if(typeof v==='boolean')return String(v);if(Array.isArray(v))return v.map(number).join(', ');return typeof v==='number'?v.toLocaleString(undefined,{maximumFractionDigits:5}):String(v)}
function addMetric(parent,name,value){const row=document.createElement('div');row.className='metric';const key=document.createElement('span'),val=document.createElement('strong');key.textContent=name;val.textContent=number(value);row.append(key,val);parent.append(row)}
async function refresh(){try{const response=await fetch('/api/metrics?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);const data=await response.json();document.getElementById('title').textContent=data.root_name+' — live experiment';document.getElementById('state').textContent='Updated '+new Date().toLocaleTimeString()+' · polls every 5 seconds';runsNode.replaceChildren();
for(const run of data.runs){const card=document.createElement('article');const title=document.createElement('h2');title.textContent=run.label;card.append(title);const latest=run.latest_episode||{};addMetric(card,'Completed episodes',run.completed_episodes);addMetric(card,'Episode return',latest.episode_return);addMetric(card,'J (weighted delay)',latest.weighted_delay_mean);addMetric(card,'Energy (J / UAV)',latest.per_uav_energy_j);addMetric(card,'Deadline completion',latest.deadline_completion);addMetric(card,'Validation points',run.validation_points);if(run.status&&Object.keys(run.status).length){const p=document.createElement('pre');p.textContent=JSON.stringify(run.status,null,2);card.append(p)}for(const [name,summary]of Object.entries(run.evaluations||{})){const label=document.createElement('p');label.textContent=name+': J='+number(summary.weighted_delay_mean)+'; completion='+number(summary.deadline_completion);card.append(label)}runsNode.append(card)}
if(!data.runs.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='Waiting for logs. This dashboard only reads existing files.';runsNode.append(empty)}
const current=new Set();for(const info of data.images){current.add(info.path);let item=images.get(info.path);if(!item){const figure=document.createElement('figure'),link=document.createElement('a'),img=document.createElement('img'),caption=document.createElement('figcaption');link.target='_blank';link.rel='noopener';link.append(img);caption.textContent=info.path;figure.append(link,caption);figuresNode.append(figure);item={figure,link,img,version:null};images.set(info.path,item)}if(item.version!==info.mtime_ns){const url='/'+info.path.split('/').map(encodeURIComponent).join('/')+'?v='+info.mtime_ns;item.img.src=url;item.img.alt=info.path;item.link.href=url;item.version=info.mtime_ns}}for(const [path,item]of images){if(!current.has(path)){item.figure.remove();images.delete(path)}}
}catch(error){document.getElementById('state').textContent='Waiting for logs/server: '+error.message}}
refresh();setInterval(refresh,5000);
</script></body></html>'''


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = Path(root).resolve()
        super().__init__(*args, **kwargs)

    def log_message(self, _format, *_args):
        pass

    def _send(self, content, content_type, head=False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; object-src 'none'; base-uri 'none'"
                         if content_type.startswith("text/html") else "default-src 'none'; style-src 'unsafe-inline'; sandbox")
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _get(self, head=False):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8", head)
        if path == "/api/metrics":
            body = json.dumps(metrics_snapshot(self.root), ensure_ascii=False, allow_nan=False).encode("utf-8")
            return self._send(body, "application/json; charset=utf-8", head)
        try:
            artifact = safe_path(self.root, self.path)
            content = artifact.read_bytes()
        except ValueError:
            return self.send_error(403, "Artifact path rejected")
        except (OSError, FileNotFoundError):
            return self.send_error(404, "Artifact not found")
        return self._send(content, CONTENT_TYPES[artifact.suffix.lower()], head)

    def do_GET(self):
        self._get()

    def do_HEAD(self):
        self._get(head=True)


def serve(root: Path, port=8765):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not 0 <= int(port) <= 65535:
        raise ValueError("port must be in [0,65535]")
    handler = partial(DashboardHandler, root=root)
    with ThreadingHTTPServer(("127.0.0.1", int(port)), handler) as server:
        print("Read-only dashboard: http://127.0.0.1:%d/ (Ctrl+C to stop)" % server.server_port, flush=True)
        try:
            server.serve_forever(poll_interval=.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(Path(args.run), args.port)
