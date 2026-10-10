"""
Lightweight Brain Encoding Explorer Web Server.

Serves:
- HTML UI from app/templates/index.html
- CSS & JS from app/static/
- Figure graphics from figures/ and results/
- Results API from /api/data
"""

from typing import Dict, Any
import http.server
import socketserver
import json
import os
import sys
import urllib.parse
import webbrowser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_DIR = os.path.join(PROJECT_ROOT, "app")
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")

PORT = 7860


def load_all_summaries() -> Dict[str, Any]:
    """Loads all existing results/summary_{subject}.json files on disk."""
    summaries = {}
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("summary_") and f.endswith(".json"):
                s_id = f.replace("summary_", "").replace(".json", "")
                try:
                    with open(os.path.join(results_dir, f), "r", encoding="utf-8") as fp:
                        summaries[s_id] = json.load(fp)
                except Exception:
                    pass
    return summaries


class BrainExplorerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Main Page
        if path in ["/", "/index.html"]:
            index_path = os.path.join(TEMPLATES_DIR, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # 2. Static Assets (CSS, JS)
        elif path.startswith("/static/"):
            rel_file = path.replace("/static/", "")
            file_path = os.path.join(STATIC_DIR, rel_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                if file_path.endswith(".css"):
                    self.send_header("Content-Type", "text/css; charset=utf-8")
                elif file_path.endswith(".js"):
                    self.send_header("Content-Type", "application/javascript; charset=utf-8")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # 3. JSON Data API
        elif path == "/api/data":
            summaries = load_all_summaries()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(summaries).encode("utf-8"))
            return

        # 4. Figures & Images
        elif path.startswith("/figures/") or path.startswith("/results/"):
            rel_file = path.lstrip("/")
            file_path = os.path.join(PROJECT_ROOT, rel_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                if file_path.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif file_path.endswith(".json"):
                    self.send_header("Content-Type", "application/json")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence console log noise for clean terminal output
        pass


def run_server(port: int = PORT):
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), BrainExplorerHandler) as httpd:
            url = f"http://localhost:{port}"
            print("\n" + "=" * 60)
            print(f"🧠 Brain Encoding Explorer running at: {url}")
            print(f"   Frontend HTML : app/templates/index.html")
            print(f"   Frontend CSS  : app/static/style.css")
            print(f"   Frontend JS   : app/static/app.js")
            print(f"   Press Ctrl+C to stop the server.")
            print("=" * 60 + "\n")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    run_server()
