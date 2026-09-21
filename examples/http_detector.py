"""Tiny HTTP wrapper around any scoring function, to try `gauntlet bench --url`.

    python examples/http_detector.py &            # serves POST /score on :9000
    python -m gauntlet bench --url http://127.0.0.1:9000/score --name my-service
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from keyword_detector import score


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
        body = json.dumps({"score": score([text])[0]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 9000), H).serve_forever()
