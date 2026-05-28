import http.server
import socketserver
import webbrowser
import threading
import time
import sys

PORT = 8000

class PresentationHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching to ensure editing code is immediately reflected in browser
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def open_browser():
    # Wait for server to start before launching browser
    time.sleep(1.2)
    url = f"http://localhost:{PORT}/index.html"
    print(f"[*] Opening browser to {url}...")
    webbrowser.open(url)

def run_server():
    # Force use of UTF-8 encoding for text files on Windows/mac
    # standard python HTTP server will load based on system default locale
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), PresentationHandler) as httpd:
            print(f"[+] Server started successfully on http://localhost:{PORT}")
            print("[*] Press Ctrl+C to stop the server.")
            httpd.serve_forever()
    except Exception as e:
        print(f"[-] Error starting server: {e}", file=sys.stderr)
        print("[-] Check if port 8000 is already in use by another application.", file=sys.stderr)

if __name__ == "__main__":
    # Configure MIME type support explicitly
    # Some platforms don't have .css or .js preconfigured
    http.server.SimpleHTTPRequestHandler.extensions_map.update({
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.html': 'text/html',
    })

    # Start browser-open thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start server in main thread
    run_server()
