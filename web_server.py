#!/usr/bin/env python3
"""
Simple web server to serve the frontend interface for the Retail Demand Forecasting System.
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from pathlib import Path

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Get the project directory
    project_dir = Path(__file__).parent
    web_dir = project_dir / "web"
    
    # Change to project directory
    os.chdir(project_dir)
    
    # Check if web directory exists
    if not web_dir.exists():
        print(f"Error: Web directory not found at {web_dir}")
        sys.exit(1)
    
    # Server configuration
    host = 'localhost'
    port = 3000
    
    # Create server
    server_address = (host, port)
    httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
    
    print(f"🌐 Starting web server...")
    print(f"📍 Server running at: http://{host}:{port}")
    print(f"📂 Serving files from: {web_dir}")
    print(f"🔗 Frontend URL: http://{host}:{port}/index.html")
    print(f"⚠️  Make sure the FastAPI server is running on http://localhost:8000")
    print(f"🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Open browser automatically
    try:
        webbrowser.open(f'http://{host}:{port}/index.html')
        print("🚀 Browser opened automatically")
    except:
        print("⚠️  Could not open browser automatically")
        print(f"   Please open http://{host}:{port}/index.html manually")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.server_close()

if __name__ == "__main__":
    main()
