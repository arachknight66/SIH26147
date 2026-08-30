from __future__ import annotations
import sys
from app.ui.web_app import launch_web_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    launch_web_server(port=port, open_browser=True)
