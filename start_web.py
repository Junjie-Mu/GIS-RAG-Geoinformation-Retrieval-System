"""
Script to launch the GIS-RAG web interface.
"""
import sys
import os
import subprocess
import time
import socket
import webbrowser
from config.settings import settings

if __name__ == "__main__":
    show_host = settings.WEB_HOST
    if show_host in {"0.0.0.0", "::"}:
        show_host = "127.0.0.1"

    print(f"🌐 启动GIS-RAG Web界面...")
    print(f"🔗 地址: http://{show_host}:{settings.WEB_PORT}")
    
    # Start using streamlit run command
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/web/app.py",
        "--server.address", settings.WEB_HOST,
        "--server.port", str(settings.WEB_PORT),
        "--theme.base", "light",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
    ]
    
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_FILEWATCHERTYPE", "none")
    proc = subprocess.Popen(cmd, env=env)

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection((show_host, int(settings.WEB_PORT)), timeout=0.5):
                break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.2)

    try:
        webbrowser.open(f"http://{show_host}:{settings.WEB_PORT}", new=2)
    except Exception:
        pass

    proc.wait()




