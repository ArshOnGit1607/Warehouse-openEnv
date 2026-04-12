import sys
import os
import uvicorn

# Add root directory to sys.path to allow importing from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
