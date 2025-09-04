#!/usr/bin/env python3
"""
Simple script to run the Animal Classify app
"""
import subprocess
import sys
import os

def main():
    print("🐄 Starting Cow & Buffalo AI Recognition App...")
    print("=" * 50)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Path to the virtual environment
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found!")
        print("Please run: py -3.11 -m venv .venv")
        return
    
    # Run the streamlit app
    try:
        print("🚀 Launching Streamlit app...")
        print("📱 App will open at: http://localhost:8503")
        print("⏹️  Press Ctrl+C to stop the app")
        print("=" * 50)
        
        subprocess.run([
            venv_python, "-m", "streamlit", "run", "app.py",
            "--server.port=8503",
            "--server.address=0.0.0.0",
            "--server.headless=false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()


