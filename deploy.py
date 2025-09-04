#!/usr/bin/env python3
"""
Deployment script for Animal Classify App
Supports multiple free hosting platforms
"""
import os
import subprocess
import sys

def deploy_to_streamlit_cloud():
    """Deploy to Streamlit Cloud (Free)"""
    print("🚀 Deploying to Streamlit Cloud...")
    print("=" * 50)
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("📦 Initializing Git repository...")
        subprocess.run(['git', 'init'])
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', 'Initial commit: Animal Classify App'])
    
    print("✅ Ready for Streamlit Cloud deployment!")
    print("\n📋 Next steps:")
    print("1. Create a GitHub repository")
    print("2. Push your code: git remote add origin <your-repo-url>")
    print("3. Push: git push -u origin main")
    print("4. Go to https://share.streamlit.io")
    print("5. Connect your GitHub repo and deploy!")

def deploy_to_huggingface():
    """Deploy to Hugging Face Spaces (Free)"""
    print("🤗 Deploying to Hugging Face Spaces...")
    print("=" * 50)
    
    # Create app.py for Hugging Face
    hf_app_content = '''import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import your app
from app import main

if __name__ == "__main__":
    main()
'''
    
    with open('app_hf.py', 'w') as f:
        f.write(hf_app_content)
    
    print("✅ Created Hugging Face compatible app!")
    print("\n📋 Next steps:")
    print("1. Go to https://huggingface.co/new-space")
    print("2. Create a new Space with Streamlit SDK")
    print("3. Upload your files")
    print("4. Your app will be live at: https://huggingface.co/spaces/yourusername/your-space-name")

def main():
    print("🐄 Animal Classify App - Deployment Helper")
    print("=" * 50)
    
    print("Choose deployment option:")
    print("1. Streamlit Cloud (Recommended - Easiest)")
    print("2. Hugging Face Spaces (Free GPU)")
    print("3. Show all options")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        deploy_to_streamlit_cloud()
    elif choice == "2":
        deploy_to_huggingface()
    elif choice == "3":
        show_all_options()
    else:
        print("❌ Invalid choice!")

def show_all_options():
    print("\n🌐 All Free Hosting Options:")
    print("=" * 50)
    
    options = [
        ("Streamlit Cloud", "https://share.streamlit.io", "✅ Easiest, built for Streamlit"),
        ("Hugging Face Spaces", "https://huggingface.co/spaces", "✅ Free GPU, great for AI apps"),
        ("Railway", "https://railway.app", "✅ Custom domains, easy deployment"),
        ("Render", "https://render.com", "✅ Free tier, good performance"),
        ("Heroku", "https://heroku.com", "⚠️ Limited free tier"),
        ("Replit", "https://replit.com", "✅ Online IDE + hosting"),
        ("PythonAnywhere", "https://pythonanywhere.com", "✅ Free tier available")
    ]
    
    for i, (name, url, description) in enumerate(options, 1):
        print(f"{i}. {name}")
        print(f"   URL: {url}")
        print(f"   Note: {description}")
        print()

if __name__ == "__main__":
    main()


