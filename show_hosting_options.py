#!/usr/bin/env python3
"""
Show all free hosting options for Animal Classify App
"""
def show_all_options():
    print("🌐 FREE HOSTING OPTIONS FOR YOUR ANIMAL CLASSIFY APP")
    print("=" * 60)
    
    options = [
        ("1. Streamlit Cloud", "https://share.streamlit.io", "✅ EASIEST - Built for Streamlit apps", "Free forever, one-click deploy"),
        ("2. Hugging Face Spaces", "https://huggingface.co/spaces", "✅ FREE GPU - Perfect for AI apps", "Free GPU, great for ML models"),
        ("3. Railway", "https://railway.app", "✅ CUSTOM DOMAINS - Easy deployment", "Free tier with custom domains"),
        ("4. Render", "https://render.com", "✅ GOOD PERFORMANCE - Reliable hosting", "Free tier, good uptime"),
        ("5. Replit", "https://replit.com", "✅ ONLINE IDE - Code and host together", "Free hosting + online editor"),
        ("6. PythonAnywhere", "https://pythonanywhere.com", "✅ PYTHON FOCUSED - Made for Python", "Free tier available"),
        ("7. Heroku", "https://heroku.com", "⚠️ LIMITED FREE - Popular but limited", "Free tier with restrictions")
    ]
    
    for name, url, status, description in options:
        print(f"\n{name}")
        print(f"   🌐 URL: {url}")
        print(f"   📊 Status: {status}")
        print(f"   📝 Description: {description}")
    
    print("\n" + "=" * 60)
    print("🚀 RECOMMENDED: Streamlit Cloud (Easiest)")
    print("🤖 BEST FOR AI: Hugging Face Spaces (Free GPU)")
    print("🌐 MOST FEATURES: Railway (Custom domains)")
    print("=" * 60)

if __name__ == "__main__":
    show_all_options()

