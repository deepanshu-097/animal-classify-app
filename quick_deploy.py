#!/usr/bin/env python3
"""
Quick deployment script for Animal Classify App
"""
import os
import subprocess
import webbrowser

def quick_streamlit_deploy():
    """Quick deployment to Streamlit Cloud"""
    print("🚀 QUICK DEPLOYMENT TO STREAMLIT CLOUD")
    print("=" * 50)
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("📦 Initializing Git repository...")
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit: Animal Classify App'], check=True)
        print("✅ Git repository initialized!")
    
    print("\n📋 NEXT STEPS:")
    print("1. 🌐 Go to: https://github.com/new")
    print("2. 📝 Create repository named: animal-classify-app")
    print("3. 🔗 Copy the repository URL")
    print("4. ⬆️  Run these commands:")
    print("   git remote add origin <YOUR_REPO_URL>")
    print("   git push -u origin main")
    print("5. 🚀 Go to: https://share.streamlit.io")
    print("6. 🎯 Deploy your app!")
    
    # Open GitHub in browser
    print("\n🌐 Opening GitHub for you...")
    webbrowser.open("https://github.com/new")
    
    print("\n⏰ Total time: ~5 minutes")
    print("🎉 Your app will be live at: https://your-app-name.streamlit.app/")

def quick_huggingface_deploy():
    """Quick deployment to Hugging Face Spaces"""
    print("🤗 QUICK DEPLOYMENT TO HUGGING FACE SPACES")
    print("=" * 50)
    
    print("\n📋 NEXT STEPS:")
    print("1. 🌐 Go to: https://huggingface.co/new-space")
    print("2. 📝 Name: animal-classify")
    print("3. 🎯 SDK: Streamlit")
    print("4. 👁️  Visibility: Public")
    print("5. 📁 Upload all your files")
    print("6. 🚀 Auto-deploy!")
    
    # Open Hugging Face in browser
    print("\n🌐 Opening Hugging Face for you...")
    webbrowser.open("https://huggingface.co/new-space")
    
    print("\n⏰ Total time: ~3 minutes")
    print("🎉 Your app will be live at: https://huggingface.co/spaces/YOUR_USERNAME/animal-classify")

def main():
    print("🐄 ANIMAL CLASSIFY APP - QUICK DEPLOYMENT")
    print("=" * 50)
    print("Choose your deployment method:")
    print("1. 🚀 Streamlit Cloud (Recommended - Easiest)")
    print("2. 🤗 Hugging Face Spaces (Free GPU)")
    print("3. 📖 Show deployment guide")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        quick_streamlit_deploy()
    elif choice == "2":
        quick_huggingface_deploy()
    elif choice == "3":
        print("\n📖 Opening deployment guide...")
        os.system("start DEPLOYMENT_GUIDE.md")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()

