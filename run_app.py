#!/usr/bin/env python
"""
Launcher script for the Section Analyzer Streamlit app
This script performs checks before launching the app
"""

import os
import sys
import subprocess
import importlib.util

def check_streamlit():
    """Check if Streamlit is installed"""
    spec = importlib.util.find_spec('streamlit')
    if spec is None:
        print("❌ Streamlit is not installed!")
        print("Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            print("Please install manually: pip install streamlit")
            return False
    return True

def check_critical_packages():
    """Check for critical packages"""
    critical = ['numpy', 'pandas', 'matplotlib', 'sectionproperties', 'shapely']
    missing = []
    
    for package in critical:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
    
    return missing

def main():
    print("=" * 60)
    print("Section Analyzer - App Launcher")
    print("=" * 60)
    print()
    
    # Check if Home.py exists
    if not os.path.exists('Home.py'):
        print("❌ Error: Home.py not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease ensure you're running this script from the app directory.")
        sys.exit(1)
    
    # Check Streamlit
    if not check_streamlit():
        sys.exit(1)
    
    # Check critical packages
    missing = check_critical_packages()
    if missing:
        print("⚠️ Warning: Missing critical packages:", ", ".join(missing))
        print("\nAttempting to install missing packages...")
        
        for package in missing:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"  ❌ Failed to install {package}")
        
        print()
        
        # Re-check
        missing = check_critical_packages()
        if missing:
            print("⚠️ Still missing packages:", ", ".join(missing))
            print("The app may not work correctly.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        print("✅ All critical packages are installed")
    
    # Create required directories
    for dir_name in ['modules', 'data']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created {dir_name}/ directory")
    
    print()
    print("Launching Streamlit app...")
    print("-" * 40)
    print("If the app doesn't open automatically, navigate to:")
    print("  http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the app")
    print("-" * 40)
    
    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Home.py"])
    except KeyboardInterrupt:
        print("\n\n✅ App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nTry running directly:")
        print("  streamlit run Home.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
