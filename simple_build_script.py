#!/usr/bin/env python3
"""
Simple Build Script for DeepFaceLive with Freeze-Frame
Quick and easy way to create executable
"""

import os
import sys
import subprocess
from pathlib import Path


def quick_build():
    """Quick build using auto-py-to-exe alternative"""
    print("🚀 DeepFaceLive Quick Build Script")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    main_py = current_dir / "main.py"
    
    if not main_py.exists():
        print("❌ main.py not found in current directory")
        print("Please run this script from your DeepFaceLive root directory")
        return False
    
    print(f"✓ Found main.py in {current_dir}")
    
    # Install PyInstaller if needed
    try:
        import PyInstaller
        print(f"✓ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyInstaller"])
        print("✓ PyInstaller installed")
    
    # Simple build command
    print("🔨 Building executable...")
    
    build_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",  # Single executable file
        "--noconsole",  # No console window
        "--name", "DeepFaceLive_FreezeFrame",
        "--add-data", "freeze_frame_modules;freeze_frame_modules" if os.name == 'nt' else "freeze_frame_modules:freeze_frame_modules",
        "--add-data", "apps;apps" if os.name == 'nt' else "apps:apps",
        "--add-data", "localization;localization" if os.name == 'nt' else "localization:localization",
        "--add-data", "resources;resources" if os.name == 'nt' else "resources:resources",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "torch",
        "--hidden-import", "freeze_frame_manager",
        "--hidden-import", "deepfacelive_integration",
        "main.py"
    ]
    
    # Add config files
    config_files = ["freeze_config.json", "detector_config.json", "freeze_integration_config.json"]
    for config in config_files:
        if (current_dir / config).exists():
            if os.name == 'nt':
                build_cmd.extend(["--add-data", f"{config};."])
            else:
                build_cmd.extend(["--add-data", f"{config}:."])
    
    print("Running:", " ".join(build_cmd))
    
    try:
        result = subprocess.run(build_cmd, check=True)
        
        print("\n🎉 Build completed successfully!")
        print(f"📁 Executable location: {current_dir}/dist/DeepFaceLive_FreezeFrame.exe" if os.name == 'nt' else f"{current_dir}/dist/DeepFaceLive_FreezeFrame")
        
        # Create launcher script
        create_launcher()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def create_launcher():
    """Create simple launcher script"""
    print("📋 Creating launcher script...")
    
    current_dir = Path.cwd()
    dist_dir = current_dir / "dist"
    
    if os.name == 'nt':  # Windows
        launcher_content = '''@echo off
echo Starting DeepFaceLive with Freeze-Frame...
cd /d "%~dp0"
DeepFaceLive_FreezeFrame.exe run DeepFaceLive
pause
'''
        launcher_path = dist_dir / "Start_DeepFaceLive.bat"
    else:  # Linux/Mac
        launcher_content = '''#!/bin/bash
echo "Starting DeepFaceLive with Freeze-Frame..."
cd "$(dirname "$0")"
./DeepFaceLive_FreezeFrame run DeepFaceLive
read -p "Press Enter to close..."
'''
        launcher_path = dist_dir / "start_deepfacelive.sh"
    
    try:
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        if os.name != 'nt':
            os.chmod(launcher_path, 0o755)
        
        print(f"✓ Launcher created: {launcher_path}")
    except Exception as e:
        print(f"⚠ Could not create launcher: {e}")


def install_requirements():
    """Install minimum requirements for building"""
    print("📦 Installing build requirements...")
    
    requirements = [
        "PyInstaller>=5.0",
        "opencv-python",
        "numpy",
        "Pillow",
        "PyQt5"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ {req}")
        except subprocess.CalledProcessError:
            print(f"⚠ Failed to install {req}")


def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if os.name != 'nt':
        return
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "DeepFaceLive FreezeFrame.lnk")
        target = str(Path.cwd() / "dist" / "DeepFaceLive_FreezeFrame.exe")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.Arguments = "run DeepFaceLive"
        shortcut.WorkingDirectory = str(Path.cwd() / "dist")
        shortcut.save()
        
        print(f"✓ Desktop shortcut created")
        
    except ImportError:
        print("⚠ Could not create desktop shortcut (missing winshell/pywin32)")
    except Exception as e:
        print(f"⚠ Could not create desktop shortcut: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick build script for DeepFaceLive")
    parser.add_argument("--install-deps", action="store_true", help="Install build dependencies")
    parser.add_argument("--shortcut", action="store_true", help="Create desktop shortcut (Windows)")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_requirements()
    
    success = quick_build()
    
    if success and args.shortcut:
        create_desktop_shortcut()
    
    if success:
        print("\n🎉 SUCCESS! Your DeepFaceLive executable is ready!")
        print("\nTo run:")
        if os.name == 'nt':
            print("  • Double-click: dist/Start_DeepFaceLive.bat")
            print("  • Or run: dist/DeepFaceLive_FreezeFrame.exe run DeepFaceLive")
        else:
            print("  • Run: ./dist/start_deepfacelive.sh")
            print("  • Or: ./dist/DeepFaceLive_FreezeFrame run DeepFaceLive")
    else:
        print("\n❌ Build failed. Check the error messages above.")
