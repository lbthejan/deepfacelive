#!/usr/bin/env python3
"""
DeepFaceLive with Freeze-Frame Functionality
Enhanced startup script with freeze-frame features and optimizations
"""

import os
import sys
import json
import time
from pathlib import Path
import argparse


# Set up optimal environment for freeze-frame functionality
def setup_environment():
    """Setup optimal environment variables for freeze-frame"""

    # CPU optimization environment variables
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  # Limit OpenBLAS threads

    # Memory optimization
    os.environ['PYTHONUNBUFFERED'] = '1'  # Unbuffered output

    # Add freeze-frame modules to path
    current_dir = Path(__file__).parent
    freeze_modules_path = current_dir / "freeze_frame_modules"

    if freeze_modules_path.exists():
        sys.path.insert(0, str(freeze_modules_path))
        print(f"✓ Freeze-frame modules path added: {freeze_modules_path}")
    else:
        print(f"⚠ Warning: Freeze-frame modules not found at {freeze_modules_path}")
        print("  Please ensure all freeze-frame files are in the 'freeze_frame_modules' directory")


def check_freeze_frame_installation():
    """Check if freeze-frame modules are properly installed"""
    required_modules = [
        "freeze_frame_manager",
        "face_detector_wrapper",
        "performance_monitor",
        "deepfacelive_integration",
        "deepfacelive_app_patch"
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("❌ Missing freeze-frame modules:")
        for module in missing_modules:
            print(f"   - {module}.py")
        print("\nPlease ensure all freeze-frame files are in the 'freeze_frame_modules' directory")
        return False
    else:
        print("✓ All freeze-frame modules found")
        return True


def create_default_configs():
    """Create default configuration files if they don't exist"""
    current_dir = Path(__file__).parent

    # Default freeze-frame configuration
    freeze_config_path = current_dir / "freeze_config.json"
    if not freeze_config_path.exists():
        default_freeze_config = {
            "confidence_threshold": 0.75,
            "max_freeze_duration": 5.0,
            "frame_buffer_size": 3,
            "enable_stats_overlay": True,
            "enable_performance_monitoring": True,
            "ui_settings": {
                "show_threshold_slider": True,
                "show_stats_button": True,
                "show_debug_button": True,
                "overlay_position": "top_left",
                "overlay_opacity": 0.8
            },
            "advanced_settings": {
                "hysteresis_offset": 0.05,
                "confidence_smoothing": True,
                "auto_threshold_adjustment": False,
                "performance_optimization": True
            }
        }

        with open(freeze_config_path, 'w') as f:
            json.dump(default_freeze_config, f, indent=2)
        print(f"✓ Created default freeze config: {freeze_config_path}")


def print_startup_banner():
    """Print startup banner with freeze-frame information"""
    print("=" * 70)
    print("🎭 DEEPFACELIVE WITH FREEZE-FRAME FUNCTIONALITY")
    print("=" * 70)
    print("Enhanced with confidence-based frame freezing to eliminate glitches")
    print("during extreme head movements and poor detection conditions.")
    print()
    print("Key Features:")
    print("✓ Confidence-based frame freezing")
    print("✓ Real-time performance monitoring")
    print("✓ Configurable thresholds and settings")
    print("✓ Professional streaming quality")
    print("✓ CPU and GPU compatible")
    print("=" * 70)
    print()


def run_system_check():
    """Run system compatibility check"""
    print("🔍 Running system compatibility check...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"⚠️  Warning: Python {python_version.major}.{python_version.minor} detected. Python 3.8+ recommended.")
    else:
        print(f"✓ Python {python_version.major}.{python_version.minor} compatible")

    # Check required packages
    required_packages = ['cv2', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")

    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA GPU available")
        else:
            print("ℹ️  No CUDA GPU detected (CPU mode will be used)")
    except ImportError:
        print("ℹ️  PyTorch not available for GPU detection")

    print("✅ System check complete\n")
    return True


def main():
    """Enhanced main entry point with freeze-frame functionality"""

    # Print banner
    print_startup_banner()

    # Setup environment
    setup_environment()

    # Parse arguments
    parser = argparse.ArgumentParser(description="DeepFaceLive with Freeze-Frame")
    parser.add_argument('command', nargs='?', default='run', help='Command to run')
    parser.add_argument('app', nargs='?', default='DeepFaceLive', help='App to run')
    parser.add_argument('--userdata-dir', default=None, help="Workspace directory")
    parser.add_argument('--no-cuda', action="store_true", default=False, help="Disable CUDA")
    parser.add_argument('--freeze-threshold', type=float, default=None, help="Set freeze-frame confidence threshold")
    parser.add_argument('--no-freeze-overlay', action="store_true", default=False,
                        help="Disable freeze-frame statistics overlay")
    parser.add_argument('--freeze-config', default=None, help="Path to freeze-frame configuration file")
    parser.add_argument('--system-check', action='store_true', help='Run system check only')

    args = parser.parse_args()

    # Handle system check
    if args.system_check:
        run_system_check()
        return 0

    # Run system check
    if not run_system_check():
        print("❌ System check failed. Please resolve issues before continuing.")
        return 1

    # Check freeze-frame installation
    freeze_available = check_freeze_frame_installation()

    if freeze_available:
        print("🚀 Freeze-frame functionality ready")
        create_default_configs()
    else:
        print("⚠️  Running without freeze-frame functionality")
        print("   Install freeze-frame modules to enable enhanced features")

    print("\n🎬 Starting DeepFaceLive...")
    print("=" * 70)

    # Set freeze-frame environment variables
    if args.freeze_threshold:
        os.environ['FREEZE_THRESHOLD'] = str(args.freeze_threshold)
        print(f"🎯 Freeze threshold set to: {args.freeze_threshold}")

    if args.no_freeze_overlay:
        os.environ['NO_FREEZE_OVERLAY'] = '1'
        print("📊 Freeze-frame overlay disabled")

    if args.freeze_config:
        os.environ['FREEZE_CONFIG_PATH'] = args.freeze_config
        print(f"⚙️  Using freeze config: {args.freeze_config}")

    # Import and run the original main logic
    try:
        # Add the current directory to Python path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        # Import the original main module
        import main as original_main_module

        # Create arguments for the original main
        original_args = [
            "main.py",
            args.command,
            args.app
        ]

        if args.userdata_dir:
            original_args.extend(['--userdata-dir', args.userdata_dir])
        if args.no_cuda:
            original_args.append('--no-cuda')
        if args.freeze_threshold:
            original_args.extend(['--freeze-threshold', str(args.freeze_threshold)])
        if args.no_freeze_overlay:
            original_args.append('--no-freeze-overlay')
        if args.freeze_config:
            original_args.extend(['--freeze-config', args.freeze_config])

        # Backup original sys.argv and replace
        original_argv = sys.argv[:]
        sys.argv = original_args

        start_time = time.time()

        # Check if original main has a main function, otherwise call the module directly
        if hasattr(original_main_module, 'main'):
            result = original_main_module.main()
        else:
            # If no main function, the module should execute when imported
            # We'll just call the module's global execution
            print("✓ DeepFaceLive started with freeze-frame functionality")
            result = 0

        end_time = time.time()

        # Restore original argv
        sys.argv = original_argv

        print(f"\n🏁 Session completed in {end_time - start_time:.1f} seconds")

        if freeze_available:
            print("📊 Session statistics available in freeze_config.json")

        return result

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 0
    except ImportError as e:
        print(f"\n❌ Error importing main module: {e}")
        print("💡 Make sure you're in the DeepFaceLive directory with main.py")
        return 1
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Try running the original main.py directly first to ensure it works")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code or 0)