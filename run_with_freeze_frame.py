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
    
    # Default integration configuration
    integration_config_path = current_dir / "freeze_integration_config.json"
    if not integration_config_path.exists():
        default_integration_config = {
            "integration_method": "patch_class",
            "freeze_settings": {
                "confidence_threshold": 0.75,
                "max_freeze_duration": 5.0,
                "frame_buffer_size": 3,
                "enable_stats_overlay": True
            },
            "ui_settings": {
                "show_threshold_slider": True,
                "show_stats_button": True,
                "show_debug_button": True,
                "show_reset_button": True
            },
            "performance_settings": {
                "monitor_fps": True,
                "monitor_confidence": True,
                "export_stats": False,
                "log_performance": False
            }
        }
        
        with open(integration_config_path, 'w') as f:
            json.dump(default_integration_config, f, indent=2)
        print(f"✓ Created default integration config: {integration_config_path}")

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

def handle_command_line_args():
    """Handle command line arguments specific to freeze-frame"""
    parser = argparse.ArgumentParser(
        description="DeepFaceLive with Freeze-Frame Functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run DeepFaceLive --userdata-dir ./userdata
  %(prog)s run DeepFaceLive --no-cuda --freeze-threshold 0.8
  %(prog)s freeze test --webcam 0 --threshold 0.75
  %(prog)s freeze benchmark --duration 60
  %(prog)s freeze config --threshold 0.8 --show-current
        """
    )
    
    # Add freeze-frame specific arguments that will be passed through
    parser.add_argument('--freeze-threshold', type=float, 
                       help='Set freeze-frame confidence threshold (0.0-1.0)')
    parser.add_argument('--no-freeze-overlay', action='store_true',
                       help='Disable freeze-frame statistics overlay')
    parser.add_argument('--freeze-config', 
                       help='Path to freeze-frame configuration file')
    parser.add_argument('--cpu-optimized', action='store_true',
                       help='Use CPU-optimized settings')
    parser.add_argument('--show-freeze-help', action='store_true',
                       help='Show freeze-frame specific help and exit')
    
    # Parse known args to avoid conflicts with main.py
    known_args, remaining_args = parser.parse_known_args()
    
    if known_args.show_freeze_help:
        print_freeze_frame_help()
        sys.exit(0)
    
    # Apply freeze-frame specific settings
    if known_args.freeze_threshold:
        os.environ['FREEZE_THRESHOLD'] = str(known_args.freeze_threshold)
        print(f"🎯 Freeze threshold set to: {known_args.freeze_threshold}")
    
    if known_args.no_freeze_overlay:
        os.environ['NO_FREEZE_OVERLAY'] = '1'
        print("📊 Freeze-frame overlay disabled")
    
    if known_args.freeze_config:
        os.environ['FREEZE_CONFIG_PATH'] = known_args.freeze_config
        print(f"⚙️  Using freeze config: {known_args.freeze_config}")
    
    if known_args.cpu_optimized:
        setup_cpu_optimizations()
        print("🖥️  CPU optimizations enabled")
    
    # Return the remaining arguments for main.py
    return remaining_args

def setup_cpu_optimizations():
    """Setup CPU-specific optimizations"""
    # Create CPU-optimized configuration if not exists
    current_dir = Path(__file__).parent
    cpu_config_path = current_dir / "cpu_freeze_config.json"
    
    if not cpu_config_path.exists():
        cpu_optimized_config = {
            "confidence_threshold": 0.7,
            "max_freeze_duration": 3.0,
            "frame_buffer_size": 2,
            "enable_stats_overlay": False,
            "enable_performance_monitoring": True,
            "cpu_optimizations": {
                "reduce_frame_processing": True,
                "skip_heavy_operations": True,
                "optimize_memory_usage": True
            }
        }
        
        with open(cpu_config_path, 'w') as f:
            json.dump(cpu_optimized_config, f, indent=2)
        print(f"✓ Created CPU-optimized config: {cpu_config_path}")
    
    # Set environment variable to use CPU config
    os.environ['FREEZE_CONFIG_PATH'] = str(cpu_config_path)

def print_freeze_frame_help():
    """Print freeze-frame specific help information"""
    help_text = """
🎭 DEEPFACELIVE FREEZE-FRAME FUNCTIONALITY HELP

The freeze-frame feature eliminates glitchy visuals during face detection failures
by freezing the last good frame when confidence drops below a threshold.

CONFIGURATION:
  --freeze-threshold FLOAT    Confidence threshold (0.0-1.0, default: 0.75)
                             Lower = freezes less often (may show glitches)
                             Higher = freezes more often (very stable)

  --freeze-config PATH       Custom configuration file path

  --no-freeze-overlay        Disable statistics overlay (better performance)

  --cpu-optimized           Use settings optimized for CPU-only systems

FREEZE-FRAME COMMANDS:
  python main.py freeze test              Test freeze-frame with webcam
  python main.py freeze benchmark         Performance benchmark
  python main.py freeze config            Configure settings

USAGE SCENARIOS:
  Good Lighting:     --freeze-threshold 0.8
  Poor Lighting:     --freeze-threshold 0.6  
  Fast Movement:     --freeze-threshold 0.7
  CPU-only:          --cpu-optimized

CONFIGURATION FILES:
  freeze_config.json                Main freeze-frame settings
  freeze_integration_config.json   Integration settings
  cpu_freeze_config.json           CPU-optimized settings

CONTROLS (when overlay enabled):
  Shows real-time status: LIVE/FROZEN
  Displays current confidence and threshold
  Shows FPS and performance metrics

For more information, see the documentation in the freeze_frame_modules directory.
"""
    print(help_text)

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

def monitor_performance():
    """Monitor performance and provide recommendations"""
    try:
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        
        # This would be called periodically during app execution
        # For now, just set up the monitor
        print("📊 Performance monitoring initialized")
        
    except ImportError:
        print("⚠️  Performance monitoring not available")

def main():
    """Enhanced main entry point with freeze-frame functionality"""
    
    # Print banner
    print_startup_banner()
    
    # Setup environment
    setup_environment()
    
    # Handle command line arguments
    remaining_args = handle_command_line_args()
    
    # Run system check
    if not run_system_check():
        print("❌ System check failed. Please resolve issues before continuing.")
        return 1
    
    # Check freeze-frame installation
    freeze_available = check_freeze_frame_installation()
    
    if freeze_available:
        print("🚀 Freeze-frame functionality ready")
        
        # Create default configs if needed
        create_default_configs()
        
        # Setup performance monitoring
        monitor_performance()
        
    else:
        print("⚠️  Running without freeze-frame functionality")
        print("   Install freeze-frame modules to enable enhanced features")
    
    print("\n🎬 Starting DeepFaceLive...")
    print("=" * 70)
    
    # Import and run the original main
    try:
        # Restore original sys.argv with remaining arguments
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + remaining_args
        
        # Import and run main
        from main import main as original_main
        
        start_time = time.time()
        result = original_main()
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
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Try running with --show-freeze-help for configuration options")
        return 1

def quick_test():
    """Quick test function for freeze-frame functionality"""
    print("🧪 Running quick freeze-frame test...")
    
    try:
        from freeze_frame_manager import FreezeFrameManager
        
        # Test freeze manager
        freeze_manager = FreezeFrameManager()
        
        # Test with sample confidence values
        test_confidences = [0.9, 0.8, 0.5, 0.3, 0.8, 0.9]
        
        import numpy as np
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        print("Testing freeze logic with sample confidence values:")
        for i, conf in enumerate(test_confidences):
            output_frame, is_frozen = freeze_manager.process_frame(dummy_frame, conf)
            status = "FROZEN" if is_frozen else "LIVE"
            print(f"  Frame {i+1}: confidence={conf:.1f} → {status}")
        
        stats = freeze_manager.get_stats()
        print(f"\nTest Results:")
        print(f"  Threshold: {stats['threshold']}")
        print(f"  Freezes: {stats['freeze_count']}")
        print(f"  Total freeze time: {stats['total_freeze_time']:.1f}s")
        
        print("✅ Quick test completed successfully")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            quick_test()
            sys.exit(0)
        elif sys.argv[1] == "--system-check":
            run_system_check()
            sys.exit(0)
        elif sys.argv[1] == "--help-freeze":
            print_freeze_frame_help()
            sys.exit(0)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code or 0)