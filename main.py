import argparse
import os
import platform
from pathlib import Path

from xlib import appargs as lib_appargs
from xlib import os as lib_os

import sys

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import collections

collections.Iterable = Iterable


# Freeze-frame functionality setup
def setup_freeze_frame():
    """Setup freeze-frame functionality"""
    current_dir = Path(__file__).parent
    freeze_modules_path = current_dir / "freeze_frame_modules"

    if freeze_modules_path.exists():
        if str(freeze_modules_path) not in sys.path:
            sys.path.insert(0, str(freeze_modules_path))

        try:
            # Test import with correct module path
            from freeze_frame_manager import FreezeFrameManager
            print("✓ Freeze-frame functionality enabled")
            return True
        except ImportError as e:
            print(f"⚠️  Freeze-frame modules found but couldn't import: {e}")
            return False
    else:
        print("ℹ️  No freeze-frame modules found (running original DeepFaceLive)")
        return False


# Initialize freeze-frame
FREEZE_FRAME_AVAILABLE = setup_freeze_frame()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()

    def run_DeepFaceLive(args):
        userdata_path = Path(args.userdata_dir) if args.userdata_dir else Path.cwd()
        lib_appargs.set_arg_bool('NO_CUDA', args.no_cuda)

        print(
            'Running DeepFaceLive with freeze-frame functionality.' if FREEZE_FRAME_AVAILABLE else 'Running original DeepFaceLive.')

        # Import the DeepFaceLive app
        from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp

        # Apply freeze-frame integration if available
        if FREEZE_FRAME_AVAILABLE:
            try:
                print("🚀 Integrating freeze-frame functionality...")

                # FIXED: Correct imports without relative paths
                from freeze_frame_modules.deepfacelive_integration import DeepFaceLiveFreezeProcessor
                from freeze_frame_modules.freeze_frame_manager import FreezeFrameManager

                # Create the enhanced app class
                class EnhancedDeepFaceLiveApp(DeepFaceLiveApp):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)

                        # Initialize freeze processor with error handling
                        try:
                            self.freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)

                            # Set freeze threshold from args or environment
                            threshold = getattr(args, 'freeze_threshold', None) or \
                                        float(os.environ.get('FREEZE_THRESHOLD', '0.75'))
                            self.freeze_processor.freeze_manager.update_threshold(threshold)

                            # Configure overlay
                            if getattr(args, 'no_freeze_overlay', False) or \
                                    os.environ.get('NO_FREEZE_OVERLAY'):
                                self.freeze_processor.show_stats_overlay = False

                            print(f"✓ Freeze-frame initialized (threshold: {threshold})")
                        except Exception as e:
                            print(f"⚠️  Failed to initialize freeze processor: {e}")
                            self.freeze_processor = None

                    def on_app_start(self):
                        """Called when app starts - setup freeze processor"""
                        result = super().on_app_start()

                        # Try to connect to face detector
                        if self.freeze_processor:
                            try:
                                # Look for face detector in the app with multiple strategies
                                face_detector = None

                                # Strategy 1: Check backends list
                                if hasattr(self, '_backends'):
                                    for backend in self._backends:
                                        if hasattr(backend,
                                                   '__class__') and 'FaceDetector' in backend.__class__.__name__:
                                            face_detector = backend
                                            break

                                # Strategy 2: Check direct attributes
                                if not face_detector:
                                    for attr_name in ['face_detector', '_face_detector', 'detector']:
                                        if hasattr(self, attr_name):
                                            face_detector = getattr(self, attr_name)
                                            break

                                if face_detector:
                                    self.freeze_processor.initialize_face_detector(face_detector)
                                    print("✓ Freeze processor connected to face detector")
                                else:
                                    print("⚠️  Could not find face detector for freeze processor")

                            except Exception as e:
                                print(f"⚠️  Could not connect freeze processor to detector: {e}")

                        return result

                    def _process_frame_freeze(self, frame, backend_dict):
                        """Enhanced frame processing with freeze-frame logic"""

                        # Safety check for freeze processor
                        if not self.freeze_processor:
                            return super()._process_frame(frame, backend_dict) if hasattr(super(),
                                                                                          '_process_frame') else frame

                        # Check if we have necessary components
                        face_swapper = backend_dict.get('face_swapper')
                        face_merger = backend_dict.get('face_merger')

                        if not face_swapper:
                            return super()._process_frame(frame, backend_dict) if hasattr(super(),
                                                                                          '_process_frame') else frame

                        try:
                            # Use freeze processor if available and connected
                            if (self.freeze_processor and
                                    self.freeze_processor.face_detector_adapter):

                                return self.freeze_processor.process_frame(
                                    frame, face_swapper, None, face_merger
                                )
                            else:
                                # Fallback to original processing
                                return super()._process_frame(frame, backend_dict) if hasattr(super(),
                                                                                              '_process_frame') else frame

                        except Exception as e:
                            print(f"Freeze-frame processing error: {e}")
                            # Fallback to original processing
                            return super()._process_frame(frame, backend_dict) if hasattr(super(),
                                                                                          '_process_frame') else frame

                    def _process_frame(self, frame, backend_dict):
                        """Override frame processing to include freeze-frame"""
                        if hasattr(self, 'freeze_processor') and self.freeze_processor:
                            return self._process_frame_freeze(frame, backend_dict)
                        else:
                            return super()._process_frame(frame, backend_dict) if hasattr(super(),
                                                                                          '_process_frame') else frame

                # Use the enhanced app
                app_class = EnhancedDeepFaceLiveApp
                print("✓ Enhanced DeepFaceLive app created")

            except Exception as e:
                print(f"⚠️  Error setting up freeze-frame: {e}")
                print("   Falling back to original DeepFaceLive")
                app_class = DeepFaceLiveApp
        else:
            app_class = DeepFaceLiveApp

        # Create and run the app
        try:
            app = app_class(userdata_path=userdata_path)
            app.run()
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            print("Please check your DeepFaceLive installation and try again.")

    p = run_subparsers.add_parser('DeepFaceLive')
    p.add_argument('--userdata-dir', default=None, action=fixPathAction, help="Workspace directory.")
    p.add_argument('--no-cuda', action="store_true", default=False, help="Disable CUDA.")

    # Freeze-frame specific arguments
    if FREEZE_FRAME_AVAILABLE:
        p.add_argument('--freeze-threshold', type=float, default=None,
                       help="Set freeze-frame confidence threshold (0.0-1.0, default: 0.75)")
        p.add_argument('--no-freeze-overlay', action="store_true", default=False,
                       help="Disable freeze-frame statistics overlay")
        p.add_argument('--freeze-config', default=None,
                       help="Path to freeze-frame configuration file")

    p.set_defaults(func=run_DeepFaceLive)

    # Rest of the original main.py code remains the same...
    dev_parser = subparsers.add_parser("dev")
    dev_subparsers = dev_parser.add_subparsers()

    def run_split_large_files(args):
        from scripts import dev
        dev.split_large_files()

    p = dev_subparsers.add_parser('split_large_files')
    p.set_defaults(func=run_split_large_files)

    def run_merge_large_files(args):
        from scripts import dev
        dev.merge_large_files(delete_parts=args.delete_parts)

    p = dev_subparsers.add_parser('merge_large_files')
    p.add_argument('--delete-parts', action="store_true", default=False)
    p.set_defaults(func=run_merge_large_files)

    def run_extract_FaceSynthetics(args):
        from scripts import dev

        inputdir_path = Path(args.input_dir)
        faceset_path = Path(args.faceset_path)

        dev.extract_FaceSynthetics(inputdir_path, faceset_path)

    p = dev_subparsers.add_parser('extract_FaceSynthetics')
    p.add_argument('--input-dir', default=None, action=fixPathAction, help="FaceSynthetics directory.")
    p.add_argument('--faceset-path', default=None, action=fixPathAction, help="output .dfs path")
    p.set_defaults(func=run_extract_FaceSynthetics)

    train_parser = subparsers.add_parser("train", help="Train neural network.")
    train_parsers = train_parser.add_subparsers()

    def train_FaceAligner(args):
        lib_os.set_process_priority(lib_os.ProcessPriority.IDLE)
        from apps.trainers.FaceAligner.FaceAlignerTrainerApp import FaceAlignerTrainerApp
        FaceAlignerTrainerApp(workspace_path=Path(args.workspace_dir), faceset_path=Path(args.faceset_path))

    p = train_parsers.add_parser('FaceAligner')
    p.add_argument('--workspace-dir', default=None, action=fixPathAction, help="Workspace directory.")
    p.add_argument('--faceset-path', default=None, action=fixPathAction, help=".dfs path")
    p.set_defaults(func=train_FaceAligner)

    # Freeze-frame utility commands
    if FREEZE_FRAME_AVAILABLE:
        freeze_parser = subparsers.add_parser("freeze", help="Freeze-frame utilities.")
        freeze_subparsers = freeze_parser.add_subparsers()

        def test_freeze_frame(args):
            """Test freeze-frame functionality with webcam"""
            print("🧪 Testing freeze-frame functionality...")

            try:
                from freeze_frame_modules.freeze_frame_manager import FreezeFrameManager
                from freeze_frame_modules.performance_monitor import PerformanceMonitor
                import cv2
                import numpy as np
                import time

                freeze_manager = FreezeFrameManager(confidence_threshold=args.threshold)
                performance_monitor = PerformanceMonitor()

                cap = cv2.VideoCapture(args.webcam)
                if not cap.isOpened():
                    print(f"❌ Could not open webcam {args.webcam}")
                    return

                print(f"✓ Testing with webcam {args.webcam}, threshold {args.threshold}")
                print("Controls: 'q' to quit, 's' for stats, '+'/'-' to adjust threshold")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Simulate face detection confidence
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray) / 255.0
                    confidence = max(0.0, min(1.0, brightness + np.random.normal(0, 0.1)))

                    start_time = time.time()
                    output_frame, is_frozen = freeze_manager.process_frame(frame, confidence)
                    process_time = time.time() - start_time

                    performance_monitor.update(process_time, confidence, is_frozen)

                    # Add status overlay
                    status = "FROZEN" if is_frozen else "LIVE"
                    color = (0, 0, 255) if is_frozen else (0, 255, 0)
                    cv2.putText(output_frame, f"{status} - Conf: {confidence:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(output_frame, f"Threshold: {args.threshold:.2f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Freeze-Frame Test", output_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        stats = freeze_manager.get_stats()
                        perf_stats = performance_monitor.get_rolling_stats()
                        print(f"Stats - Freezes: {stats['freeze_count']}, "
                              f"FPS: {perf_stats['fps']:.1f}, "
                              f"Freeze Ratio: {perf_stats['freeze_ratio']:.1%}")
                    elif key == ord('+') or key == ord('='):
                        args.threshold = min(1.0, args.threshold + 0.05)
                        freeze_manager.update_threshold(args.threshold)
                        print(f"Threshold increased to: {args.threshold:.2f}")
                    elif key == ord('-'):
                        args.threshold = max(0.0, args.threshold - 0.05)
                        freeze_manager.update_threshold(args.threshold)
                        print(f"Threshold decreased to: {args.threshold:.2f}")

                cap.release()
                cv2.destroyAllWindows()

                # Final stats
                final_stats = freeze_manager.get_stats()
                print(f"\n✅ Test completed - Total freezes: {final_stats['freeze_count']}")

            except Exception as e:
                print(f"❌ Test failed: {e}")

        p = freeze_subparsers.add_parser('test')
        p.add_argument('--webcam', type=int, default=0, help="Webcam index")
        p.add_argument('--threshold', type=float, default=0.75, help="Confidence threshold")
        p.set_defaults(func=test_freeze_frame)

        def config_freeze_frame(args):
            """Configure freeze-frame settings"""
            print("⚙️  Configuring freeze-frame settings...")

            try:
                from freeze_frame_modules.freeze_frame_manager import FreezeFrameManager
                import json

                config_path = Path("freeze_config.json")

                if args.show_current:
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        print("Current configuration:")
                        for key, value in config.items():
                            print(f"  {key}: {value}")
                    else:
                        print("No configuration file found")
                    return

                # Load or create config
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {
                        "confidence_threshold": 0.75,
                        "max_freeze_duration": 5.0,
                        "frame_buffer_size": 3,
                        "enable_stats_overlay": True
                    }

                # Update config with provided arguments
                if args.threshold is not None:
                    config["confidence_threshold"] = args.threshold
                    print(f"✓ Threshold set to: {args.threshold}")

                if args.buffer_size is not None:
                    config["frame_buffer_size"] = args.buffer_size
                    print(f"✓ Buffer size set to: {args.buffer_size}")

                if args.max_freeze is not None:
                    config["max_freeze_duration"] = args.max_freeze
                    print(f"✓ Max freeze duration set to: {args.max_freeze}s")

                # Save config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                print(f"✓ Configuration saved to: {config_path}")

            except Exception as e:
                print(f"❌ Configuration failed: {e}")

        p = freeze_subparsers.add_parser('config')
        p.add_argument('--threshold', type=float, help="Set confidence threshold")
        p.add_argument('--buffer-size', type=int, help="Set frame buffer size")
        p.add_argument('--max-freeze', type=float, help="Set maximum freeze duration")
        p.add_argument('--show-current', action='store_true', help="Show current configuration")
        p.set_defaults(func=config_freeze_frame)

    def bad_args(arguments):
        parser.print_help()
        if FREEZE_FRAME_AVAILABLE:
            print("\n🎭 Freeze-frame commands available:")
            print("  python main.py freeze test          - Test freeze-frame functionality")
            print("  python main.py freeze config        - Configure freeze-frame settings")
            print("\n🚀 Enhanced DeepFaceLive usage:")
            print("  python main.py run DeepFaceLive --freeze-threshold 0.8")
            print("  python main.py run DeepFaceLive --no-freeze-overlay")
        exit(0)

    parser.set_defaults(func=bad_args)

    args = parser.parse_args()
    args.func(args)


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

try:
    from ui_integration_patch import integrate_freeze_ui_into_main
    integrate_freeze_ui_into_main()
except ImportError:
    pass

if __name__ == '__main__':
    main()