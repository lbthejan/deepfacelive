import argparse
import os
import platform
from pathlib import Path

from xlib import appargs as lib_appargs
from xlib import os as lib_os

# Freeze-frame functionality
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "freeze_frame_modules"))

try:
    from deepfacelive_app_patch import DeepFaceLiveAppPatch

    FREEZE_FRAME_AVAILABLE = True
    print("✓ Freeze-frame functionality loaded successfully")
except ImportError as e:
    print(f"Warning: Freeze-frame modules not found ({e}). Running without freeze-frame functionality.")
    print("To enable freeze-frame: ensure all module files are in the 'freeze_frame_modules' directory")
    FREEZE_FRAME_AVAILABLE = False


# onnxruntime==1.8.0 requires CUDA_PATH_V11_2, but 1.8.1 don't
# keep the code if they return that behaviour
# if __name__ == '__main__':
#     if platform.system() == 'Windows':
#         if 'CUDA_PATH' not in os.environ:
#             raise Exception('CUDA_PATH should be set to environ')
#         # set environ for onnxruntime
#         # os.environ['CUDA_PATH_V11_2'] = os.environ['CUDA_PATH']

# from modelhub.onnx import InsightFaceSwap

# x = InsightFaceSwap(InsightFaceSwap.get_available_devices()[0])


# import code
# code.interact(local=dict(globals(), **locals()))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run", help="Run the application.")
    run_subparsers = run_parser.add_subparsers()

    def run_DeepFaceLive(args):
        userdata_path = Path(args.userdata_dir) if args.userdata_dir else Path.cwd()
        lib_appargs.set_arg_bool('NO_CUDA', args.no_cuda)

        print('Running DeepFaceLive with freeze-frame functionality.')
        from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp

        # Apply freeze-frame patch if available
        if FREEZE_FRAME_AVAILABLE:
            try:
                print("Applying freeze-frame enhancements...")
                PatchedApp = DeepFaceLiveAppPatch.patch_deepfacelive_app(DeepFaceLiveApp)

                # Create and run the patched app
                app_instance = PatchedApp(userdata_path=userdata_path)

                # Load freeze-frame configuration if available
                freeze_config_path = userdata_path / "freeze_config.json"
                if freeze_config_path.exists():
                    print(f"Loading freeze-frame config from: {freeze_config_path}")
                else:
                    print("Using default freeze-frame configuration")

                print("✓ DeepFaceLive started with freeze-frame functionality")
                print("  - Confidence-based frame freezing enabled")
                print("  - Performance monitoring active")
                print("  - Statistics available via UI controls")

                app_instance.run()

            except Exception as e:
                print(f"Error with freeze-frame functionality: {e}")
                print("Falling back to original DeepFaceLive...")
                DeepFaceLiveApp(userdata_path=userdata_path).run()
        else:
            print("Running original DeepFaceLive (freeze-frame not available)")
            DeepFaceLiveApp(userdata_path=userdata_path).run()

    p = run_subparsers.add_parser('DeepFaceLive')
    p.add_argument('--userdata-dir', default=None, action=fixPathAction, help="Workspace directory.")
    p.add_argument('--no-cuda', action="store_true", default=False, help="Disable CUDA.")
    p.add_argument('--freeze-threshold', type=float, default=None,
                   help="Set freeze-frame confidence threshold (0.0-1.0)")
    p.add_argument('--no-freeze-overlay', action="store_true", default=False,
                   help="Disable freeze-frame statistics overlay")
    p.add_argument('--freeze-config', default=None, help="Path to freeze-frame configuration file")
    p.set_defaults(func=run_DeepFaceLive)

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

    # Freeze-frame specific commands
    if FREEZE_FRAME_AVAILABLE:
        freeze_parser = subparsers.add_parser("freeze", help="Freeze-frame utilities.")
        freeze_subparsers = freeze_parser.add_subparsers()

        def test_freeze_frame(args):
            """Test freeze-frame functionality"""
            print("Testing freeze-frame functionality...")
            try:
                from usage_examples import basic_webcam_test
                basic_webcam_test()
            except ImportError:
                print("Usage examples not found. Running basic test...")
                from freeze_frame_manager import FreezeFrameManager
                import cv2
                import numpy as np

                freeze_manager = FreezeFrameManager(confidence_threshold=args.threshold)
                cap = cv2.VideoCapture(args.webcam)

                print(f"Testing with webcam {args.webcam}, threshold {args.threshold}")
                print("Press 'q' to quit")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Simulate confidence
                    confidence = np.random.uniform(0.3, 0.9)
                    output_frame, is_frozen = freeze_manager.process_frame(frame, confidence)

                    status = "FROZEN" if is_frozen else "LIVE"
                    color = (0, 0, 255) if is_frozen else (0, 255, 0)
                    cv2.putText(output_frame, f"{status} - {confidence:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    cv2.imshow("Freeze-Frame Test", output_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

        p = freeze_subparsers.add_parser('test')
        p.add_argument('--webcam', type=int, default=0, help="Webcam index")
        p.add_argument('--threshold', type=float, default=0.75, help="Confidence threshold")
        p.set_defaults(func=test_freeze_frame)

        def benchmark_freeze_frame(args):
            """Benchmark freeze-frame performance"""
            print("Running freeze-frame performance benchmark...")
            try:
                from cpu_optimization_config import benchmark_cpu_performance
                benchmark_cpu_performance()
            except ImportError:
                print("CPU optimization not found. Running basic benchmark...")
                from performance_monitor import PerformanceMonitor
                import time
                import numpy as np

                monitor = PerformanceMonitor()
                start_time = time.time()
                frame_count = 0

                while time.time() - start_time < args.duration:
                    process_start = time.time()
                    # Simulate frame processing
                    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    time.sleep(0.01)  # Simulate processing time
                    process_time = time.time() - process_start

                    monitor.update(process_time, np.random.uniform(0.5, 0.9), False)
                    frame_count += 1

                stats = monitor.get_comprehensive_stats()
                print(f"Processed {frame_count} frames in {args.duration} seconds")
                print(f"Average FPS: {stats['current']['fps']:.1f}")

        p = freeze_subparsers.add_parser('benchmark')
        p.add_argument('--duration', type=int, default=30, help="Benchmark duration in seconds")
        p.set_defaults(func=benchmark_freeze_frame)

        def config_freeze_frame(args):
            """Configure freeze-frame settings"""
            print("Configuring freeze-frame settings...")
            from freeze_frame_manager import FreezeFrameManager

            freeze_manager = FreezeFrameManager()

            if args.threshold:
                freeze_manager.update_threshold(args.threshold)
                print(f"Threshold updated to: {args.threshold}")

            if args.buffer_size:
                freeze_manager.frame_buffer_size = args.buffer_size
                freeze_manager.save_config()
                print(f"Buffer size updated to: {args.buffer_size}")

            if args.show_current:
                stats = freeze_manager.get_stats()
                print(f"Current configuration:")
                print(f"  Threshold: {stats['threshold']}")
                print(f"  Buffer size: {freeze_manager.frame_buffer_size}")

        p = freeze_subparsers.add_parser('config')
        p.add_argument('--threshold', type=float, help="Set confidence threshold")
        p.add_argument('--buffer-size', type=int, help="Set frame buffer size")
        p.add_argument('--show-current', action='store_true', help="Show current configuration")
        p.set_defaults(func=config_freeze_frame)

    def bad_args(arguments):
        parser.print_help()
        if FREEZE_FRAME_AVAILABLE:
            print("\nFreeze-frame commands available:")
            print("  python main.py freeze test          - Test freeze-frame functionality")
            print("  python main.py freeze benchmark     - Run performance benchmark")
            print("  python main.py freeze config        - Configure freeze-frame settings")
        exit(0)

    parser.set_defaults(func=bad_args)

    args = parser.parse_args()

    # Handle freeze-frame specific arguments for DeepFaceLive command
    if hasattr(args, 'freeze_threshold') and args.freeze_threshold is not None:
        if FREEZE_FRAME_AVAILABLE:
            # Store freeze-frame arguments for the app to use
            os.environ['FREEZE_THRESHOLD'] = str(args.freeze_threshold)
            print(f"Freeze threshold set to: {args.freeze_threshold}")

    if hasattr(args, 'no_freeze_overlay') and args.no_freeze_overlay:
        os.environ['NO_FREEZE_OVERLAY'] = '1'
        print("Freeze-frame overlay disabled")

    if hasattr(args, 'freeze_config') and args.freeze_config:
        os.environ['FREEZE_CONFIG_PATH'] = args.freeze_config
        print(f"Using freeze config: {args.freeze_config}")

    args.func(args)


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


if __name__ == '__main__':
    main()

# import code
# code.interact(local=dict(globals(), **locals()))