"""
Usage Examples for DeepFaceLive Freeze-Frame Feature
Demonstrates different ways to use and configure the freeze-frame functionality
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Import freeze-frame modules
from freeze_frame_manager import FreezeFrameManager
from face_detector_wrapper import FaceDetectorWrapper
from performance_monitor import PerformanceMonitor
from deepfacelive_integration import DeepFaceLiveFreezeProcessor


def basic_webcam_test():
    """
    Basic test using webcam to demonstrate freeze-frame functionality
    """
    print("Basic Webcam Test - Freeze Frame Demo")
    print("Controls: 'q' to quit, 's' for stats, 't' to adjust threshold")
    
    # Initialize components
    freeze_manager = FreezeFrameManager(confidence_threshold=0.75)
    performance_monitor = PerformanceMonitor()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        process_start = time.time()
        
        # Simulate face detection confidence based on frame properties
        # In real usage, this would come from actual face detection
        confidence = simulate_face_detection_confidence(frame)
        
        # Process frame with freeze logic
        output_frame, is_frozen = freeze_manager.process_frame(frame, confidence)
        
        # Update performance monitoring
        process_time = time.time() - process_start
        performance_monitor.update(process_time, confidence, is_frozen)
        
        # Add visual indicators
        output_frame = add_status_overlay(output_frame, confidence, is_frozen, 
                                        freeze_manager.confidence_threshold)
        
        # Display frame
        cv2.imshow('Freeze-Frame Demo', output_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print_current_stats(freeze_manager, performance_monitor)
        elif key == ord('t'):
            new_threshold = adjust_threshold_interactive(freeze_manager.confidence_threshold)
            freeze_manager.update_threshold(new_threshold)
        
        frame_count += 1
    
    # Cleanup and final stats
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print_final_stats(freeze_manager, performance_monitor, frame_count, start_time)


def simulate_face_detection_confidence(frame):
    """
    Simulate face detection confidence for demo purposes
    In real usage, this would come from actual face detector
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use frame brightness as a proxy for detection quality
    brightness = np.mean(gray) / 255.0
    
    # Add some noise to simulate real detection variance
    noise = np.random.normal(0, 0.1)
    confidence = np.clip(brightness + noise, 0.0, 1.0)
    
    # Simulate detection failures (very low confidence) occasionally
    if np.random.random() < 0.05:  # 5% chance of detection failure
        confidence = np.random.uniform(0.0, 0.3)
    
    return confidence


def add_status_overlay(frame, confidence, is_frozen, threshold):
    """Add status information overlay to frame"""
    overlay = frame.copy()
    
    # Status indicator
    status_text = "FROZEN" if is_frozen else "LIVE"
    status_color = (0, 0, 255) if is_frozen else (0, 255, 0)
    
    # Background rectangle
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status text
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    # Confidence info
    cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Threshold info
    cv2.putText(frame, f"Threshold: {threshold:.3f}", (20, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Confidence bar
    bar_width = 200
    bar_height = 10
    bar_x, bar_y = 20, 105
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Confidence level bar
    conf_width = int(confidence * bar_width)
    conf_color = (0, 255, 0) if confidence >= threshold else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), conf_color, -1)
    
    # Threshold line
    thresh_x = int(threshold * bar_width) + bar_x
    cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), (255, 255, 0), 2)
    
    return frame


def print_current_stats(freeze_manager, performance_monitor):
    """Print current statistics"""
    freeze_stats = freeze_manager.get_stats()
    perf_stats = performance_monitor.get_rolling_stats()
    
    print("\n" + "="*40)
    print("CURRENT STATISTICS")
    print("="*40)
    print(f"Status: {'FROZEN' if freeze_stats['is_frozen'] else 'LIVE'}")
    print(f"Current Confidence: {freeze_stats['current_confidence']:.3f}")
    print(f"Threshold: {freeze_stats['threshold']:.3f}")
    print(f"FPS: {perf_stats['fps']:.1f}")
    print(f"Freeze Count: {freeze_stats['freeze_count']}")
    print(f"Total Freeze Time: {freeze_stats['total_freeze_time']:.1f}s")
    print(f"Freeze Ratio: {perf_stats['freeze_ratio']:.1%}")
    print("="*40)


def adjust_threshold_interactive(current_threshold):
    """Interactive threshold adjustment"""
    print(f"\nCurrent threshold: {current_threshold:.3f}")
    try:
        new_threshold = float(input("Enter new threshold (0.0-1.0): "))
        new_threshold = max(0.0, min(1.0, new_threshold))
        print(f"Threshold updated to: {new_threshold:.3f}")
        return new_threshold
    except ValueError:
        print("Invalid input. Keeping current threshold.")
        return current_threshold


def print_final_stats(freeze_manager, performance_monitor, frame_count, start_time):
    """Print final session statistics"""
    session_duration = time.time() - start_time
    avg_fps = frame_count / session_duration
    
    freeze_stats = freeze_manager.get_stats()
    
    print("\n" + "="*50)
    print("FINAL SESSION STATISTICS")
    print("="*50)
    print(f"Session Duration: {session_duration:.1f} seconds")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total Freezes: {freeze_stats['freeze_count']}")
    print(f"Total Freeze Time: {freeze_stats['total_freeze_time']:.1f}s")
    print(f"Freeze Percentage: {(freeze_stats['total_freeze_time']/session_duration)*100:.1f}%")
    print("="*50)


def advanced_configuration_example():
    """
    Example of advanced configuration options
    """
    print("Advanced Configuration Example")
    
    # Custom configuration
    custom_config = {
        "confidence_threshold": 0.8,
        "max_freeze_duration": 3.0,
        "frame_buffer_size": 5,
        "enable_stats_overlay": True
    }
    
    # Initialize with custom config
    freeze_manager = FreezeFrameManager(
        confidence_threshold=custom_config["confidence_threshold"],
        config_file="custom_freeze_config.json"
    )
    
    # Save custom configuration
    freeze_manager.save_config()
    
    print("Custom configuration created and saved")
    print(f"Threshold: {custom_config['confidence_threshold']}")
    print(f"Max freeze duration: {custom_config['max_freeze_duration']}s")
    print(f"Frame buffer size: {custom_config['frame_buffer_size']}")


def performance_testing_example():
    """
    Example of performance testing and optimization
    """
    print("Performance Testing Example")
    
    # Test different configurations
    test_configs = [
        {"threshold": 0.6, "buffer_size": 1, "name": "Responsive"},
        {"threshold": 0.75, "buffer_size": 3, "name": "Balanced"},
        {"threshold": 0.85, "buffer_size": 5, "name": "Conservative"}
    ]
    
    for config in test_configs:
        print(f"\nTesting {config['name']} configuration...")
        
        freeze_manager = FreezeFrameManager(
            confidence_threshold=config["threshold"]
        )
        freeze_manager.frame_buffer_size = config["buffer_size"]
        
        performance_monitor = PerformanceMonitor()
        
        # Simulate frame processing
        for i in range(100):
            # Simulate varying confidence
            confidence = 0.5 + 0.4 * np.sin(i * 0.1) + np.random.normal(0, 0.1)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Create dummy frame
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            output_frame, is_frozen = freeze_manager.process_frame(dummy_frame, confidence)
            process_time = time.time() - start_time
            
            performance_monitor.update(process_time, confidence, is_frozen)
        
        # Print results
        stats = performance_monitor.get_rolling_stats()
        freeze_stats = freeze_manager.get_stats()
        
        print(f"  Freeze ratio: {stats['freeze_ratio']:.1%}")
        print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
        print(f"  Confidence stability: {stats['confidence_stability']:.3f}")
        print(f"  Total freezes: {freeze_stats['freeze_count']}")


def integration_with_deepfacelive_example():
    """
    Example of how to integrate with actual DeepFaceLive components
    """
    print("DeepFaceLive Integration Example")
    
    # This would be used with actual DeepFaceLive components
    try:
        # Initialize freeze processor
        processor = DeepFaceLiveFreezeProcessor()
        
        # Simulate DeepFaceLive components (in real usage, these would be actual instances)
        class MockFaceSwapper:
            def process_frame(self, frame, faces):
                # Simulate face swapping
                return frame
        
        class MockPredictor:
            def predict(self, face_img):
                # Simulate prediction
                return face_img
        
        class MockFaceDetector:
            def extract(self, frame):
                # Simulate face detection
                return [{"confidence": 0.8, "bbox": [100, 100, 200, 200]}]
        
        # Mock components
        face_swapper = MockFaceSwapper()
        predictor = MockPredictor()
        face_detector = MockFaceDetector()
        
        # Initialize detector
        processor.initialize_face_detector(face_detector)
        
        # Test frame processing
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        output_frame = processor.process_frame(test_frame, face_swapper, predictor)
        
        print("✓ Integration test successful")
        print(f"Output frame shape: {output_frame.shape}")
        
        # Print stats
        stats = processor.get_stats()
        print(f"Freeze processor stats: {stats}")
        
    except Exception as e:
        print(f"Integration test failed: {e}")


def confidence_tuning_example():
    """
    Example of how to tune confidence thresholds for different scenarios
    """
    print("Confidence Tuning Example")
    
    scenarios = {
        "Good Lighting": [0.9, 0.8, 0.85, 0.9, 0.88, 0.92, 0.87],
        "Poor Lighting": [0.6, 0.5, 0.4, 0.7, 0.55, 0.45, 0.6],
        "Fast Movement": [0.8, 0.3, 0.2, 0.85, 0.4, 0.1, 0.75],
        "Stable Position": [0.9, 0.88, 0.91, 0.89, 0.9, 0.87, 0.92]
    }
    
    recommended_thresholds = {
        "Good Lighting": 0.8,
        "Poor Lighting": 0.6,
        "Fast Movement": 0.65,
        "Stable Position": 0.85
    }
    
    for scenario, confidences in scenarios.items():
        print(f"\nScenario: {scenario}")
        print(f"Sample confidences: {confidences}")
        print(f"Recommended threshold: {recommended_thresholds[scenario]}")
        
        # Test with recommended threshold
        freeze_manager = FreezeFrameManager(confidence_threshold=recommended_thresholds[scenario])
        
        freeze_count = 0
        for conf in confidences:
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            _, is_frozen = freeze_manager.process_frame(dummy_frame, conf)
            if is_frozen:
                freeze_count += 1
        
        freeze_ratio = freeze_count / len(confidences)
        print(f"Freeze ratio with recommended threshold: {freeze_ratio:.1%}")


def real_time_monitoring_example():
    """
    Example of real-time monitoring and adjustment
    """
    print("Real-time Monitoring Example")
    print("This example shows how to monitor and adjust settings in real-time")
    
    freeze_manager = FreezeFrameManager()
    performance_monitor = PerformanceMonitor()
    
    # Simulate real-time monitoring
    print("\nStarting real-time monitoring simulation...")
    
    for minute in range(5):  # Simulate 5 minutes
        print(f"\nMinute {minute + 1}:")
        
        # Simulate 60 frames (1 per second)
        for second in range(60):
            # Simulate varying conditions
            if minute < 2:
                # Good conditions initially
                confidence = 0.8 + np.random.normal(0, 0.1)
            else:
                # Degrading conditions
                confidence = 0.6 + np.random.normal(0, 0.15)
            
            confidence = np.clip(confidence, 0.0, 1.0)
            
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            start_time = time.time()
            _, is_frozen = freeze_manager.process_frame(dummy_frame, confidence)
            process_time = time.time() - start_time
            
            performance_monitor.update(process_time, confidence, is_frozen)
        
        # Check performance and adjust if needed
        stats = performance_monitor.get_rolling_stats()
        health = performance_monitor.get_performance_health()
        
        print(f"  Health Score: {health['health_score']:.0f}/100 ({health['status']})")
        print(f"  Freeze Ratio: {stats['freeze_ratio']:.1%}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
        
        # Auto-adjust threshold based on performance
        if stats['freeze_ratio'] > 0.3:  # Too much freezing
            new_threshold = max(0.3, freeze_manager.confidence_threshold - 0.1)
            freeze_manager.update_threshold(new_threshold)
            print(f"  → Lowered threshold to {new_threshold:.2f} (too much freezing)")
        elif stats['freeze_ratio'] < 0.05:  # Too little freezing, might miss issues
            new_threshold = min(0.9, freeze_manager.confidence_threshold + 0.05)
            freeze_manager.update_threshold(new_threshold)
            print(f"  → Raised threshold to {new_threshold:.2f} (might be too lenient)")
        
        if health['recommendations']:
            print(f"  Recommendations: {', '.join(health['recommendations'])}")


def batch_processing_example():
    """
    Example of batch processing video files with freeze-frame
    """
    print("Batch Processing Example")
    print("This example shows how to process video files with freeze-frame functionality")
    
    # Simulate processing a video file
    def process_video_with_freeze_frame(video_path, output_path):
        freeze_manager = FreezeFrameManager(confidence_threshold=0.75)
        performance_monitor = PerformanceMonitor()
        
        print(f"Processing video: {video_path}")
        
        # In real usage, you would use cv2.VideoCapture to read the video
        # Here we simulate with random frames
        total_frames = 300  # Simulate 10 seconds at 30fps
        processed_frames = []
        
        for frame_num in range(total_frames):
            # Simulate varying detection quality
            if frame_num < 50 or frame_num > 250:
                # Good detection at start and end
                confidence = 0.85 + np.random.normal(0, 0.05)
            else:
                # Challenging section in the middle
                confidence = 0.5 + 0.3 * np.sin(frame_num * 0.1) + np.random.normal(0, 0.1)
            
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Simulate frame processing
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            start_time = time.time()
            output_frame, is_frozen = freeze_manager.process_frame(dummy_frame, confidence)
            process_time = time.time() - start_time
            
            performance_monitor.update(process_time, confidence, is_frozen)
            processed_frames.append(output_frame)
            
            # Progress indicator
            if frame_num % 50 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"  Progress: {progress:.0f}%")
        
        # Final statistics
        stats = performance_monitor.get_comprehensive_stats()
        freeze_stats = freeze_manager.get_stats()
        
        print(f"✓ Video processing complete")
        print(f"  Total frames: {total_frames}")
        print(f"  Freeze ratio: {stats['current']['freeze_ratio']:.1%}")
        print(f"  Average FPS: {stats['current']['fps']:.1f}")
        print(f"  Total freezes: {freeze_stats['freeze_count']}")
        
        return processed_frames, stats
    
    # Process multiple videos
    video_files = ["sample1.mp4", "sample2.mp4", "sample3.mp4"]
    
    for video_file in video_files:
        output_file = f"processed_{video_file}"
        frames, stats = process_video_with_freeze_frame(video_file, output_file)
        print(f"Saved processed video: {output_file}\n")


def command_line_interface_example():
    """
    Example of command-line interface for freeze-frame testing
    """
    import argparse
    
    def create_cli():
        parser = argparse.ArgumentParser(description="DeepFaceLive Freeze-Frame Testing")
        
        # Main command
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Run freeze-frame test')
        test_parser.add_argument('--webcam', type=int, default=0, help='Webcam index')
        test_parser.add_argument('--threshold', type=float, default=0.75, help='Confidence threshold')
        test_parser.add_argument('--show-stats', action='store_true', help='Show statistics overlay')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configure freeze-frame settings')
        config_parser.add_argument('--threshold', type=float, help='Set confidence threshold')
        config_parser.add_argument('--buffer-size', type=int, help='Set frame buffer size')
        config_parser.add_argument('--max-freeze', type=float, help='Set maximum freeze duration')
        
        # Benchmark command
        bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
        bench_parser.add_argument('--duration', type=int, default=60, help='Benchmark duration in seconds')
        
        return parser
    
    def handle_test_command(args):
        """Handle test command"""
        print(f"Running test with webcam {args.webcam}, threshold {args.threshold}")
        if args.show_stats:
            print("Statistics overlay enabled")
        # Call basic_webcam_test() here
    
    def handle_config_command(args):
        """Handle config command"""
        config_updates = {}
        if args.threshold:
            config_updates['confidence_threshold'] = args.threshold
        if args.buffer_size:
            config_updates['frame_buffer_size'] = args.buffer_size
        if args.max_freeze:
            config_updates['max_freeze_duration'] = args.max_freeze
        
        print(f"Updating configuration: {config_updates}")
        # Update configuration file here
    
    def handle_benchmark_command(args):
        """Handle benchmark command"""
        print(f"Running benchmark for {args.duration} seconds")
        # Call performance_testing_example() here
    
    # Example CLI usage
    print("Command Line Interface Example")
    print("Usage examples:")
    print("  python usage_examples.py test --webcam 0 --threshold 0.8 --show-stats")
    print("  python usage_examples.py config --threshold 0.75 --buffer-size 3")
    print("  python usage_examples.py benchmark --duration 120")


def main():
    """
    Main function to run different examples
    """
    print("DeepFaceLive Freeze-Frame Usage Examples")
    print("="*50)
    
    examples = {
        "1": ("Basic Webcam Test", basic_webcam_test),
        "2": ("Advanced Configuration", advanced_configuration_example),
        "3": ("Performance Testing", performance_testing_example),
        "4": ("DeepFaceLive Integration", integration_with_deepfacelive_example),
        "5": ("Confidence Tuning", confidence_tuning_example),
        "6": ("Real-time Monitoring", real_time_monitoring_example),
        "7": ("Batch Processing", batch_processing_example),
        "8": ("CLI Interface", command_line_interface_example)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nEnter example number to run (or 'all' to run all non-interactive examples):")
    choice = input("> ").strip()
    
    if choice.lower() == 'all':
        # Run non-interactive examples
        for key in ["2", "3", "4", "5", "6", "7", "8"]:
            print(f"\n{'='*20} Running Example {key} {'='*20}")
            examples[key][1]()
    elif choice in examples:
        print(f"\n{'='*20} Running {examples[choice][0]} {'='*20}")
        examples[choice][1]()
    else:
        print("Invalid choice. Please select a valid example number.")


if __name__ == "__main__":
    main() 