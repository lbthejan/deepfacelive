"""
DeepFaceLive Integration Module
Main integration point for freeze-frame functionality with DeepFaceLive
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Any
import threading
from pathlib import Path

# Import our freeze-frame modules
from freeze_frame_manager import FreezeFrameManager
from face_detector_wrapper import FaceDetectorWrapper, DeepFaceLiveDetectorAdapter
from performance_monitor import PerformanceMonitor


class DeepFaceLiveFreezeProcessor:
    """
    Main processor that integrates freeze-frame functionality into DeepFaceLive
    """
    
    def __init__(self, userdata_path: Path = None):
        """
        Initialize the freeze-frame processor
        
        Args:
            userdata_path: Path to user data directory for config storage
        """
        # Initialize components
        config_path = str(userdata_path / "freeze_config.json") if userdata_path else "freeze_config.json"
        self.freeze_manager = FreezeFrameManager(config_file=config_path)
        self.performance_monitor = PerformanceMonitor()
        self.face_detector_adapter = None
        
        # State management
        self.is_running = False
        self.show_stats_overlay = True
        self.show_debug_info = False
        self.last_process_time = time.time()
        
        # Threading for UI updates
        self.stats_lock = threading.Lock()
        
        print("DeepFaceLive Freeze-Frame Processor initialized")
    
    def initialize_face_detector(self, face_detector, face_marker_detector=None):
        """
        Initialize face detector adapter
        
        Args:
            face_detector: DeepFaceLive face detector instance
            face_marker_detector: Optional face marker detector
        """
        self.face_detector_adapter = DeepFaceLiveDetectorAdapter(
            face_detector, 
            face_marker_detector
        )
        print("Face detector adapter initialized")
    
    def process_frame(self, input_frame: np.ndarray, face_swapper, predictor, face_enhancer=None) -> np.ndarray:
        """
        Main frame processing function with freeze-frame logic
        
        Args:
            input_frame: Raw input frame from webcam
            face_swapper: Face swapper instance
            predictor: Face predictor instance
            face_enhancer: Optional face enhancer
            
        Returns:
            np.ndarray: Processed output frame (live or frozen)
        """
        start_time = time.time()
        
        try:
            # Extract faces and get confidence
            faces, confidence = self._extract_faces_with_confidence(input_frame)
            
            # Process frame if faces detected
            if faces and confidence > 0.1:  # Minimum threshold for processing
                swapped_frame = self._perform_face_swap(
                    input_frame, faces, face_swapper, predictor, face_enhancer
                )
            else:
                # No faces detected or confidence too low
                swapped_frame = input_frame.copy()
                confidence = 0.0
            
            # Apply freeze-frame logic
            output_frame, is_frozen = self.freeze_manager.process_frame(
                swapped_frame, confidence
            )
            
            # Update performance monitoring
            processing_time = time.time() - start_time
            self.performance_monitor.update(processing_time, confidence, is_frozen)
            
            # Add overlays if enabled
            if self.show_stats_overlay or self.show_debug_info:
                output_frame = self._add_overlay_info(output_frame, confidence, is_frozen, processing_time)
            
            self.last_process_time = time.time()
            return output_frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Return last good frame or original frame on error
            if self.freeze_manager.last_good_frame is not None:
                return self.freeze_manager.last_good_frame.copy()
            else:
                return input_frame.copy()
    
    def _extract_faces_with_confidence(self, frame: np.ndarray) -> Tuple[Any, float]:
        """
        Extract faces and confidence from frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple[Any, float]: (faces, confidence)
        """
        if self.face_detector_adapter is None:
            # Fallback: assume no detection
            return None, 0.0
        
        try:
            faces, confidence = self.face_detector_adapter.extract_faces(frame)
            return faces, confidence
        except Exception as e:
            print(f"Error extracting faces: {e}")
            return None, 0.0
    
    def _perform_face_swap(self, input_frame: np.ndarray, faces, face_swapper, predictor, face_enhancer=None) -> np.ndarray:
        """
        Perform face swapping on detected faces
        
        Args:
            input_frame: Original input frame
            faces: Detected faces
            face_swapper: Face swapper instance
            predictor: Face predictor instance
            face_enhancer: Optional face enhancer
            
        Returns:
            np.ndarray: Frame with swapped faces
        """
        try:
            # This will depend on DeepFaceLive's specific API
            # The exact implementation will vary based on the version
            
            # For newer versions, face_swapper might have a process method
            if hasattr(face_swapper, 'process'):
                result = face_swapper.process(input_frame, faces)
                if result is not None:
                    swapped_frame = result
                else:
                    swapped_frame = input_frame.copy()
            
            # Alternative: manual face swapping process
            else:
                swapped_frame = input_frame.copy()
                
                # Process each detected face
                for face in (faces if isinstance(faces, list) else [faces]):
                    # Get face alignment and features
                    if hasattr(face, 'face_align_img') and face.face_align_img is not None:
                        # Use aligned face for swapping
                        if hasattr(predictor, 'predict') and callable(predictor.predict):
                            # Predict swapped face
                            predicted = predictor.predict(face.face_align_img)
                            
                            # Merge back to original frame
                            if predicted is not None and hasattr(face, 'face_urect'):
                                swapped_frame = self._merge_face_to_frame(
                                    swapped_frame, predicted, face.face_urect
                                )
            
            # Apply face enhancement if available
            if face_enhancer is not None and hasattr(face_enhancer, 'enhance'):
                enhanced_frame = face_enhancer.enhance(swapped_frame)
                if enhanced_frame is not None:
                    swapped_frame = enhanced_frame
            
            return swapped_frame
            
        except Exception as e:
            print(f"Error in face swapping: {e}")
            return input_frame.copy()
    
    def _merge_face_to_frame(self, frame: np.ndarray, face_img: np.ndarray, face_urect) -> np.ndarray:
        """
        Merge swapped face back to original frame
        
        Args:
            frame: Original frame
            face_img: Swapped face image
            face_urect: Face rectangle information
            
        Returns:
            np.ndarray: Frame with merged face
        """
        try:
            # This is a simplified merge - actual implementation depends on DeepFaceLive's method
            if hasattr(face_urect, 'get_coords'):
                coords = face_urect.get_coords()
                x1, y1, x2, y2 = coords
                
                # Resize face image to fit rectangle
                face_resized = cv2.resize(face_img, (x2-x1, y2-y1))
                
                # Simple overlay (in practice, you'd use more sophisticated blending)
                frame[y1:y2, x1:x2] = face_resized
            
            return frame
            
        except Exception as e:
            print(f"Error merging face: {e}")
            return frame
    
    def _add_overlay_info(self, frame: np.ndarray, confidence: float, is_frozen: bool, processing_time: float) -> np.ndarray:
        """
        Add information overlay to frame
        
        Args:
            frame: Input frame
            confidence: Current confidence
            is_frozen: Whether frame is frozen
            processing_time: Time taken to process frame
            
        Returns:
            np.ndarray: Frame with overlay
        """
        overlay_frame = frame.copy()
        
        if self.show_stats_overlay:
            # Status indicator
            status = "FROZEN" if is_frozen else "LIVE"
            status_color = (0, 0, 255) if is_frozen else (0, 255, 0)
            
            # Background for text
            cv2.rectangle(overlay_frame, (10, 10), (250, 100), (0, 0, 0, 128), -1)
            
            # Status text
            cv2.putText(overlay_frame, status, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Confidence
            cv2.putText(overlay_frame, f"Conf: {confidence:.3f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS
            fps = self.performance_monitor.get_current_fps()
            cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.show_debug_info:
            # Additional debug information
            stats = self.freeze_manager.get_stats()
            debug_y = 120
            
            debug_info = [
                f"Threshold: {stats['threshold']:.2f}",
                f"Freeze Count: {stats['freeze_count']}",
                f"Freeze Time: {stats['total_freeze_time']:.1f}s",
                f"Process Time: {processing_time*1000:.1f}ms"
            ]
            
            for i, info in enumerate(debug_info):
                cv2.putText(overlay_frame, info, (20, debug_y + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay_frame
    
    def update_freeze_threshold(self, threshold: float):
        """Update freeze threshold"""
        self.freeze_manager.update_threshold(threshold)
        print(f"Freeze threshold updated to: {threshold:.3f}")
    
    def toggle_stats_overlay(self):
        """Toggle statistics overlay"""
        self.show_stats_overlay = not self.show_stats_overlay
        print(f"Stats overlay: {'ON' if self.show_stats_overlay else 'OFF'}")
    
    def toggle_debug_info(self):
        """Toggle debug information"""
        self.show_debug_info = not self.show_debug_info
        print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics"""
        with self.stats_lock:
            freeze_stats = self.freeze_manager.get_stats()
            performance_stats = self.performance_monitor.get_comprehensive_stats()
            
            return {
                'freeze': freeze_stats,
                'performance': performance_stats,
                'detector': self.face_detector_adapter.wrapper.get_detection_stats() if self.face_detector_adapter else {}
            }
    
    def print_stats_summary(self):
        """Print performance summary"""
        self.performance_monitor.print_stats_summary()
        
        # Print freeze-specific stats
        freeze_stats = self.freeze_manager.get_stats()
        print(f"\n🧊 FREEZE-FRAME STATISTICS:")
        print(f"   Current Status: {'FROZEN' if freeze_stats['is_frozen'] else 'LIVE'}")
        print(f"   Confidence Threshold: {freeze_stats['threshold']:.3f}")
        print(f"   Current Confidence: {freeze_stats['current_confidence']:.3f}")
        print(f"   Total Freezes: {freeze_stats['freeze_count']}")
        print(f"   Total Freeze Time: {freeze_stats['total_freeze_time']:.1f}s")
    
    def reset_stats(self):
        """Reset all statistics"""
        self.freeze_manager.reset()
        self.performance_monitor.reset()
        print("Statistics reset")
    
    def start_processing(self):
        """Start processing mode"""
        self.is_running = True
        print("Freeze-frame processing started")
    
    def stop_processing(self):
        """Stop processing mode"""
        self.is_running = False
        print("Freeze-frame processing stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_processing()
        self.freeze_manager.save_config()
        print("Freeze-frame processor cleaned up")


class DeepFaceLiveAppIntegration:
    """
    Integration helper for DeepFaceLive application
    This class helps integrate the freeze-frame processor into the main app
    """
    
    @staticmethod
    def integrate_into_app(app_instance, userdata_path: Path = None):
        """
        Integrate freeze-frame processor into DeepFaceLive app
        
        Args:
            app_instance: DeepFaceLive app instance
            userdata_path: Path to user data directory
            
        Returns:
            DeepFaceLiveFreezeProcessor: Integrated processor
        """
        # Create freeze processor
        freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)
        
        # Store reference in app
        app_instance.freeze_processor = freeze_processor
        
        # Hook into face detector if available
        if hasattr(app_instance, 'face_detector'):
            freeze_processor.initialize_face_detector(app_instance.face_detector)
        
        # Add UI controls if app has UI
        if hasattr(app_instance, 'add_control'):
            DeepFaceLiveAppIntegration._add_ui_controls(app_instance, freeze_processor)
        
        print("Freeze-frame processor integrated into DeepFaceLive app")
        return freeze_processor
    
    @staticmethod
    def _add_ui_controls(app_instance, freeze_processor):
        """
        Add UI controls for freeze-frame feature
        
        Args:
            app_instance: DeepFaceLive app instance
            freeze_processor: Freeze processor instance
        """
        try:
            # Add threshold slider
            app_instance.add_control(
                'freeze_threshold',
                'slider',
                min_val=0.3,
                max_val=0.95,
                initial_val=0.75,
                callback=freeze_processor.update_freeze_threshold,
                label="Freeze Threshold"
            )
            
            # Add stats toggle button
            app_instance.add_control(
                'stats_toggle',
                'button',
                callback=freeze_processor.toggle_stats_overlay,
                label="Toggle Stats"
            )
            
            # Add debug toggle button
            app_instance.add_control(
                'debug_toggle',
                'button',
                callback=freeze_processor.toggle_debug_info,
                label="Toggle Debug"
            )
            
            # Add reset button
            app_instance.add_control(
                'reset_stats',
                'button',
                callback=freeze_processor.reset_stats,
                label="Reset Stats"
            )
            
        except Exception as e:
            print(f"Could not add UI controls: {e}")
            print("UI controls will need to be added manually")


# Example usage and testing functions
def test_freeze_processor():
    """Test the freeze processor with webcam"""
    import cv2
    
    processor = DeepFaceLiveFreezeProcessor()
    cap = cv2.VideoCapture(0)
    
    print("Testing freeze processor (press 'q' to quit, 's' for stats, 'd' for debug)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate face detection confidence for testing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        mock_confidence = max(0.0, min(1.0, brightness))
        
        # Process frame (without actual face swapping for test)
        output_frame, is_frozen = processor.freeze_manager.process_frame(frame, mock_confidence)
        
        # Add test overlay
        status = "FROZEN" if is_frozen else "LIVE"
        color = (0, 0, 255) if is_frozen else (0, 255, 0)
        cv2.putText(output_frame, f"{status} - Conf: {mock_confidence:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Freeze Processor Test", output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            processor.print_stats_summary()
        elif key == ord('d'):
            processor.toggle_debug_info()
    
    cap.release()
    cv2.destroyAllWindows()
    processor.cleanup()


if __name__ == "__main__":
    test_freeze_processor()