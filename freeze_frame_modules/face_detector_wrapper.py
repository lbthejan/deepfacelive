"""
Face Detector Wrapper to extract confidence scores
Compatible with DeepFaceLive's face detection system
"""

import numpy as np
from typing import Tuple, Any, List, Optional


class FaceDetectorWrapper:
    def __init__(self, original_detector=None):
        """
        Wrapper for face detector to extract confidence
        
        Args:
            original_detector: Original face detector instance
        """
        self.detector = original_detector
        self.last_confidence = 0.0
        self.detection_history = []
        self.history_size = 5
    
    def detect(self, frame: np.ndarray) -> Tuple[Any, float]:
        """
        Detect faces and extract confidence
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple[Any, float]: (detection_result, confidence)
        """
        if self.detector is None:
            return None, 0.0
        
        try:
            # Call original detector
            result = self.detector.extract(frame) if hasattr(self.detector, 'extract') else self.detector.detect(frame)
            
            # Extract confidence based on detector type and result
            confidence = self.extract_confidence(result, frame)
            self.last_confidence = confidence
            
            # Update detection history
            self.update_detection_history(confidence, result)
            
            return result, confidence
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None, 0.0
    
    def extract_confidence(self, detection_result: Any, frame: np.ndarray) -> float:
        """
        Extract confidence from detection result
        
        Args:
            detection_result: Result from face detector
            frame: Input frame
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        try:
            # Handle None or empty results
            if detection_result is None:
                return 0.0
            
            # For InsightFace SCRFD detector results
            if hasattr(detection_result, 'det_score') and detection_result.det_score is not None:
                if len(detection_result.det_score) > 0:
                    return float(np.max(detection_result.det_score))
            
            # For InsightFace with list of faces
            if isinstance(detection_result, list) and len(detection_result) > 0:
                confidences = []
                for face in detection_result:
                    if hasattr(face, 'det_score'):
                        confidences.append(float(face.det_score))
                    elif hasattr(face, 'confidence'):
                        confidences.append(float(face.confidence))
                    elif isinstance(face, dict):
                        if 'det_score' in face:
                            confidences.append(float(face['det_score']))
                        elif 'confidence' in face:
                            confidences.append(float(face['confidence']))
                
                if confidences:
                    return max(confidences)  # Use highest confidence face
            
            # For MediaPipe results
            if hasattr(detection_result, 'detections') and detection_result.detections:
                confidences = [det.score[0] for det in detection_result.detections if hasattr(det, 'score')]
                if confidences:
                    return max(confidences)
            
            # For OpenCV DNN results
            if isinstance(detection_result, np.ndarray):
                if detection_result.shape[-1] >= 3:  # Assuming [x, y, confidence, ...]
                    confidences = detection_result[:, 2] if len(detection_result.shape) > 1 else [detection_result[2]]
                    valid_confidences = [c for c in confidences if c > 0]
                    if valid_confidences:
                        return max(valid_confidences)
            
            # For DeepFaceLive specific face objects
            if hasattr(detection_result, 'face_urect') and hasattr(detection_result, 'face_align_img'):
                # If we have aligned face, assume good detection
                return 0.8
            
            # Fallback: estimate confidence from detection quality
            return self.estimate_confidence_from_detection(detection_result, frame)
            
        except Exception as e:
            print(f"Error extracting confidence: {e}")
            return 0.0
    
    def estimate_confidence_from_detection(self, detection_result: Any, frame: np.ndarray) -> float:
        """
        Estimate confidence when not directly available
        
        Args:
            detection_result: Detection result
            frame: Input frame
            
        Returns:
            float: Estimated confidence
        """
        try:
            # Check if any face was detected
            if detection_result is None:
                return 0.0
            
            # For list of detections
            if isinstance(detection_result, list):
                if len(detection_result) == 0:
                    return 0.0
                elif len(detection_result) == 1:
                    return 0.85  # Single face detected - good
                else:
                    return 0.65  # Multiple faces - less certain which is target
            
            # For single detection object
            if hasattr(detection_result, 'face_urect') or hasattr(detection_result, 'bbox'):
                return 0.8  # Face with bounding box detected
            
            # For numpy arrays (coordinates/features)
            if isinstance(detection_result, np.ndarray) and detection_result.size > 0:
                return 0.75
            
            # Basic detection present
            return 0.7
            
        except:
            return 0.0
    
    def update_detection_history(self, confidence: float, result: Any):
        """Update detection history for stability analysis"""
        self.detection_history.append({
            'confidence': confidence,
            'has_detection': result is not None,
            'timestamp': time.time() if 'time' in globals() else 0
        })
        
        # Keep history size limited
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
    
    def get_stability_score(self) -> float:
        """Get detection stability score based on recent history"""
        if len(self.detection_history) < 2:
            return 1.0
        
        confidences = [h['confidence'] for h in self.detection_history]
        if not confidences:
            return 0.0
        
        # Calculate variance in confidence
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        
        # Lower variance = higher stability
        stability = max(0.0, 1.0 - variance)
        return stability
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {
                'avg_confidence': 0.0,
                'stability_score': 0.0,
                'detection_rate': 0.0
            }
        
        confidences = [h['confidence'] for h in self.detection_history]
        detections = [h['has_detection'] for h in self.detection_history]
        
        return {
            'avg_confidence': sum(confidences) / len(confidences),
            'stability_score': self.get_stability_score(),
            'detection_rate': sum(detections) / len(detections),
            'last_confidence': self.last_confidence
        }


class DeepFaceLiveDetectorAdapter:
    """
    Adapter specifically for DeepFaceLive's face detection system
    """
    
    def __init__(self, face_detector, face_marker_detector=None):
        """
        Initialize adapter for DeepFaceLive detectors
        
        Args:
            face_detector: Face detector from DeepFaceLive
            face_marker_detector: Face marker detector (optional)
        """
        self.face_detector = face_detector
        self.face_marker_detector = face_marker_detector
        self.wrapper = FaceDetectorWrapper()
    
    def extract_faces(self, frame: np.ndarray) -> Tuple[Any, float]:
        """
        Extract faces using DeepFaceLive's detection system
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple[Any, float]: (face_result, confidence)
        """
        try:
            # Use DeepFaceLive's face detector
            faces = self.face_detector.extract(frame)
            
            if faces is None or len(faces) == 0:
                return None, 0.0
            
            # Extract confidence from faces
            confidence = self.extract_deepfacelive_confidence(faces)
            
            return faces, confidence
            
        except Exception as e:
            print(f"Error in DeepFaceLive face extraction: {e}")
            return None, 0.0
    
    def extract_deepfacelive_confidence(self, faces) -> float:
        """
        Extract confidence from DeepFaceLive face objects
        
        Args:
            faces: Face detection results from DeepFaceLive
            
        Returns:
            float: Confidence score
        """
        if not faces:
            return 0.0
        
        try:
            # For single face
            if not isinstance(faces, list):
                faces = [faces]
            
            confidences = []
            
            for face in faces:
                # Check for various confidence attributes
                if hasattr(face, 'det_score'):
                    confidences.append(float(face.det_score))
                elif hasattr(face, 'confidence'):
                    confidences.append(float(face.confidence))
                elif hasattr(face, 'score'):
                    confidences.append(float(face.score))
                else:
                    # Estimate based on face quality
                    if hasattr(face, 'face_align_img') and face.face_align_img is not None:
                        # Good aligned face
                        confidences.append(0.85)
                    elif hasattr(face, 'face_urect'):
                        # Face with rectangle
                        confidences.append(0.75)
                    else:
                        # Basic detection
                        confidences.append(0.6)
            
            return max(confidences) if confidences else 0.0
            
        except Exception as e:
            print(f"Error extracting DeepFaceLive confidence: {e}")
            return 0.5  # Neutral confidence if error


# Time import for timestamp functionality
import time