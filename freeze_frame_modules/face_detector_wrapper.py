"""
Enhanced Face Detector Wrapper with improved error handling and detector compatibility
Compatible with DeepFaceLive's face detection system
"""

import numpy as np
from typing import Tuple, Any, List, Optional, Union
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FaceDetectorWrapper:
    def __init__(self, original_detector=None):
        """
        Wrapper for face detector to extract confidence

        Args:
            original_detector: Original face detector instance
        """
        self.detector = original_detector
        self.detector_type = self._identify_detector_type()
        self.last_confidence = 0.0
        self.detection_history = []
        self.history_size = 5
        self.fallback_confidence = 0.5  # Fallback when confidence can't be extracted

        logger.info(f"FaceDetectorWrapper initialized with detector type: {self.detector_type}")

    def _identify_detector_type(self) -> str:
        """Identify the type of face detector"""
        if self.detector is None:
            return "none"

        detector_class = self.detector.__class__.__name__.lower()

        if 'insight' in detector_class:
            return "insightface"
        elif 'mediapipe' in detector_class:
            return "mediapipe"
        elif 'opencv' in detector_class or 'dnn' in detector_class:
            return "opencv_dnn"
        elif 's3fd' in detector_class:
            return "s3fd"
        elif 'retinaface' in detector_class:
            return "retinaface"
        else:
            return "unknown"

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
            # Call original detector with error handling
            result = self._safe_detector_call(frame)

            # Extract confidence based on detector type and result
            confidence = self.extract_confidence(result, frame)
            self.last_confidence = confidence

            # Update detection history
            self.update_detection_history(confidence, result)

            return result, confidence

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None, 0.0

    def _safe_detector_call(self, frame: np.ndarray) -> Any:
        """Safely call the detector with multiple method attempts"""
        # Common method names for face detection
        method_names = ['extract', 'detect', 'forward', 'predict', '__call__']

        for method_name in method_names:
            if hasattr(self.detector, method_name):
                try:
                    method = getattr(self.detector, method_name)
                    result = method(frame)
                    logger.debug(f"Successfully called detector method: {method_name}")
                    return result
                except Exception as e:
                    logger.debug(f"Failed to call {method_name}: {e}")
                    continue

        # If no method worked, raise an error
        raise RuntimeError(f"Could not find compatible detection method for {self.detector_type}")

    def extract_confidence(self, detection_result: Any, frame: np.ndarray) -> float:
        """
        Extract confidence from detection result with enhanced compatibility

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

            # Type-specific confidence extraction
            confidence = self._extract_by_detector_type(detection_result)
            if confidence is not None:
                return float(np.clip(confidence, 0.0, 1.0))

            # Generic confidence extraction methods
            confidence = self._extract_generic_confidence(detection_result)
            if confidence is not None:
                return float(np.clip(confidence, 0.0, 1.0))

            # Fallback: estimate confidence from detection quality
            return self.estimate_confidence_from_detection(detection_result, frame)

        except Exception as e:
            logger.warning(f"Error extracting confidence: {e}")
            return self.fallback_confidence

    def _extract_by_detector_type(self, detection_result: Any) -> Optional[float]:
        """Extract confidence based on known detector types"""

        if self.detector_type == "insightface":
            return self._extract_insightface_confidence(detection_result)
        elif self.detector_type == "mediapipe":
            return self._extract_mediapipe_confidence(detection_result)
        elif self.detector_type == "opencv_dnn":
            return self._extract_opencv_confidence(detection_result)
        elif self.detector_type == "s3fd":
            return self._extract_s3fd_confidence(detection_result)
        elif self.detector_type == "retinaface":
            return self._extract_retinaface_confidence(detection_result)

        return None

    def _extract_insightface_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence from InsightFace results"""
        try:
            # InsightFace SCRFD detector results
            if hasattr(result, 'det_score') and result.det_score is not None:
                if len(result.det_score) > 0:
                    return float(np.max(result.det_score))

            # List of InsightFace face objects
            if isinstance(result, list) and len(result) > 0:
                confidences = []
                for face in result:
                    if hasattr(face, 'det_score'):
                        confidences.append(float(face.det_score))
                    elif hasattr(face, 'confidence'):
                        confidences.append(float(face.confidence))

                if confidences:
                    return max(confidences)

            # Single face object
            if hasattr(result, 'det_score'):
                return float(result.det_score)
            elif hasattr(result, 'confidence'):
                return float(result.confidence)

        except Exception as e:
            logger.debug(f"InsightFace confidence extraction failed: {e}")

        return None

    def _extract_mediapipe_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence from MediaPipe results"""
        try:
            if hasattr(result, 'detections') and result.detections:
                confidences = []
                for det in result.detections:
                    if hasattr(det, 'score') and len(det.score) > 0:
                        confidences.append(det.score[0])

                if confidences:
                    return max(confidences)

        except Exception as e:
            logger.debug(f"MediaPipe confidence extraction failed: {e}")

        return None

    def _extract_opencv_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence from OpenCV DNN results"""
        try:
            if isinstance(result, np.ndarray):
                if result.ndim >= 2 and result.shape[-1] >= 3:
                    # Assuming format [batch, detections, (x, y, confidence, ...)]
                    confidences = result[:, 2] if len(result.shape) > 1 else [result[2]]
                    valid_confidences = [c for c in confidences if c > 0]
                    if valid_confidences:
                        return max(valid_confidences)

        except Exception as e:
            logger.debug(f"OpenCV confidence extraction failed: {e}")

        return None

    def _extract_s3fd_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence from S3FD results"""
        try:
            # S3FD typically returns numpy arrays with confidence scores
            if isinstance(result, np.ndarray):
                if result.ndim >= 2 and result.shape[-1] >= 5:
                    # Format: [x1, y1, x2, y2, confidence, ...]
                    confidences = result[:, 4] if len(result.shape) > 1 else [result[4]]
                    valid_confidences = [c for c in confidences if c > 0]
                    if valid_confidences:
                        return max(valid_confidences)

            # List format
            elif isinstance(result, list) and len(result) > 0:
                confidences = []
                for detection in result:
                    if isinstance(detection, (list, np.ndarray)) and len(detection) >= 5:
                        confidences.append(detection[4])

                if confidences:
                    return max(confidences)

        except Exception as e:
            logger.debug(f"S3FD confidence extraction failed: {e}")

        return None

    def _extract_retinaface_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence from RetinaFace results"""
        try:
            # RetinaFace returns dict with face locations and scores
            if isinstance(result, dict):
                confidences = []
                for face_key in result:
                    if isinstance(result[face_key], dict) and 'score' in result[face_key]:
                        confidences.append(result[face_key]['score'])

                if confidences:
                    return max(confidences)

            # List of detections
            elif isinstance(result, list):
                confidences = []
                for detection in result:
                    if isinstance(detection, dict) and 'score' in detection:
                        confidences.append(detection['score'])
                    elif hasattr(detection, 'score'):
                        confidences.append(detection.score)

                if confidences:
                    return max(confidences)

        except Exception as e:
            logger.debug(f"RetinaFace confidence extraction failed: {e}")

        return None

    def _extract_generic_confidence(self, detection_result: Any) -> Optional[float]:
        """Generic confidence extraction for unknown detector types"""
        try:
            # Check for common attribute names
            confidence_attrs = ['confidence', 'score', 'det_score', 'prob', 'probability']

            for attr in confidence_attrs:
                if hasattr(detection_result, attr):
                    value = getattr(detection_result, attr)
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        return float(max(value))

            # Check if it's a dictionary
            if isinstance(detection_result, dict):
                for attr in confidence_attrs:
                    if attr in detection_result:
                        value = detection_result[attr]
                        if isinstance(value, (int, float)):
                            return float(value)
                        elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                            return float(max(value))

            # Check if it's a list of objects
            if isinstance(detection_result, list) and len(detection_result) > 0:
                confidences = []
                for item in detection_result:
                    for attr in confidence_attrs:
                        if hasattr(item, attr):
                            value = getattr(item, attr)
                            if isinstance(value, (int, float)):
                                confidences.append(float(value))
                                break
                        elif isinstance(item, dict) and attr in item:
                            value = item[attr]
                            if isinstance(value, (int, float)):
                                confidences.append(float(value))
                                break

                if confidences:
                    return max(confidences)

        except Exception as e:
            logger.debug(f"Generic confidence extraction failed: {e}")

        return None

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

            # For numpy arrays (check if valid detections)
            if isinstance(detection_result, np.ndarray):
                if detection_result.size == 0:
                    return 0.0
                elif detection_result.ndim >= 2:
                    # Multiple detections
                    return 0.75 if detection_result.shape[0] == 1 else 0.65
                else:
                    return 0.75

            # For single detection objects
            if hasattr(detection_result, 'face_urect') or hasattr(detection_result, 'bbox'):
                return 0.8  # Face with bounding box detected

            # For dictionaries
            if isinstance(detection_result, dict) and len(detection_result) > 0:
                return 0.75

            # Basic detection present
            return 0.7

        except Exception as e:
            logger.debug(f"Confidence estimation failed: {e}")
            return self.fallback_confidence

    def update_detection_history(self, confidence: float, result: Any):
        """Update detection history for stability analysis"""
        try:
            self.detection_history.append({
                'confidence': confidence,
                'has_detection': result is not None,
                'timestamp': time.time(),
                'result_type': type(result).__name__
            })

            # Keep history size limited
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)

        except Exception as e:
            logger.debug(f"Failed to update detection history: {e}")

    def get_stability_score(self) -> float:
        """Get detection stability score based on recent history"""
        if len(self.detection_history) < 2:
            return 1.0

        try:
            confidences = [h['confidence'] for h in self.detection_history]
            if not confidences:
                return 0.0

            # Calculate variance in confidence
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

            # Lower variance = higher stability
            stability = max(0.0, 1.0 - variance)
            return stability

        except Exception as e:
            logger.debug(f"Stability calculation failed: {e}")
            return 0.5

    def get_detection_stats(self) -> dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {
                'avg_confidence': 0.0,
                'stability_score': 0.0,
                'detection_rate': 0.0,
                'detector_type': self.detector_type
            }

        try:
            confidences = [h['confidence'] for h in self.detection_history]
            detections = [h['has_detection'] for h in self.detection_history]

            return {
                'avg_confidence': sum(confidences) / len(confidences),
                'stability_score': self.get_stability_score(),
                'detection_rate': sum(detections) / len(detections),
                'last_confidence': self.last_confidence,
                'detector_type': self.detector_type,
                'history_length': len(self.detection_history)
            }

        except Exception as e:
            logger.error(f"Failed to get detection stats: {e}")
            return {
                'avg_confidence': 0.0,
                'stability_score': 0.0,
                'detection_rate': 0.0,
                'detector_type': self.detector_type,
                'error': str(e)
            }


class DeepFaceLiveDetectorAdapter:
    """
    Enhanced adapter specifically for DeepFaceLive's face detection system
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
        self.wrapper = FaceDetectorWrapper(face_detector)

        # DeepFaceLive specific settings
        self.deepfacelive_confidence_mapping = {
            'high_quality': 0.9,
            'medium_quality': 0.75,
            'low_quality': 0.5,
            'very_low_quality': 0.3
        }

        logger.info("DeepFaceLive detector adapter initialized")

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
            faces = self._safe_extract_faces(frame)

            if faces is None or (isinstance(faces, list) and len(faces) == 0):
                return None, 0.0

            # Extract confidence from faces
            confidence = self.extract_deepfacelive_confidence(faces)

            return faces, confidence

        except Exception as e:
            logger.error(f"Error in DeepFaceLive face extraction: {e}")
            return None, 0.0

    def _safe_extract_faces(self, frame: np.ndarray) -> Any:
        """Safely extract faces with error handling"""
        try:
            # Try multiple extraction methods
            extraction_methods = ['extract', 'detect', 'process']

            for method_name in extraction_methods:
                if hasattr(self.face_detector, method_name):
                    try:
                        method = getattr(self.face_detector, method_name)
                        result = method(frame)
                        logger.debug(f"Successfully used {method_name} for face extraction")
                        return result
                    except Exception as e:
                        logger.debug(f"Method {method_name} failed: {e}")
                        continue

            # If no method worked
            raise RuntimeError("Could not find compatible face extraction method")

        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None

    def extract_deepfacelive_confidence(self, faces) -> float:
        """
        Extract confidence from DeepFaceLive face objects with enhanced handling

        Args:
            faces: Face detection results from DeepFaceLive

        Returns:
            float: Confidence score
        """
        if not faces:
            return 0.0

        try:
            # Ensure faces is a list
            if not isinstance(faces, list):
                faces = [faces]

            confidences = []

            for face in faces:
                confidence = self._extract_single_face_confidence(face)
                if confidence > 0:
                    confidences.append(confidence)

            if confidences:
                # Return the highest confidence if multiple faces
                return max(confidences)
            else:
                # Estimate confidence based on face quality
                return self._estimate_face_quality_confidence(faces)

        except Exception as e:
            logger.error(f"Error extracting DeepFaceLive confidence: {e}")
            return 0.5  # Neutral confidence if error

    def _extract_single_face_confidence(self, face) -> float:
        """Extract confidence from a single face object"""
        try:
            # Check for various confidence attributes
            confidence_attrs = ['det_score', 'confidence', 'score', 'quality', 'detection_confidence']

            for attr in confidence_attrs:
                if hasattr(face, attr):
                    value = getattr(face, attr)
                    if isinstance(value, (int, float)) and value > 0:
                        return float(value)

            # Check dictionary-style access
            if hasattr(face, '__getitem__'):
                for attr in confidence_attrs:
                    try:
                        value = face[attr]
                        if isinstance(value, (int, float)) and value > 0:
                            return float(value)
                    except (KeyError, TypeError):
                        continue

            return 0.0

        except Exception as e:
            logger.debug(f"Single face confidence extraction failed: {e}")
            return 0.0

    def _estimate_face_quality_confidence(self, faces) -> float:
        """Estimate confidence based on face object quality indicators"""
        try:
            total_quality = 0.0
            quality_indicators = 0

            for face in faces:
                # Check for face alignment quality
                if hasattr(face, 'face_align_img') and face.face_align_img is not None:
                    total_quality += 0.85
                    quality_indicators += 1

                # Check for face rectangle
                elif hasattr(face, 'face_urect') and face.face_urect is not None:
                    total_quality += 0.75
                    quality_indicators += 1

                # Check for landmarks
                elif hasattr(face, 'landmarks') and face.landmarks is not None:
                    total_quality += 0.8
                    quality_indicators += 1

                # Basic detection
                else:
                    total_quality += 0.6
                    quality_indicators += 1

            if quality_indicators > 0:
                return total_quality / quality_indicators
            else:
                return 0.5

        except Exception as e:
            logger.debug(f"Quality estimation failed: {e}")
            return 0.5

    def get_adapter_stats(self) -> dict:
        """Get comprehensive adapter statistics"""
        try:
            wrapper_stats = self.wrapper.get_detection_stats()

            adapter_stats = {
                'adapter_type': 'DeepFaceLive',
                'has_face_detector': self.face_detector is not None,
                'has_marker_detector': self.face_marker_detector is not None,
                'detector_class': self.face_detector.__class__.__name__ if self.face_detector else 'None'
            }

            return {**wrapper_stats, **adapter_stats}

        except Exception as e:
            logger.error(f"Failed to get adapter stats: {e}")
            return {'error': str(e)}


# Time import for timestamp functionality
import time