"""
Freeze Frame Manager for DeepFaceLive
Manages frame freezing based on face detection confidence
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
import json
import os


class FreezeFrameManager:
    def __init__(self, confidence_threshold: float = 0.75, config_file: str = "freeze_config.json"):
        """
        Initialize the Freeze Frame Manager

        Args:
            confidence_threshold: Minimum confidence to continue live processing
            config_file: Path to configuration file
        """
        self.confidence_threshold = confidence_threshold
        self.config_file = config_file
        self.last_good_frame = None
        self.last_good_confidence = 0.0
        self.is_frozen = False
        self.freeze_start_time = None
        self.max_freeze_duration = 5.0  # Maximum freeze time in seconds
        self.frame_buffer_size = 3  # Number of frames to average confidence
        self.confidence_buffer = []

        # Load configuration if exists
        self.load_config()

        # Performance metrics
        self.freeze_count = 0
        self.total_freeze_time = 0.0

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
                    self.max_freeze_duration = config.get('max_freeze_duration', self.max_freeze_duration)
                    self.frame_buffer_size = config.get('frame_buffer_size', self.frame_buffer_size)
                    print(f"Loaded freeze-frame config: threshold={self.confidence_threshold}")
            except Exception as e:
                print(f"Error loading freeze config: {e}")

    def save_config(self):
        """Save current configuration to file"""
        config = {
            'confidence_threshold': self.confidence_threshold,
            'max_freeze_duration': self.max_freeze_duration,
            'frame_buffer_size': self.frame_buffer_size
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving freeze config: {e}")

    def update_confidence_buffer(self, confidence: float):
        """Update confidence buffer with smoothing"""
        self.confidence_buffer.append(confidence)
        if len(self.confidence_buffer) > self.frame_buffer_size:
            self.confidence_buffer.pop(0)

    def get_smoothed_confidence(self) -> float:
        """Get smoothed confidence from buffer"""
        if not self.confidence_buffer:
            return 0.0
        return sum(self.confidence_buffer) / len(self.confidence_buffer)

    def should_freeze(self, confidence: float) -> bool:
        """
        Determine if frame should be frozen based on confidence

        Args:
            confidence: Current face detection confidence (0.0-1.0)

        Returns:
            bool: True if frame should be frozen
        """
        self.update_confidence_buffer(confidence)
        smoothed_confidence = self.get_smoothed_confidence()

        # Check if we should start freezing
        if not self.is_frozen and smoothed_confidence < self.confidence_threshold:
            return True

        # Check if we should continue freezing
        if self.is_frozen:
            # Unfreeze if confidence is good or max freeze time exceeded
            if smoothed_confidence >= self.confidence_threshold + 0.05:  # Hysteresis
                return False
            elif self.freeze_start_time and (time.time() - self.freeze_start_time) > self.max_freeze_duration:
                return False
            else:
                return True

        return False

    def process_frame(self, current_frame: np.ndarray, confidence: float) -> Tuple[np.ndarray, bool]:
        """
        Process frame with freeze logic

        Args:
            current_frame: Current processed frame from face swap
            confidence: Face detection confidence

        Returns:
            Tuple[np.ndarray, bool]: (output_frame, is_frozen)
        """
        should_freeze = self.should_freeze(confidence)

        if should_freeze:
            if not self.is_frozen:
                # Start freezing
                self.is_frozen = True
                self.freeze_start_time = time.time()
                self.freeze_count += 1
                print(f"Frame frozen - confidence: {confidence:.3f} < {self.confidence_threshold}")

            # Return last good frame if available
            if self.last_good_frame is not None:
                return self.last_good_frame.copy(), True
            else:
                # No good frame stored yet, return current frame
                return current_frame, True
        else:
            if self.is_frozen:
                # Stop freezing
                freeze_duration = time.time() - self.freeze_start_time if self.freeze_start_time else 0
                self.total_freeze_time += freeze_duration
                self.is_frozen = False
                self.freeze_start_time = None
                print(f"Frame unfrozen - confidence: {confidence:.3f} >= {self.confidence_threshold}")

            # Store as last good frame and return current frame
            self.last_good_frame = current_frame.copy()
            self.last_good_confidence = confidence
            return current_frame, False

    def get_stats(self) -> dict:
        """Get freeze statistics"""
        return {
            'freeze_count': self.freeze_count,
            'total_freeze_time': self.total_freeze_time,
            'is_frozen': self.is_frozen,
            'current_confidence': self.get_smoothed_confidence(),
            'threshold': self.confidence_threshold,
            'last_good_confidence': self.last_good_confidence
        }

    def update_threshold(self, new_threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        self.save_config()

    def reset(self):
        """Reset the freeze manager state"""
        self.last_good_frame = None
        self.is_frozen = False
        self.freeze_start_time = None
        self.confidence_buffer.clear()
        self.freeze_count = 0
        self.total_freeze_time = 0.0