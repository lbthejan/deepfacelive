"""
Performance Monitor for DeepFaceLive Freeze-Frame Feature
Tracks FPS, confidence, and freeze statistics
"""

import time
import numpy as np
from typing import Dict, List
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of frames to track for rolling averages
        """
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.freeze_states = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)
        
        # Cumulative statistics
        self.total_frames = 0
        self.total_freeze_frames = 0
        self.total_processing_time = 0.0
        
        # Session start time
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        
        # FPS calculation
        self.fps_calculation_interval = 1.0  # Update FPS every second
        self.last_fps_calculation = time.time()
        self.frames_since_fps_calc = 0
        self.current_fps = 0.0
        
    def update(self, processing_time: float, confidence: float, is_frozen: bool):
        """
        Update performance metrics
        
        Args:
            processing_time: Time taken to process current frame (seconds)
            confidence: Face detection confidence (0.0-1.0)
            is_frozen: Whether frame is currently frozen
        """
        current_time = time.time()
        
        # Update rolling window data
        self.processing_times.append(processing_time)
        self.confidences.append(confidence)
        self.freeze_states.append(is_frozen)
        self.frame_timestamps.append(current_time)
        
        # Update cumulative statistics
        self.total_frames += 1
        if is_frozen:
            self.total_freeze_frames += 1
        self.total_processing_time += processing_time
        
        # Update FPS calculation
        self.frames_since_fps_calc += 1
        if current_time - self.last_fps_calculation >= self.fps_calculation_interval:
            self.current_fps = self.frames_since_fps_calc / (current_time - self.last_fps_calculation)
            self.last_fps_calculation = current_time
            self.frames_since_fps_calc = 0
        
        self.last_update_time = current_time
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_rolling_stats(self) -> Dict:
        """Get rolling window statistics"""
        if not self.processing_times:
            return {
                'fps': 0.0,
                'avg_confidence': 0.0,
                'freeze_ratio': 0.0,
                'avg_processing_time': 0.0,
                'confidence_stability': 0.0
            }
        
        # Calculate rolling averages
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        avg_confidence = sum(self.confidences) / len(self.confidences)
        freeze_ratio = sum(self.freeze_states) / len(self.freeze_states)
        
        # Calculate FPS from processing times
        if avg_processing_time > 0:
            theoretical_fps = 1.0 / avg_processing_time
        else:
            theoretical_fps = 0.0
        
        # Calculate confidence stability (lower variance = higher stability)
        if len(self.confidences) > 1:
            confidence_variance = np.var(list(self.confidences))
            confidence_stability = max(0.0, 1.0 - confidence_variance)
        else:
            confidence_stability = 1.0
        
        return {
            'fps': self.current_fps,
            'theoretical_fps': theoretical_fps,
            'avg_confidence': avg_confidence,
            'freeze_ratio': freeze_ratio,
            'avg_processing_time': avg_processing_time,
            'confidence_stability': confidence_stability
        }
    
    def get_session_stats(self) -> Dict:
        """Get overall session statistics"""
        session_duration = time.time() - self.session_start_time
        
        if self.total_frames == 0:
            return {
                'session_duration': session_duration,
                'total_frames': 0,
                'session_fps': 0.0,
                'total_freeze_ratio': 0.0,
                'avg_processing_time': 0.0
            }
        
        session_fps = self.total_frames / session_duration if session_duration > 0 else 0.0
        total_freeze_ratio = self.total_freeze_frames / self.total_frames
        avg_processing_time = self.total_processing_time / self.total_frames
        
        return {
            'session_duration': session_duration,
            'total_frames': self.total_frames,
            'total_freeze_frames': self.total_freeze_frames,
            'session_fps': session_fps,
            'total_freeze_ratio': total_freeze_ratio,
            'avg_processing_time': avg_processing_time
        }
    
    def get_comprehensive_stats(self) -> Dict:
        """Get all statistics combined"""
        rolling_stats = self.get_rolling_stats()
        session_stats = self.get_session_stats()
        
        return {
            'current': rolling_stats,
            'session': session_stats,
            'last_update': self.last_update_time,
            'window_size': len(self.processing_times)
        }
    
    def get_performance_health(self) -> Dict:
        """
        Assess performance health and provide recommendations
        
        Returns:
            Dict with health status and recommendations
        """
        stats = self.get_rolling_stats()
        
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check FPS
        if stats['fps'] < 15:
            health_score -= 30
            issues.append("Low FPS")
            recommendations.append("Consider reducing video quality or closing other applications")
        elif stats['fps'] < 24:
            health_score -= 15
            issues.append("Moderate FPS")
            recommendations.append("Monitor system resources")
        
        # Check freeze ratio
        if stats['freeze_ratio'] > 0.3:
            health_score -= 25
            issues.append("High freeze ratio")
            recommendations.append("Lower confidence threshold or improve lighting")
        elif stats['freeze_ratio'] > 0.1:
            health_score -= 10
            issues.append("Moderate freeze ratio")
            recommendations.append("Check face detection stability")
        
        # Check confidence stability
        if stats['confidence_stability'] < 0.7:
            health_score -= 20
            issues.append("Unstable confidence")
            recommendations.append("Improve lighting or camera stability")
        
        # Check processing time
        if stats['avg_processing_time'] > 0.1:  # >100ms per frame
            health_score -= 15
            issues.append("High processing time")
            recommendations.append("Check system performance or optimize settings")
        
        # Determine health status
        if health_score >= 90:
            status = "Excellent"
        elif health_score >= 75:
            status = "Good"
        elif health_score >= 60:
            status = "Fair"
        elif health_score >= 40:
            status = "Poor"
        else:
            status = "Critical"
        
        return {
            'health_score': max(0, health_score),
            'status': status,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def reset(self):
        """Reset all statistics"""
        self.processing_times.clear()
        self.confidences.clear()
        self.freeze_states.clear()
        self.frame_timestamps.clear()
        
        self.total_frames = 0
        self.total_freeze_frames = 0
        self.total_processing_time = 0.0
        
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.last_fps_calculation = time.time()
        self.frames_since_fps_calc = 0
        self.current_fps = 0.0
    
    def export_stats(self, filename: str = None) -> Dict:
        """
        Export statistics to file or return as dict
        
        Args:
            filename: Optional filename to save stats as JSON
            
        Returns:
            Dict containing all statistics
        """
        comprehensive_stats = self.get_comprehensive_stats()
        performance_health = self.get_performance_health()
        
        export_data = {
            'export_timestamp': time.time(),
            'statistics': comprehensive_stats,
            'performance_health': performance_health,
            'configuration': {
                'window_size': self.window_size,
                'fps_calculation_interval': self.fps_calculation_interval
            }
        }
        
        if filename:
            import json
            try:
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                print(f"Statistics exported to {filename}")
            except Exception as e:
                print(f"Error exporting statistics: {e}")
        
        return export_data
    
    def print_stats_summary(self):
        """Print a formatted summary of current statistics"""
        rolling = self.get_rolling_stats()
        session = self.get_session_stats()
        health = self.get_performance_health()
        
        print("\n" + "="*50)
        print("DEEPFACELIVE FREEZE-FRAME PERFORMANCE SUMMARY")
        print("="*50)
        
        print(f"\n📊 CURRENT PERFORMANCE (last {len(self.processing_times)} frames):")
        print(f"   FPS: {rolling['fps']:.1f}")
        print(f"   Avg Confidence: {rolling['avg_confidence']:.3f}")
        print(f"   Freeze Ratio: {rolling['freeze_ratio']:.1%}")
        print(f"   Confidence Stability: {rolling['confidence_stability']:.3f}")
        
        print(f"\n📈 SESSION TOTALS:")
        print(f"   Duration: {session['session_duration']:.1f}s")
        print(f"   Total Frames: {session['total_frames']}")
        print(f"   Session FPS: {session['session_fps']:.1f}")
        print(f"   Total Freeze Ratio: {session['total_freeze_ratio']:.1%}")
        
        print(f"\n🏥 HEALTH STATUS: {health['status']} ({health['health_score']:.0f}/100)")
        if health['issues']:
            print(f"   Issues: {', '.join(health['issues'])}")
        if health['recommendations']:
            print(f"   Recommendations:")
            for rec in health['recommendations']:
                print(f"     • {rec}")
        
        print("="*50 + "\n")