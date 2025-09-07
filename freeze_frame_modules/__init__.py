"""
DeepFaceLive Freeze-Frame Module
Enhanced face swapping with confidence-based frame freezing
"""

__version__ = "1.0.0"
__author__ = "DeepFaceLive Freeze-Frame Enhancement"

# Import main classes for easy access
try:
    from .freeze_frame_manager import FreezeFrameManager
    from .face_detector_wrapper import FaceDetectorWrapper, DeepFaceLiveDetectorAdapter
    from .performance_monitor import PerformanceMonitor
    from .deepfacelive_integration import DeepFaceLiveFreezeProcessor
    from .deepfacelive_app_patch import DeepFaceLiveAppPatch

    __all__ = [
        'FreezeFrameManager',
        'FaceDetectorWrapper',
        'DeepFaceLiveDetectorAdapter',
        'PerformanceMonitor',
        'DeepFaceLiveFreezeProcessor',
        'DeepFaceLiveAppPatch'
    ]

    print("✓ Freeze-frame modules loaded successfully")

except ImportError as e:
    print(f"⚠ Warning: Some freeze-frame modules could not be imported: {e}")
    __all__ = []