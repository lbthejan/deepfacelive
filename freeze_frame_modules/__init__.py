"""
DeepFaceLive Freeze-Frame Module
Enhanced face swapping with confidence-based frame freezing
"""

__version__ = "1.0.0"
__author__ = "DeepFaceLive Freeze-Frame Enhancement"

# Safe imports with proper error handling
def _safe_import():
    """Safely import modules with proper error handling"""
    imported_modules = {}
    errors = []

    module_imports = [
        ('FreezeFrameManager', 'freeze_frame_manager'),
        ('FaceDetectorWrapper', 'face_detector_wrapper'),
        ('DeepFaceLiveDetectorAdapter', 'face_detector_wrapper'),
        ('PerformanceMonitor', 'performance_monitor'),
        ('DeepFaceLiveFreezeProcessor', 'deepfacelive_integration')
    ]

    for class_name, module_name in module_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            imported_modules[class_name] = getattr(module, class_name)
        except ImportError as e:
            errors.append(f"Failed to import {class_name} from {module_name}: {e}")
        except AttributeError as e:
            errors.append(f"Class {class_name} not found in {module_name}: {e}")

    return imported_modules, errors

# Perform safe imports
_modules, _import_errors = _safe_import()

# Export available modules
globals().update(_modules)
__all__ = list(_modules.keys())

if _modules:
    print(f"✓ Freeze-frame modules loaded successfully: {', '.join(_modules.keys())}")
else:
    print("⚠ Warning: No freeze-frame modules could be imported")

if _import_errors:
    print("⚠ Import warnings:")
    for error in _import_errors:
        print(f"  - {error}")

# Backward compatibility
def get_available_modules():
    """Get list of successfully imported modules"""
    return list(_modules.keys())

def is_module_available(module_name):
    """Check if a specific module is available"""
    return module_name in _modules

def get_import_errors():
    """Get list of import errors"""
    return _import_errors.copy()

# Module status
FREEZE_FRAME_READY = len(_modules) >= 3  # At least core modules available