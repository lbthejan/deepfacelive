# Critical Bug Fixes for DeepFaceLive Freeze-Frame

# 1. Fix import paths in main.py
def setup_freeze_frame():
    """Setup freeze-frame functionality with proper imports"""
    current_dir = Path(__file__).parent
    freeze_modules_path = current_dir / "freeze_frame_modules"

    if freeze_modules_path.exists():
        if str(freeze_modules_path) not in sys.path:
            sys.path.insert(0, str(freeze_modules_path))

        try:
            # FIXED: Use full module path
            from freeze_frame_modules.freeze_frame_manager import FreezeFrameManager
            print("✓ Freeze-frame functionality enabled")
            return True
        except ImportError as e:
            print(f"⚠️  Freeze-frame modules found but couldn't import: {e}")
            return False
    else:
        print("ℹ️  No freeze-frame modules found")
        return False

# 2. Fix JSON loading with proper error handling
def load_config_safe(self):
    """Load configuration with proper error handling"""
    if os.path.exists(self.config_file):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
                self.max_freeze_duration = config.get('max_freeze_duration', self.max_freeze_duration)
                self.frame_buffer_size = config.get('frame_buffer_size', self.frame_buffer_size)
                print(f"Loaded freeze-frame config: threshold={self.confidence_threshold}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading freeze config: {e}, using defaults")
        except Exception as e:
            print(f"Unexpected error loading config: {e}")

# 3. Fix method call issue - safer approach
class EnhancedDeepFaceLiveApp(DeepFaceLiveApp):
    def process_frame_safe(self, frame, backend_dict):
        """Safe frame processing with fallback"""
        try:
            # Try freeze processing first
            if (hasattr(self, 'freeze_processor') and 
                self.freeze_processor and
                self.freeze_processor.face_detector_adapter):
                
                face_swapper = backend_dict.get('face_swapper')
                if face_swapper:
                    return self.freeze_processor.process_frame(
                        frame, face_swapper, None, backend_dict.get('face_merger')
                    )
            
            # Fallback - return original frame if no processing available
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

# 4. Fix confidence buffer memory management
def update_confidence_buffer_safe(self, confidence: float):
    """Update confidence buffer with size limit"""
    self.confidence_buffer.append(confidence)
    # FIXED: Prevent memory leak
    while len(self.confidence_buffer) > self.frame_buffer_size:
        self.confidence_buffer.pop(0)

# 5. Fix platform-specific file paths
def get_data_separator():
    """Get platform-specific data separator for PyInstaller"""
    return ';' if os.name == 'nt' else ':'

def add_data_files(build_cmd, source, dest):
    """Add data files with platform-specific separator"""
    separator = get_data_separator()
    build_cmd.extend(["--add-data", f"{source}{separator}{dest}"])

# 6. Fix thread-safe performance monitoring
import threading

class ThreadSafePerformanceMonitor:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._lock = threading.Lock()
        self.processing_times = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.freeze_states = deque(maxlen=window_size)
    
    def update(self, processing_time: float, confidence: float, is_frozen: bool):
        """Thread-safe update"""
        with self._lock:
            self.processing_times.append(processing_time)
            self.confidences.append(confidence)
            self.freeze_states.append(is_frozen)

# 7. Fix robust face detector finding
def find_face_detector_robust(app_instance):
    """Robustly find face detector in app"""
    face_detector = None
    
    # Strategy 1: Check common attribute names
    for attr_name in ['face_detector', '_face_detector', 'detector', 'face_det']:
        if hasattr(app_instance, attr_name):
            candidate = getattr(app_instance, attr_name)
            if candidate and hasattr(candidate, 'extract'):  # Has detection method
                face_detector = candidate
                print(f"Found face detector via attribute: {attr_name}")
                break
    
    # Strategy 2: Check backends/modules list
    if not face_detector:
        for attr_name in ['_backends', 'backends', 'modules', '_modules']:
            if hasattr(app_instance, attr_name):
                backends = getattr(app_instance, attr_name)
                if backends:
                    for backend in backends:
                        class_name = backend.__class__.__name__.lower()
                        if 'detector' in class_name and hasattr(backend, 'extract'):
                            face_detector = backend
                            print(f"Found face detector in {attr_name}: {backend.__class__.__name__}")
                            break
                if face_detector:
                    break
    
    return face_detector

# 8. Fix exception handling with specific exceptions
def safe_freeze_processing(self, frame, face_swapper, predictor=None, face_enhancer=None):
    """Process frame with comprehensive error handling"""
    try:
        return self.process_frame(frame, face_swapper, predictor, face_enhancer)
    except AttributeError as e:
        print(f"Attribute error in freeze processing: {e}")
        return frame
    except TypeError as e:
        print(f"Type error in freeze processing: {e}")
        return frame
    except ValueError as e:
        print(f"Value error in freeze processing: {e}")
        return frame
    except Exception as e:
        print(f"Unexpected error in freeze processing: {e}")
        return frame

# 9. Fix import validation
def validate_freeze_modules():
    """Validate all required freeze modules are available"""
    required_modules = [
        'freeze_frame_manager',
        'face_detector_wrapper', 
        'performance_monitor',
        'deepfacelive_integration'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(f'freeze_frame_modules.{module}')
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Missing freeze modules: {missing}")
        return False
    
    print("All freeze modules validated")
    return True

# 10. Fix configuration validation
def validate_config(config_dict):
    """Validate configuration values"""
    valid_config = {}
    
    # Validate threshold
    threshold = config_dict.get('confidence_threshold', 0.75)
    valid_config['confidence_threshold'] = max(0.0, min(1.0, float(threshold)))
    
    # Validate max freeze duration
    max_freeze = config_dict.get('max_freeze_duration', 5.0)
    valid_config['max_freeze_duration'] = max(0.1, min(30.0, float(max_freeze)))
    
    # Validate buffer size
    buffer_size = config_dict.get('frame_buffer_size', 3)
    valid_config['frame_buffer_size'] = max(1, min(20, int(buffer_size)))
    
    return valid_config