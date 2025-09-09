"""
UI Enhancement for DeepFaceLive Freeze-Frame Feature
Adds a button to activate/deactivate freeze functionality in the main UI
"""

import sys
from pathlib import Path
from localization import L
from xlib import qt as qtx

# Add freeze-frame modules to path
current_dir = Path(__file__).parent
freeze_modules_path = current_dir / "freeze_frame_modules"
if freeze_modules_path.exists():
    sys.path.insert(0, str(freeze_modules_path))

try:
    from freeze_frame_manager import FreezeFrameManager
    from deepfacelive_integration import DeepFaceLiveFreezeProcessor
    FREEZE_AVAILABLE = True
except ImportError:
    FREEZE_AVAILABLE = False


class QFreezeFramePanel(qtx.QXWidget):
    """
    Freeze Frame Control Panel Widget
    Provides UI controls for freeze-frame functionality
    """
    
    def __init__(self, freeze_processor=None):
        super().__init__()
        self.freeze_processor = freeze_processor
        self.is_freeze_enabled = False
        
        self._create_ui()
        self._setup_connections()
    
    def _create_ui(self):
        """Create the UI elements"""
        # Main freeze enable/disable button
        self.btn_freeze_toggle = qtx.QXPushButton(
            text="Enable Freeze-Frame",
            fixed_size=(150, 30),
            released=self._on_freeze_toggle
        )
        self.btn_freeze_toggle.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # Threshold slider
        self.threshold_label = qtx.QXLabel("Confidence Threshold:")
        self.threshold_slider = qtx.QXSlider(
            orientation=qtx.Qt.Orientation.Horizontal,
            min=30, max=95, value=75,
            tick_position=qtx.QSlider.TickPosition.TicksBelow,
            tick_interval=10
        )
        self.threshold_value_label = qtx.QXLabel("0.75")
        
        # Status indicators
        self.status_label = qtx.QXLabel("Status: Disabled")
        self.confidence_label = qtx.QXLabel("Confidence: --")
        self.freeze_count_label = qtx.QXLabel("Freezes: 0")
        
        # Advanced controls (collapsible)
        self.advanced_section = qtx.QXCollapsibleSection(
            title="Advanced Settings",
            is_opened=False
        )
        
        # Buffer size control
        buffer_label = qtx.QXLabel("Frame Buffer Size:")
        self.buffer_spinbox = qtx.QXSpinBox(min=1, max=10, value=3)
        
        # Max freeze duration control
        duration_label = qtx.QXLabel("Max Freeze Duration (s):")
        self.duration_spinbox = qtx.QXDoubleSpinBox(
            min=1.0, max=10.0, value=5.0, decimals=1, step=0.5
        )
        
        # Stats toggle
        self.stats_checkbox = qtx.QXCheckBox("Show Statistics Overlay")
        self.stats_checkbox.setChecked(True)
        
        # Reset button
        self.btn_reset_stats = qtx.QXPushButton(
            text="Reset Statistics",
            fixed_size=(120, 25),
            released=self._on_reset_stats
        )
        
        # Advanced controls layout
        advanced_layout = qtx.QXVBoxLayout([
            qtx.QXHBoxLayout([buffer_label, self.buffer_spinbox]),
            qtx.QXHBoxLayout([duration_label, self.duration_spinbox]),
            self.stats_checkbox,
            self.btn_reset_stats
        ])
        self.advanced_section.setLayout(advanced_layout)
        
        # Main layout
        main_layout = qtx.QXVBoxLayout([
            # Main controls
            self.btn_freeze_toggle,
            qtx.QXFrame(fixed_height=10),  # Spacer
            
            # Threshold control
            self.threshold_label,
            qtx.QXHBoxLayout([
                self.threshold_slider,
                self.threshold_value_label
            ]),
            qtx.QXFrame(fixed_height=10),  # Spacer
            
            # Status display
            self.status_label,
            self.confidence_label,
            self.freeze_count_label,
            qtx.QXFrame(fixed_height=10),  # Spacer
            
            # Advanced section
            self.advanced_section
        ])
        
        # Main container with border
        container = qtx.QXFrameVBox([main_layout], 
                                   contents_margins=10,
                                   bg_color=qtx.QColor(240, 240, 240))
        
        self.setLayout(qtx.QXHBoxLayout([container]))
        self.setFixedWidth(300)
    
    def _setup_connections(self):
        """Setup signal connections"""
        # Threshold slider
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        
        # Advanced controls
        self.buffer_spinbox.valueChanged.connect(self._on_buffer_size_changed)
        self.duration_spinbox.valueChanged.connect(self._on_duration_changed)
        self.stats_checkbox.toggled.connect(self._on_stats_toggle)
        
        # Update timer for status display
        self.update_timer = qtx.QXTimer(interval=100, timeout=self._update_status)
    
    def set_freeze_processor(self, freeze_processor):
        """Set the freeze processor instance"""
        self.freeze_processor = freeze_processor
        if freeze_processor:
            self._sync_ui_with_processor()
    
    def _sync_ui_with_processor(self):
        """Sync UI controls with processor settings"""
        if not self.freeze_processor:
            return
        
        # Get current settings
        threshold = self.freeze_processor.freeze_manager.confidence_threshold
        buffer_size = self.freeze_processor.freeze_manager.frame_buffer_size
        max_duration = self.freeze_processor.freeze_manager.max_freeze_duration
        show_stats = self.freeze_processor.show_stats_overlay
        
        # Update UI
        self.threshold_slider.setValue(int(threshold * 100))
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self.buffer_spinbox.setValue(buffer_size)
        self.duration_spinbox.setValue(max_duration)
        self.stats_checkbox.setChecked(show_stats)
    
    def _on_freeze_toggle(self):
        """Handle freeze toggle button"""
        if not FREEZE_AVAILABLE:
            qtx.QXMessageBox.information(
                self, "Freeze-Frame Not Available",
                "Freeze-frame modules are not installed or not working properly."
            )
            return
        
        self.is_freeze_enabled = not self.is_freeze_enabled
        
        if self.is_freeze_enabled:
            self.btn_freeze_toggle.setText("Disable Freeze-Frame")
            self.btn_freeze_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #c4130b;
                }
            """)
            self.status_label.setText("Status: Active")
            self.update_timer.start()
            
            # Enable freeze processor if available
            if self.freeze_processor:
                self.freeze_processor.start_processing()
        
        else:
            self.btn_freeze_toggle.setText("Enable Freeze-Frame")
            self.btn_freeze_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            self.status_label.setText("Status: Disabled")
            self.confidence_label.setText("Confidence: --")
            self.freeze_count_label.setText("Freezes: 0")
            self.update_timer.stop()
            
            # Disable freeze processor if available
            if self.freeze_processor:
                self.freeze_processor.stop_processing()
        
        # Emit signal for main app
        self.freeze_toggled.emit(self.is_freeze_enabled)
    
    def _on_threshold_changed(self, value):
        """Handle threshold slider change"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        
        if self.freeze_processor:
            self.freeze_processor.update_freeze_threshold(threshold)
    
    def _on_buffer_size_changed(self, value):
        """Handle buffer size change"""
        if self.freeze_processor:
            self.freeze_processor.freeze_manager.frame_buffer_size = value
            self.freeze_processor.freeze_manager.save_config()
    
    def _on_duration_changed(self, value):
        """Handle max duration change"""
        if self.freeze_processor:
            self.freeze_processor.freeze_manager.max_freeze_duration = value
            self.freeze_processor.freeze_manager.save_config()
    
    def _on_stats_toggle(self, checked):
        """Handle stats overlay toggle"""
        if self.freeze_processor:
            self.freeze_processor.show_stats_overlay = checked
    
    def _on_reset_stats(self):
        """Handle reset statistics"""
        if self.freeze_processor:
            self.freeze_processor.reset_stats()
            self.freeze_count_label.setText("Freezes: 0")
    
    def _update_status(self):
        """Update status display"""
        if not self.freeze_processor or not self.is_freeze_enabled:
            return
        
        try:
            stats = self.freeze_processor.get_stats()
            freeze_stats = stats.get('freeze', {})
            
            # Update confidence display
            current_conf = freeze_stats.get('current_confidence', 0.0)
            self.confidence_label.setText(f"Confidence: {current_conf:.3f}")
            
            # Update freeze count
            freeze_count = freeze_stats.get('freeze_count', 0)
            self.freeze_count_label.setText(f"Freezes: {freeze_count}")
            
            # Update status based on freeze state
            is_frozen = freeze_stats.get('is_frozen', False)
            if is_frozen:
                self.status_label.setText("Status: FROZEN")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.status_label.setText("Status: LIVE")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        except Exception as e:
            print(f"Error updating freeze-frame status: {e}")
    
    # Custom signal
    freeze_toggled = qtx.pyqtSignal(bool)


class DeepFaceLiveAppWithFreezeUI:
    """
    Enhanced DeepFaceLive app with freeze-frame UI integration
    """
    
    def __init__(self, original_app_class):
        self.original_app_class = original_app_class
        self.freeze_processor = None
        self.freeze_panel = None
    
    def create_enhanced_app(self, *args, **kwargs):
        """Create enhanced app with freeze-frame UI"""
        
        class EnhancedApp(self.original_app_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Initialize freeze processor if available
                if FREEZE_AVAILABLE:
                    userdata_path = kwargs.get('userdata_path', Path.cwd())
                    self.freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)
                else:
                    self.freeze_processor = None
                
                # Will be set when UI is created
                self.freeze_panel = None
            
            def on_ui_create(self):
                """Override to add freeze-frame panel"""
                # Call original UI creation
                if hasattr(super(), 'on_ui_create'):
                    super().on_ui_create()
                
                # Add freeze-frame panel
                self._add_freeze_frame_panel()
            
            def _add_freeze_frame_panel(self):
                """Add freeze-frame control panel to UI"""
                try:
                    # Create freeze panel
                    self.freeze_panel = QFreezeFramePanel(self.freeze_processor)
                    
                    # Connect signals
                    self.freeze_panel.freeze_toggled.connect(self._on_freeze_toggled)
                    
                    # Add to main layout (this depends on your UI structure)
                    if hasattr(self, 'main_layout'):
                        self.main_layout.addWidget(self.freeze_panel)
                    elif hasattr(self, 'control_panel_layout'):
                        self.control_panel_layout.addWidget(self.freeze_panel)
                    elif hasattr(self, 'setLayout'):
                        # Try to add to existing layout
                        current_layout = self.layout()
                        if current_layout:
                            current_layout.addWidget(self.freeze_panel)
                    
                    print("✓ Freeze-frame UI panel added")
                
                except Exception as e:
                    print(f"⚠ Could not add freeze-frame UI panel: {e}")
            
            def _on_freeze_toggled(self, enabled):
                """Handle freeze toggle from UI"""
                print(f"Freeze-frame {'enabled' if enabled else 'disabled'} from UI")
                
                # Connect or disconnect freeze processor
                if enabled and self.freeze_processor:
                    # Try to connect to face detector
                    self._connect_freeze_to_detector()
                
            def _connect_freeze_to_detector(self):
                """Connect freeze processor to face detector"""
                try:
                    # Look for face detector in app
                    face_detector = None
                    
                    # Multiple strategies to find face detector
                    detection_attrs = ['face_detector', '_face_detector', 'detector']
                    for attr in detection_attrs:
                        if hasattr(self, attr):
                            face_detector = getattr(self, attr)
                            break
                    
                    # Check in backends list if available
                    if not face_detector and hasattr(self, 'all_backends'):
                        for backend in self.all_backends:
                            if 'FaceDetector' in backend.__class__.__name__:
                                face_detector = backend
                                break
                    
                    if face_detector:
                        self.freeze_processor.initialize_face_detector(face_detector)
                        print("✓ Freeze processor connected to face detector")
                    else:
                        print("⚠ Could not find face detector for freeze processor")
                
                except Exception as e:
                    print(f"Error connecting freeze processor: {e}")
            
            def process_frame_with_freeze(self, frame):
                """Enhanced frame processing with freeze logic"""
                if (self.freeze_panel and 
                    self.freeze_panel.is_freeze_enabled and 
                    self.freeze_processor and
                    self.freeze_processor.face_detector_adapter):
                    
                    try:
                        # Get face swapper and other components
                        face_swapper = getattr(self, 'face_swapper', None)
                        predictor = getattr(self, 'predictor', None)
                        face_enhancer = getattr(self, 'face_enhancer', None)
                        
                        if face_swapper:
                            return self.freeze_processor.process_frame(
                                frame, face_swapper, predictor, face_enhancer
                            )
                    
                    except Exception as e:
                        print(f"Error in freeze processing: {e}")
                
                # Fallback to original processing
                if hasattr(super(), 'process_frame'):
                    return super().process_frame(frame)
                else:
                    return frame
        
        return EnhancedApp(*args, **kwargs)


# Integration function for existing DeepFaceLive installations
def add_freeze_ui_to_existing_app(app_instance):
    """
    Add freeze-frame UI to an existing DeepFaceLive app instance
    """
    try:
        # Initialize freeze processor
        if FREEZE_AVAILABLE:
            userdata_path = getattr(app_instance, 'userdata_path', Path.cwd())
            freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)
        else:
            freeze_processor = None
        
        # Create freeze panel
        freeze_panel = QFreezeFramePanel(freeze_processor)
        
        # Add to app
        app_instance.freeze_processor = freeze_processor
        app_instance.freeze_panel = freeze_panel
        
        # Try to add to UI
        if hasattr(app_instance, 'main_layout'):
            app_instance.main_layout.addWidget(freeze_panel)
        elif hasattr(app_instance, 'layout') and app_instance.layout():
            app_instance.layout().addWidget(freeze_panel)
        else:
            print("⚠ Could not automatically add freeze panel to UI")
            print("  You may need to manually add freeze_panel to your UI layout")
        
        print("✓ Freeze-frame UI added to existing app")
        return freeze_panel
    
    except Exception as e:
        print(f"Error adding freeze UI to app: {e}")
        return None


# Standalone freeze control window
class FreezeFrameControlWindow(qtx.QXWindow):
    """
    Standalone freeze-frame control window
    Can be used as a separate control panel
    """
    
    def __init__(self):
        super().__init__(
            title="DeepFaceLive Freeze-Frame Control",
            size_policy=('fixed', 'fixed')
        )
        
        # Initialize freeze processor
        if FREEZE_AVAILABLE:
            self.freeze_processor = DeepFaceLiveFreezeProcessor()
        else:
            self.freeze_processor = None
        
        # Create freeze panel
        self.freeze_panel = QFreezeFramePanel(self.freeze_processor)
        
        # Add help button
        help_button = qtx.QXPushButton(
            text="Help",
            fixed_size=(60, 25),
            released=self._show_help
        )
        
        # Layout
        layout = qtx.QXVBoxLayout([
            self.freeze_panel,
            qtx.QXFrame(fixed_height=10),  # Spacer
            qtx.QXHBoxLayout([qtx.QXFrame(), help_button])  # Right-aligned help
        ])
        
        self.setLayout(layout)
        
        # Connect close event
        self.call_on_closeEvent(self._on_close)
    
    def _show_help(self):
        """Show help dialog"""
        help_text = """
DeepFaceLive Freeze-Frame Control

This panel controls the freeze-frame functionality that helps eliminate
glitches during face swapping by freezing the output when face detection
confidence drops below the threshold.

Controls:
• Enable/Disable: Toggle freeze-frame functionality
• Confidence Threshold: Minimum confidence to keep live output
• Frame Buffer Size: Number of frames to average confidence
• Max Freeze Duration: Maximum time to keep frame frozen
• Show Statistics: Display confidence and freeze info on video
• Reset Statistics: Clear freeze count and timing data

Tips:
• Higher threshold = more freezing but fewer glitches
• Lower threshold = less freezing but more glitches may show
• Adjust based on your lighting and camera conditions
        """.strip()
        
        qtx.QXMessageBox.information(self, "Help - Freeze-Frame Control", help_text)
    
    def _on_close(self):
        """Handle window close"""
        if self.freeze_processor:
            self.freeze_processor.cleanup()


def create_freeze_control_window():
    """Create standalone freeze control window"""
    return FreezeFrameControlWindow()


# Example usage in main.py modification
def modify_main_py_for_freeze_ui():
    """
    Example of how to modify main.py to include freeze UI
    """
    example_code = '''
# Add this to your main.py after importing other modules

# Import freeze UI enhancement
try:
    from freeze_ui_enhancement import DeepFaceLiveAppWithFreezeUI
    FREEZE_UI_AVAILABLE = True
except ImportError:
    FREEZE_UI_AVAILABLE = False

# Modify your run_DeepFaceLive function
def run_DeepFaceLive(args):
    userdata_path = Path(args.userdata_dir) if args.userdata_dir else Path.cwd()
    
    from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
    
    if FREEZE_UI_AVAILABLE:
        # Create enhanced app with freeze UI
        enhancer = DeepFaceLiveAppWithFreezeUI(DeepFaceLiveApp)
        app = enhancer.create_enhanced_app(userdata_path=userdata_path)
        print("✓ DeepFaceLive started with freeze-frame UI")
    else:
        # Fallback to original app
        app = DeepFaceLiveApp(userdata_path=userdata_path)
        print("Running original DeepFaceLive (no freeze UI)")
    
    app.run()
'''
    
    print("Example code for main.py integration:")
    print(example_code)


if __name__ == "__main__":
    # Test the freeze control window
    import sys
    
    app = qtx.QXMainApplication(sys.argv)
    
    window = create_freeze_control_window()
    window.show()
    
    sys.exit(app.exec())
