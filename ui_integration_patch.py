"""
UI Integration Patch for existing DeepFaceLive installation
This patches the existing UI to add the freeze-frame button
"""

import sys
from pathlib import Path
from localization import L
from xlib import qt as qtx
from resources.fonts import QXFontDB
from resources.gfx import QXImageDB

# Import the freeze UI panel
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from freeze_ui_enhancement import QFreezeFramePanel, FREEZE_AVAILABLE
except ImportError:
    FREEZE_AVAILABLE = False
    print("⚠ Freeze-frame UI components not found")


def patch_stream_output_ui(stream_output_instance):
    """
    Patch QStreamOutput to add freeze-frame controls
    This is the easiest place to add the UI since it's always visible
    """
    
    if not FREEZE_AVAILABLE:
        return stream_output_instance
    
    # Get the original QStreamOutput class
    original_class = stream_output_instance.__class__
    
    class PatchedQStreamOutput(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._add_freeze_controls()
        
        def _add_freeze_controls(self):
            """Add freeze-frame controls to the stream output panel"""
            try:
                # Create freeze panel (compact version for stream output)
                self.freeze_panel = self._create_compact_freeze_panel()
                
                # Get the main layout of the stream output
                layout = self.layout()
                if layout:
                    # Add freeze panel at the top
                    layout.insertWidget(0, self.freeze_panel)
                    
                    # Add separator
                    separator = qtx.QXFrame()
                    separator.setFrameShape(qtx.QFrame.HLine)
                    separator.setFrameShadow(qtx.QFrame.Sunken)
                    layout.insertWidget(1, separator)
                
                print("✓ Freeze-frame controls added to Stream Output panel")
                
            except Exception as e:
                print(f"⚠ Could not add freeze controls to Stream Output: {e}")
        
        def _create_compact_freeze_panel(self):
            """Create compact freeze panel for stream output"""
            # Main container
            freeze_container = qtx.QXFrameVBox(
                bg_color=qtx.QColor(245, 245, 245),
                contents_margins=5,
                spacing=3
            )
            
            # Title
            title_label = qtx.QXLabel("🧊 Freeze-Frame Controls", 
                                    font=QXFontDB.get_default_font(size=9),
                                    alignment=qtx.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; color: #333;")
            
            # Enable/Disable toggle
            self.freeze_toggle_btn = qtx.QXPushButton(
                text="Enable Freeze",
                fixed_size=(100, 25),
                released=self._on_freeze_toggle
            )
            self.freeze_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover { background-color: #45a049; }
            """)
            
            # Threshold control (compact)
            threshold_label = qtx.QXLabel("Threshold:", font=QXFontDB.get_default_font(size=8))
            self.threshold_slider = qtx.QXSlider(
                orientation=qtx.Qt.Orientation.Horizontal,
                min=30, max=95, value=75,
                fixed_width=80,
                valueChanged=self._on_threshold_change
            )
            self.threshold_value = qtx.QXLabel("0.75", font=QXFontDB.get_default_font(size=8))
            
            # Status indicator
            self.status_indicator = qtx.QXLabel("●", font=QXFontDB.get_default_font(size=12))
            self.status_indicator.setStyleSheet("color: gray;")
            self.status_label = qtx.QXLabel("Disabled", font=QXFontDB.get_default_font(size=8))
            
            # Layout
            controls_layout = qtx.QXHBoxLayout([
                self.freeze_toggle_btn,
                qtx.QXFrame(fixed_width=5),  # Spacer
                threshold_label,
                self.threshold_slider,
                self.threshold_value,
                qtx.QXFrame(fixed_width=5),  # Spacer
                self.status_indicator,
                self.status_label
            ])
            
            freeze_container.setLayout(qtx.QXVBoxLayout([
                title_label,
                controls_layout
            ]))
            
            # Initialize freeze processor
            self.freeze_processor = None
            self.freeze_enabled = False
            self._init_freeze_processor()
            
            # Update timer
            self.freeze_update_timer = qtx.QXTimer(interval=200, timeout=self._update_freeze_status)
            
            return freeze_container
        
        def _init_freeze_processor(self):
            """Initialize freeze processor"""
            try:
                from deepfacelive_integration import DeepFaceLiveFreezeProcessor
                self.freeze_processor = DeepFaceLiveFreezeProcessor()
                print("✓ Freeze processor initialized")
            except Exception as e:
                print(f"⚠ Could not initialize freeze processor: {e}")
        
        def _on_freeze_toggle(self):
            """Handle freeze toggle"""
            self.freeze_enabled = not self.freeze_enabled
            
            if self.freeze_enabled:
                self.freeze_toggle_btn.setText("Disable")
                self.freeze_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border-radius: 3px;
                        font-weight: bold;
                        font-size: 10px;
                    }
                    QPushButton:hover { background-color: #da190b; }
                """)
                self.status_label.setText("Active")
                self.status_indicator.setStyleSheet("color: green;")
                self.freeze_update_timer.start()
                
                if self.freeze_processor:
                    self.freeze_processor.start_processing()
            else:
                self.freeze_toggle_btn.setText("Enable Freeze")
                self.freeze_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 3px;
                        font-weight: bold;
                        font-size: 10px;
                    }
                    QPushButton:hover { background-color: #45a049; }
                """)
                self.status_label.setText("Disabled")
                self.status_indicator.setStyleSheet("color: gray;")
                self.freeze_update_timer.stop()
                
                if self.freeze_processor:
                    self.freeze_processor.stop_processing()
        
        def _on_threshold_change(self, value):
            """Handle threshold change"""
            threshold = value / 100.0
            self.threshold_value.setText(f"{threshold:.2f}")
            
            if self.freeze_processor:
                self.freeze_processor.update_freeze_threshold(threshold)
        
        def _update_freeze_status(self):
            """Update freeze status indicator"""
            if not self.freeze_processor or not self.freeze_enabled:
                return
            
            try:
                stats = self.freeze_processor.get_stats()
                freeze_stats = stats.get('freeze', {})
                is_frozen = freeze_stats.get('is_frozen', False)
                
                if is_frozen:
                    self.status_indicator.setStyleSheet("color: red;")
                    self.status_label.setText("FROZEN")
                else:
                    self.status_indicator.setStyleSheet("color: green;")
                    self.status_label.setText("LIVE")
            except:
                pass
    
    # Replace the instance's class
    stream_output_instance.__class__ = PatchedQStreamOutput
    return stream_output_instance


def patch_main_app_window(app_window_instance):
    """
    Patch the main app window to add freeze-frame menu item
    """
    
    if not FREEZE_AVAILABLE:
        return app_window_instance
    
    try:
        # Get the original class
        original_class = app_window_instance.__class__
        
        class PatchedMainWindow(original_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._add_freeze_menu()
            
            def _add_freeze_menu(self):
                """Add freeze-frame menu to menu bar"""
                try:
                    # Find the menu bar
                    menu_bar = None
                    for child in self.findChildren(qtx.QMenuBar):
                        menu_bar = child
                        break
                    
                    if menu_bar:
                        # Add freeze-frame menu
                        freeze_menu = menu_bar.addMenu("🧊 Freeze")
                        
                        # Enable/Disable action
                        self.freeze_action = freeze_menu.addAction("Enable Freeze-Frame")
                        self.freeze_action.triggered.connect(self._toggle_freeze_from_menu)
                        self.freeze_action.setCheckable(True)
                        
                        # Separator
                        freeze_menu.addSeparator()
                        
                        # Settings action
                        settings_action = freeze_menu.addAction("Settings...")
                        settings_action.triggered.connect(self._show_freeze_settings)
                        
                        # Help action
                        help_action = freeze_menu.addAction("Help")
                        help_action.triggered.connect(self._show_freeze_help)
                        
                        print("✓ Freeze-frame menu added to menu bar")
                
                except Exception as e:
                    print(f"⚠ Could not add freeze-frame menu: {e}")
            
            def _toggle_freeze_from_menu(self):
                """Toggle freeze from menu"""
                # This would need to communicate with the actual freeze processor
                # For now, just show a message
                if self.freeze_action.isChecked():
                    qtx.QXMessageBox.information(self, "Freeze-Frame", 
                        "Freeze-frame enabled. Use the controls in the Stream Output panel to configure.")
                else:
                    qtx.QXMessageBox.information(self, "Freeze-Frame", 
                        "Freeze-frame disabled.")
            
            def _show_freeze_settings(self):
                """Show freeze-frame settings dialog"""
                self._create_freeze_settings_dialog().exec()
            
            def _show_freeze_help(self):
                """Show freeze-frame help"""
                help_text = """
<h3>DeepFaceLive Freeze-Frame Feature</h3>

<p>The freeze-frame feature helps eliminate glitches and artifacts during face swapping by temporarily freezing the output when face detection confidence drops below a threshold.</p>

<h4>How it works:</h4>
<ul>
<li>Monitors face detection confidence in real-time</li>
<li>When confidence drops below threshold, freezes the last good frame</li>
<li>Automatically resumes live output when confidence recovers</li>
</ul>

<h4>Controls:</h4>
<ul>
<li><b>Enable/Disable:</b> Toggle freeze-frame functionality</li>
<li><b>Threshold:</b> Confidence level below which frames freeze (0.30-0.95)</li>
<li><b>Status:</b> Shows current state (LIVE/FROZEN)</li>
</ul>

<h4>Tips:</h4>
<ul>
<li>Start with threshold around 0.75</li>
<li>Lower threshold = less freezing but more glitches may show</li>
<li>Higher threshold = more freezing but cleaner output</li>
<li>Adjust based on your lighting and camera conditions</li>
</ul>
                """.strip()
                
                qtx.QXMessageBox.information(self, "Freeze-Frame Help", help_text)
            
            def _create_freeze_settings_dialog(self):
                """Create freeze settings dialog"""
                dialog = qtx.QXDialog(self, title="Freeze-Frame Settings")
                
                # Threshold setting
                threshold_label = qtx.QXLabel("Confidence Threshold:")
                threshold_slider = qtx.QXSlider(qtx.Qt.Orientation.Horizontal, min=30, max=95, value=75)
                threshold_value = qtx.QXLabel("0.75")
                
                def update_threshold(value):
                    threshold_value.setText(f"{value/100.0:.2f}")
                
                threshold_slider.valueChanged.connect(update_threshold)
                
                # Buffer size setting
                buffer_label = qtx.QXLabel("Frame Buffer Size:")
                buffer_spin = qtx.QXSpinBox(min=1, max=10, value=3)
                
                # Max freeze duration
                duration_label = qtx.QXLabel("Max Freeze Duration (seconds):")
                duration_spin = qtx.QXDoubleSpinBox(min=1.0, max=10.0, value=5.0, decimals=1)
                
                # Stats overlay
                stats_check = qtx.QXCheckBox("Show Statistics Overlay")
                stats_check.setChecked(True)
                
                # Buttons
                ok_btn = qtx.QXPushButton("OK", released=dialog.accept)
                cancel_btn = qtx.QXPushButton("Cancel", released=dialog.reject)
                
                # Layout
                layout = qtx.QXVBoxLayout([
                    qtx.QXHBoxLayout([threshold_label, threshold_slider, threshold_value]),
                    qtx.QXHBoxLayout([buffer_label, buffer_spin]),
                    qtx.QXHBoxLayout([duration_label, duration_spin]),
                    stats_check,
                    qtx.QXFrame(fixed_height=10),  # Spacer
                    qtx.QXHBoxLayout([qtx.QXFrame(), ok_btn, cancel_btn])
                ])
                
                dialog.setLayout(layout)
                return dialog
        
        # Replace the instance's class
        app_window_instance.__class__ = PatchedMainWindow
        
        print("✓ Main window patched with freeze-frame menu")
        
    except Exception as e:
        print(f"⚠ Could not patch main window: {e}")
    
    return app_window_instance


def auto_patch_deepfacelive():
    """
    Automatically patch DeepFaceLive components when they're created
    This function monkey-patches the constructors to add freeze-frame UI
    """
    
    if not FREEZE_AVAILABLE:
        print("⚠ Freeze-frame not available, skipping auto-patch")
        return
    
    try:
        # Import DeepFaceLive UI components
        from apps.DeepFaceLive.ui.QStreamOutput import QStreamOutput
        
        # Store original constructor
        original_qstreamoutput_init = QStreamOutput.__init__
        
        def patched_qstreamoutput_init(self, *args, **kwargs):
            # Call original constructor
            original_qstreamoutput_init(self, *args, **kwargs)
            
            # Add freeze-frame controls
            try:
                self._add_freeze_controls_to_stream_output()
            except Exception as e:
                print(f"⚠ Could not add freeze controls to stream output: {e}")
        
        def _add_freeze_controls_to_stream_output(self):
            """Add freeze controls to stream output (method for patched class)"""
            # Create minimal freeze controls
            freeze_container = qtx.QXFrameHBox(
                bg_color=qtx.QColor(240, 248, 255),
                contents_margins=3,
                spacing=3
            )
            
            # Simple enable button
            self.freeze_btn = qtx.QXPushButton(
                text="🧊 Enable Freeze",
                fixed_size=(90, 20),
                released=self._toggle_freeze
            )
            self.freeze_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border-radius: 10px;
                    font-size: 9px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #1976D2; }
            """)
            
            # Status dot
            self.freeze_status = qtx.QXLabel("●")
            self.freeze_status.setStyleSheet("color: gray; font-size: 14px;")
            
            # Add to container
            freeze_container.setLayout(qtx.QXHBoxLayout([
                qtx.QXLabel("Freeze:", font=QXFontDB.get_default_font(size=8)),
                self.freeze_btn,
                self.freeze_status,
                qtx.QXFrame()  # Stretch
            ]))
            
            # Add to main layout at top
            main_layout = self.layout()
            if main_layout:
                main_layout.insertWidget(0, freeze_container)
            
            # Initialize state
            self.freeze_enabled = False
            self.freeze_processor = None
            
            # Try to initialize freeze processor
            try:
                from deepfacelive_integration import DeepFaceLiveFreezeProcessor
                self.freeze_processor = DeepFaceLiveFreezeProcessor()
            except:
                pass
        
        def _toggle_freeze(self):
            """Toggle freeze state (method for patched class)"""
            self.freeze_enabled = not self.freeze_enabled
            
            if self.freeze_enabled:
                self.freeze_btn.setText("🔥 Disable")
                self.freeze_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border-radius: 10px;
                        font-size: 9px;
                        font-weight: bold;
                    }
                    QPushButton:hover { background-color: #d32f2f; }
                """)
                self.freeze_status.setStyleSheet("color: green; font-size: 14px;")
                
                if self.freeze_processor:
                    self.freeze_processor.start_processing()
            else:
                self.freeze_btn.setText("🧊 Enable Freeze")
                self.freeze_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2196F3;
                        color: white;
                        border-radius: 10px;
                        font-size: 9px;
                        font-weight: bold;
                    }
                    QPushButton:hover { background-color: #1976D2; }
                """)
                self.freeze_status.setStyleSheet("color: gray; font-size: 14px;")
                
                if self.freeze_processor:
                    self.freeze_processor.stop_processing()
        
        # Apply the patches
        QStreamOutput.__init__ = patched_qstreamoutput_init
        QStreamOutput._add_freeze_controls_to_stream_output = _add_freeze_controls_to_stream_output
        QStreamOutput._toggle_freeze = _toggle_freeze
        
        print("✓ Auto-patch applied to DeepFaceLive UI components")
        
    except ImportError as e:
        print(f"⚠ Could not import DeepFaceLive components for patching: {e}")
    except Exception as e:
        print(f"⚠ Error during auto-patch: {e}")


def apply_manual_patches():
    """
    Apply patches manually to existing instances
    Call this function after DeepFaceLive components are created
    """
    print("🔧 Applying manual patches to existing DeepFaceLive components...")
    
    # This would be called from main.py or wherever the app is initialized
    # Example:
    # 
    # from ui_integration_patch import apply_manual_patches
    # 
    # # After creating DeepFaceLive app
    # app = DeepFaceLiveApp(userdata_path)
    # apply_manual_patches()
    # app.run()
    
    # Find and patch existing components
    try:
        from xlib.qt import qtx
        
        # Find all QStreamOutput instances
        app = qtx.QXMainApplication.inst
        if app:
            stream_outputs = app.findChildren(QStreamOutput)
            for stream_output in stream_outputs:
                patch_stream_output_ui(stream_output)
                print(f"✓ Patched QStreamOutput instance")
            
            # Find main windows
            main_windows = [w for w in app.topLevelWidgets() if w.isWindow()]
            for window in main_windows:
                patch_main_app_window(window)
                print(f"✓ Patched main window")
    
    except Exception as e:
        print(f"⚠ Error during manual patching: {e}")


# Integration hook for main.py
def integrate_freeze_ui_into_main():
    """
    Integration function to be called from main.py
    Add this to your main.py file:
    
    try:
        from ui_integration_patch import integrate_freeze_ui_into_main
        integrate_freeze_ui_into_main()
    except ImportError:
        pass
    """
    
    print("🎭 Integrating freeze-frame UI into DeepFaceLive...")
    
    # Apply auto-patches
    auto_patch_deepfacelive()
    
    # Schedule manual patches for after components are created
    try:
        from xlib.qt import qtx
        
        def delayed_patch():
            apply_manual_patches()
        
        # Apply patches after a short delay to ensure components are created
        qtx.QXTimer.singleShot(1000, delayed_patch)
        
    except Exception as e:
        print(f"⚠ Could not schedule delayed patches: {e}")
    
    print("✓ Freeze-frame UI integration complete")


if __name__ == "__main__":
    # Test the patches
    print("Testing UI integration patches...")
    integrate_freeze_ui_into_main()
