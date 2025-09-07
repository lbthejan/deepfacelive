"""
DeepFaceLive App Patch
Modifications to integrate freeze-frame functionality into the main DeepFaceLive application
This file shows how to modify the existing DeepFaceLive app structure
"""

import sys
from pathlib import Path

# Add our modules to path
sys.path.append(str(Path(__file__).parent))

from deepfacelive_integration import DeepFaceLiveFreezeProcessor, DeepFaceLiveAppIntegration


class DeepFaceLiveAppPatch:
    """
    Patch class to modify DeepFaceLive app for freeze-frame functionality
    """
    
    @staticmethod
    def patch_deepfacelive_app(original_app_class):
        """
        Patch the DeepFaceLive app class to include freeze-frame functionality
        
        Args:
            original_app_class: Original DeepFaceLiveApp class
            
        Returns:
            Modified class with freeze-frame integration
        """
        
        class PatchedDeepFaceLiveApp(original_app_class):
            def __init__(self, *args, **kwargs):
                # Initialize original app
                super().__init__(*args, **kwargs)
                
                # Initialize freeze processor
                userdata_path = kwargs.get('userdata_path', Path.cwd())
                self.freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)
                
                print("DeepFaceLive app patched with freeze-frame functionality")
            
            def on_initialize(self):
                """Override initialization to add freeze-frame setup"""
                # Call original initialization
                if hasattr(super(), 'on_initialize'):
                    super().on_initialize()
                
                # Initialize freeze processor with face detector
                self._setup_freeze_processor()
            
            def _setup_freeze_processor(self):
                """Setup freeze processor with app components"""
                try:
                    # Find face detector in app
                    face_detector = None
                    if hasattr(self, 'face_detector'):
                        face_detector = self.face_detector
                    elif hasattr(self, '_face_detector'):
                        face_detector = self._face_detector
                    elif hasattr(self, 'detector'):
                        face_detector = self.detector
                    
                    if face_detector:
                        self.freeze_processor.initialize_face_detector(face_detector)
                        print("Freeze processor connected to face detector")
                    else:
                        print("Warning: Could not find face detector for freeze processor")
                
                except Exception as e:
                    print(f"Error setting up freeze processor: {e}")
            
            def process_frame(self, input_frame):
                """
                Override frame processing to include freeze-frame logic
                
                Args:
                    input_frame: Input frame from webcam
                    
                Returns:
                    Processed frame with freeze-frame functionality
                """
                try:
                    # Get app components
                    face_swapper = getattr(self, 'face_swapper', None)
                    predictor = getattr(self, 'predictor', None) or getattr(self, 'face_predictor', None)
                    face_enhancer = getattr(self, 'face_enhancer', None)
                    
                    # Use freeze processor if available and components exist
                    if (self.freeze_processor and 
                        self.freeze_processor.face_detector_adapter and 
                        face_swapper):
                        
                        return self.freeze_processor.process_frame(
                            input_frame, face_swapper, predictor, face_enhancer
                        )
                    
                    # Fallback to original processing
                    elif hasattr(super(), 'process_frame'):
                        return super().process_frame(input_frame)
                    else:
                        return input_frame
                
                except Exception as e:
                    print(f"Error in patched frame processing: {e}")
                    # Fallback to original or return input frame
                    try:
                        if hasattr(super(), 'process_frame'):
                            return super().process_frame(input_frame)
                    except:
                        pass
                    return input_frame
            
            def on_ui_create(self):
                """Override UI creation to add freeze-frame controls"""
                # Call original UI creation
                if hasattr(super(), 'on_ui_create'):
                    super().on_ui_create()
                
                # Add freeze-frame UI controls
                self._add_freeze_controls()
            
            def _add_freeze_controls(self):
                """Add freeze-frame controls to UI"""
                try:
                    # This will depend on the UI framework used by DeepFaceLive
                    # Common patterns for different UI systems:
                    
                    # For Qt-based UI
                    if hasattr(self, 'add_slider_control'):
                        self.add_slider_control(
                            "Freeze Threshold",
                            min_val=30, max_val=95, value=75,
                            callback=lambda val: self.freeze_processor.update_freeze_threshold(val/100.0)
                        )
                        
                        self.add_button_control(
                            "Toggle Stats",
                            callback=self.freeze_processor.toggle_stats_overlay
                        )
                    
                    # For Tkinter-based UI
                    elif hasattr(self, 'create_scale_widget'):
                        self.create_scale_widget(
                            label="Freeze Threshold",
                            from_=0.3, to=0.95, resolution=0.01,
                            command=self.freeze_processor.update_freeze_threshold
                        )
                    
                    # For custom UI system
                    elif hasattr(self, 'ui_manager'):
                        ui = self.ui_manager
                        ui.add_control('freeze_threshold', 'slider', 
                                     min=0.3, max=0.95, value=0.75,
                                     callback=self.freeze_processor.update_freeze_threshold)
                        ui.add_control('stats_toggle', 'button',
                                     text="Toggle Stats",
                                     callback=self.freeze_processor.toggle_stats_overlay)
                    
                    print("Freeze-frame UI controls added")
                    
                except Exception as e:
                    print(f"Could not add freeze-frame UI controls: {e}")
                    print("You may need to add controls manually through the app interface")
            
            def on_destroy(self):
                """Override cleanup to include freeze processor cleanup"""
                # Cleanup freeze processor
                if hasattr(self, 'freeze_processor'):
                    self.freeze_processor.cleanup()
                
                # Call original cleanup
                if hasattr(super(), 'on_destroy'):
                    super().on_destroy()
            
            # Add convenience methods for freeze control
            def set_freeze_threshold(self, threshold: float):
                """Set freeze threshold"""
                if hasattr(self, 'freeze_processor'):
                    self.freeze_processor.update_freeze_threshold(threshold)
            
            def get_freeze_stats(self):
                """Get freeze statistics"""
                if hasattr(self, 'freeze_processor'):
                    return self.freeze_processor.get_stats()
                return {}
            
            def print_performance_summary(self):
                """Print performance summary"""
                if hasattr(self, 'freeze_processor'):
                    self.freeze_processor.print_stats_summary()
        
        return PatchedDeepFaceLiveApp


# Alternative integration approach: Monkey patching
class MonkeyPatchIntegration:
    """
    Alternative integration using monkey patching for existing installations
    """
    
    @staticmethod
    def patch_existing_app(app_instance, userdata_path=None):
        """
        Patch an existing app instance with freeze-frame functionality
        
        Args:
            app_instance: Existing DeepFaceLive app instance
            userdata_path: Path to user data directory
        """
        # Add freeze processor to existing instance
        app_instance.freeze_processor = DeepFaceLiveFreezeProcessor(userdata_path)
        
        # Store original process_frame method
        original_process_frame = getattr(app_instance, 'process_frame', None)
        
        def patched_process_frame(input_frame):
            """Patched frame processing method"""
            try:
                # Get app components
                face_swapper = getattr(app_instance, 'face_swapper', None)
                predictor = getattr(app_instance, 'predictor', None)
                face_enhancer = getattr(app_instance, 'face_enhancer', None)
                
                # Use freeze processor if available
                if (app_instance.freeze_processor and 
                    app_instance.freeze_processor.face_detector_adapter and 
                    face_swapper):
                    
                    return app_instance.freeze_processor.process_frame(
                        input_frame, face_swapper, predictor, face_enhancer
                    )
                
                # Fallback to original processing
                elif original_process_frame:
                    return original_process_frame(input_frame)
                else:
                    return input_frame
            
            except Exception as e:
                print(f"Error in patched processing: {e}")
                if original_process_frame:
                    return original_process_frame(input_frame)
                return input_frame
        
        # Replace process_frame method
        app_instance.process_frame = patched_process_frame
        
        # Initialize face detector connection
        MonkeyPatchIntegration._connect_face_detector(app_instance)
        
        print("App instance monkey-patched with freeze-frame functionality")
    
    @staticmethod
    def _connect_face_detector(app_instance):
        """Connect freeze processor to face detector"""
        try:
            # Find face detector
            face_detector = None
            for attr_name in ['face_detector', '_face_detector', 'detector']:
                if hasattr(app_instance, attr_name):
                    face_detector = getattr(app_instance, attr_name)
                    break
            
            if face_detector:
                app_instance.freeze_processor.initialize_face_detector(face_detector)
                print("Freeze processor connected to face detector")
            else:
                print("Warning: Could not find face detector")
        
        except Exception as e:
            print(f"Error connecting face detector: {e}")


# Example usage for different integration scenarios
def integrate_with_main_py():
    """
    Example of how to integrate with the main.py file
    """
    
    # Modify the run_DeepFaceLive function in main.py
    def modified_run_DeepFaceLive(args):
        userdata_path = Path(args.userdata_dir) if args.userdata_dir else Path.cwd()
        
        # Import and patch the app class
        from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
        
        # Apply patch
        PatchedApp = DeepFaceLiveAppPatch.patch_deepfacelive_app(DeepFaceLiveApp)
        
        # Run patched app
        print('Running DeepFaceLive with freeze-frame functionality.')
        app = PatchedApp(userdata_path=userdata_path)
        app.run()
    
    return modified_run_DeepFaceLive


def integrate_with_existing_installation():
    """
    Example of how to integrate with an existing DeepFaceLive installation
    """
    
    # For existing installations, you can monkey patch at runtime
    def patch_at_runtime():
        try:
            # Import existing app
            from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
            
            # Get or create app instance
            app = DeepFaceLiveApp(userdata_path=Path.cwd())
            
            # Apply monkey patch
            MonkeyPatchIntegration.patch_existing_app(app)
            
            # Run app
            app.run()
            
        except ImportError as e:
            print(f"Could not import DeepFaceLive app: {e}")
            print("Make sure DeepFaceLive is properly installed")
    
    return patch_at_runtime


# Configuration file for easy setup
def create_integration_config():
    """Create configuration file for integration"""
    config = {
        "integration_method": "patch_class",  # or "monkey_patch"
        "freeze_settings": {
            "confidence_threshold": 0.75,
            "max_freeze_duration": 5.0,
            "frame_buffer_size": 3,
            "enable_stats_overlay": True
        },
        "ui_settings": {
            "show_threshold_slider": True,
            "show_stats_button": True,
            "show_debug_button": True
        }
    }
    
    import json
    with open("freeze_integration_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Integration configuration saved to freeze_integration_config.json")


if __name__ == "__main__":
    # Create configuration file
    create_integration_config()
    
    # Example usage
    print("DeepFaceLive Freeze-Frame Integration Patch")
    print("Choose integration method:")
    print("1. Patch app class (recommended for new installations)")
    print("2. Monkey patch existing instance (for existing installations)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        modified_runner = integrate_with_main_py()
        print("App class patch ready. Modify your main.py to use the modified_run_DeepFaceLive function.")
    elif choice == "2":
        runtime_patcher = integrate_with_existing_installation()
        print("Runtime patcher ready. Call the returned function to run with freeze-frame.")
        # runtime_patcher()  # Uncomment to run immediately
    else:
        print("Invalid choice. Please run again and select 1 or 2.")