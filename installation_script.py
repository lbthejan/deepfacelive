#!/usr/bin/env python3
"""
Installation Script for DeepFaceLive Freeze-Frame Feature
Automatically installs and configures the freeze-frame functionality
"""

import os
import sys
import shutil
import json
from pathlib import Path
import argparse


class FreezeFrameInstaller:
    def __init__(self, deepfacelive_path: Path):
        """
        Initialize installer
        
        Args:
            deepfacelive_path: Path to DeepFaceLive installation
        """
        self.deepfacelive_path = Path(deepfacelive_path)
        self.backup_path = self.deepfacelive_path / "backup_original"
        self.freeze_modules_path = self.deepfacelive_path / "freeze_frame_modules"
        
        # Files to install
        self.freeze_files = [
            "freeze_frame_manager.py",
            "face_detector_wrapper.py", 
            "performance_monitor.py",
            "deepfacelive_integration.py",
            "deepfacelive_app_patch.py"
        ]
        
        # Config files
        self.config_files = [
            "freeze_config.json",
            "freeze_integration_config.json",
            "detector_config.json"
        ]
    
    def validate_installation(self):
        """Validate DeepFaceLive installation"""
        print("Validating DeepFaceLive installation...")
        
        # Check if main.py exists
        main_py = self.deepfacelive_path / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(f"main.py not found in {self.deepfacelive_path}")
        
        # Check for apps directory
        apps_dir = self.deepfacelive_path / "apps"
        if not apps_dir.exists():
            raise FileNotFoundError(f"apps directory not found in {self.deepfacelive_path}")
        
        # Check for DeepFaceLive app
        dfl_app = apps_dir / "DeepFaceLive"
        if not dfl_app.exists():
            raise FileNotFoundError(f"DeepFaceLive app not found in {apps_dir}")
        
        print("✓ DeepFaceLive installation validated")
    
    def create_backup(self):
        """Create backup of original files"""
        print("Creating backup of original files...")
        
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
        
        self.backup_path.mkdir(exist_ok=True)
        
        # Backup main.py
        main_py = self.deepfacelive_path / "main.py"
        if main_py.exists():
            shutil.copy2(main_py, self.backup_path / "main.py")
        
        # Backup app files
        app_backup = self.backup_path / "apps"
        app_backup.mkdir(exist_ok=True)
        
        apps_dir = self.deepfacelive_path / "apps"
        if apps_dir.exists():
            shutil.copytree(apps_dir, app_backup, dirs_exist_ok=True)
        
        print(f"✓ Backup created at {self.backup_path}")
    
    def install_freeze_modules(self):
        """Install freeze-frame modules"""
        print("Installing freeze-frame modules...")
        
        # Create modules directory
        self.freeze_modules_path.mkdir(exist_ok=True)
        
        # Copy freeze-frame files (these would be the actual files from artifacts)
        current_dir = Path(__file__).parent
        
        for file_name in self.freeze_files:
            src_file = current_dir / file_name
            dst_file = self.freeze_modules_path / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"✓ Installed {file_name}")
            else:
                print(f"⚠ Warning: {file_name} not found, creating placeholder")
                self.create_placeholder_file(dst_file, file_name)
        
        # Create __init__.py for module
        init_file = self.freeze_modules_path / "__init__.py"
        with open(init_file, 'w') as f:
            f.write('"""DeepFaceLive Freeze-Frame Module"""\n')
        
        print(f"✓ Freeze-frame modules installed at {self.freeze_modules_path}")
    
    def create_placeholder_file(self, file_path: Path, file_name: str):
        """Create placeholder file with basic structure"""
        placeholders = {
            "freeze_frame_manager.py": '''"""Placeholder for freeze_frame_manager.py"""
class FreezeFrameManager:
    def __init__(self, confidence_threshold=0.75, config_file="freeze_config.json"):
        print("Please replace this placeholder with the actual freeze_frame_manager.py file")
        pass
''',
            "face_detector_wrapper.py": '''"""Placeholder for face_detector_wrapper.py"""
class FaceDetectorWrapper:
    def __init__(self, original_detector=None):
        print("Please replace this placeholder with the actual face_detector_wrapper.py file")
        pass
''',
            "performance_monitor.py": '''"""Placeholder for performance_monitor.py"""
class PerformanceMonitor:
    def __init__(self, window_size=30):
        print("Please replace this placeholder with the actual performance_monitor.py file")
        pass
''',
            "deepfacelive_integration.py": '''"""Placeholder for deepfacelive_integration.py"""
class DeepFaceLiveFreezeProcessor:
    def __init__(self, userdata_path=None):
        print("Please replace this placeholder with the actual deepfacelive_integration.py file")
        pass
''',
            "deepfacelive_app_patch.py": '''"""Placeholder for deepfacelive_app_patch.py"""
class DeepFaceLiveAppPatch:
    @staticmethod
    def patch_deepfacelive_app(original_app_class):
        print("Please replace this placeholder with the actual deepfacelive_app_patch.py file")
        return original_app_class
'''
        }
        
        content = placeholders.get(file_name, f'"""Placeholder for {file_name}"""')
        with open(file_path, 'w') as f:
            f.write(content)
    
    def install_config_files(self):
        """Install configuration files"""
        print("Installing configuration files...")
        
        configs = {
            "freeze_config.json": {
                "confidence_threshold": 0.75,
                "max_freeze_duration": 5.0,
                "frame_buffer_size": 3,
                "enable_stats_overlay": True,
                "enable_performance_monitoring": True
            },
            "freeze_integration_config.json": {
                "integration_method": "patch_class",
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
            },
            "detector_config.json": {
                "detector_type": "insightface",
                "confidence_extraction": {
                    "method": "auto_detect",
                    "fallback_confidence": 0.5
                }
            }
        }
        
        for config_name, config_data in configs.items():
            config_path = self.deepfacelive_path / config_name
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✓ Installed {config_name}")
    
    def patch_main_py(self):
        """Patch main.py to include freeze-frame functionality"""
        print("Patching main.py...")
        
        main_py = self.deepfacelive_path / "main.py"
        
        # Read original main.py
        with open(main_py, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "freeze_frame_modules" in content:
            print("✓ main.py already patched")
            return
        
        # Add import for freeze-frame modules
        import_addition = '''
# Freeze-frame functionality
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "freeze_frame_modules"))

try:
    from deepfacelive_app_patch import DeepFaceLiveAppPatch
    FREEZE_FRAME_AVAILABLE = True
except ImportError:
    print("Warning: Freeze-frame modules not found. Running without freeze-frame functionality.")
    FREEZE_FRAME_AVAILABLE = False
'''
        
        # Find the import section and add our import
        lines = content.split('\n')
        import_index = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('from xlib'):
                import_index = i + 1
                break
        
        if import_index > 0:
            lines.insert(import_index, import_addition)
        else:
            # Add at the beginning after existing imports
            lines.insert(5, import_addition)
        
        # Modify the run_DeepFaceLive function
        modified_function = '''
    def run_DeepFaceLive(args):
        userdata_path = Path(args.userdata_dir)
        lib_appargs.set_arg_bool('NO_CUDA', args.no_cuda)

        print('Running DeepFaceLive with freeze-frame functionality.')
        from apps.DeepFaceLive.DeepFaceLiveApp import DeepFaceLiveApp
        
        # Apply freeze-frame patch if available
        if FREEZE_FRAME_AVAILABLE:
            try:
                PatchedApp = DeepFaceLiveAppPatch.patch_deepfacelive_app(DeepFaceLiveApp)
                PatchedApp(userdata_path=userdata_path).run()
            except Exception as e:
                print(f"Error with freeze-frame functionality: {e}")
                print("Falling back to original DeepFaceLive...")
                DeepFaceLiveApp(userdata_path=userdata_path).run()
        else:
            DeepFaceLiveApp(userdata_path=userdata_path).run()
'''
        
        # Replace the original function
        new_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            if line.strip().startswith('def run_DeepFaceLive(args):'):
                new_lines.extend(modified_function.strip().split('\n'))
                in_function = True
                # Determine indentation level
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if in_function:
                # Check if we're still in the function
                current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
                if line.strip() and current_indent <= indent_level:
                    in_function = False
                    new_lines.append(line)
                # Skip lines within the original function
                elif not in_function:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Write modified content
        with open(main_py, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✓ main.py patched successfully")
    
    def create_startup_script(self):
        """Create startup script with freeze-frame"""
        script_content = '''#!/usr/bin/env python3
"""
DeepFaceLive with Freeze-Frame Functionality
Startup script for the enhanced version
"""

import sys
from pathlib import Path

# Add freeze-frame modules to path
sys.path.append(str(Path(__file__).parent / "freeze_frame_modules"))

def main():
    """Main entry point"""
    # Import and run the patched main
    from main import main as original_main
    original_main()

if __name__ == "__main__":
    main()
'''
        
        script_path = self.deepfacelive_path / "run_with_freeze_frame.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        
        print(f"✓ Startup script created: {script_path}")
    
    def verify_installation(self):
        """Verify the installation"""
        print("Verifying installation...")
        
        # Check if modules exist
        for file_name in self.freeze_files:
            file_path = self.freeze_modules_path / file_name
            if not file_path.exists():
                print(f"⚠ Warning: {file_name} not found")
            else:
                print(f"✓ {file_name} installed")
        
        # Check config files
        for config_name in self.config_files:
            config_path = self.deepfacelive_path / config_name
            if config_path.exists():
                print(f"✓ {config_name} installed")
        
        print("✓ Installation verification complete")
    
    def install(self, create_backup=True):
        """Run complete installation"""
        print("Starting DeepFaceLive Freeze-Frame installation...")
        
        try:
            # Validate
            self.validate_installation()
            
            # Backup
            if create_backup:
                self.create_backup()
            
            # Install modules
            self.install_freeze_modules()
            
            # Install configs
            self.install_config_files()
            
            # Patch main.py
            self.patch_main_py()
            
            # Create startup script
            self.create_startup_script()
            
            # Verify
            self.verify_installation()
            
            print("\n" + "="*50)
            print("INSTALLATION COMPLETE!")
            print("="*50)
            print("\nNext steps:")
            print("1. Copy the actual freeze-frame module files to:")
            print(f"   {self.freeze_modules_path}")
            print("2. Run DeepFaceLive using:")
            print(f"   python {self.deepfacelive_path}/main.py run DeepFaceLive")
            print("   OR")
            print(f"   python {self.deepfacelive_path}/run_with_freeze_frame.py run DeepFaceLive")
            print("\nConfiguration files:")
            print(f"- {self.deepfacelive_path}/freeze_config.json")
            print(f"- {self.deepfacelive_path}/freeze_integration_config.json")
            print(f"\nBackup location: {self.backup_path}")
            
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            print("Please check the error and try again.")
            raise
    
    def uninstall(self):
        """Uninstall freeze-frame functionality"""
        print("Uninstalling freeze-frame functionality...")
        
        # Restore from backup
        if self.backup_path.exists():
            main_backup = self.backup_path / "main.py"
            if main_backup.exists():
                shutil.copy2(main_backup, self.deepfacelive_path / "main.py")
                print("✓ main.py restored from backup")
        
        # Remove freeze modules
        if self.freeze_modules_path.exists():
            shutil.rmtree(self.freeze_modules_path)
            print("✓ Freeze-frame modules removed")
        
        # Remove config files
        for config_name in self.config_files:
            config_path = self.deepfacelive_path / config_name
            if config_path.exists():
                config_path.unlink()
                print(f"✓ {config_name} removed")
        
        # Remove startup script
        script_path = self.deepfacelive_path / "run_with_freeze_frame.py"
        if script_path.exists():
            script_path.unlink()
            print("✓ Startup script removed")
        
        print("✓ Uninstallation complete")


def main():
    parser = argparse.ArgumentParser(description="Install freeze-frame functionality for DeepFaceLive")
    parser.add_argument("deepfacelive_path", help="Path to DeepFaceLive installation")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall freeze-frame functionality")
    
    args = parser.parse_args()
    
    installer = FreezeFrameInstaller(args.deepfacelive_path)
    
    if args.uninstall:
        installer.uninstall()
    else:
        installer.install(create_backup=not args.no_backup)


if __name__ == "__main__":
    main()