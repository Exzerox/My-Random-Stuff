import os
import sys
import subprocess
import platform
from pathlib import Path
import importlib.metadata
import pkg_resources

def setup_cuda_environment():
    print("Setting up CUDA environment...")
    
    # Get CUDA installation paths
    cuda_path = os.environ.get('CUDA_PATH', '')
    if not cuda_path:
        # Try to find CUDA installation
        possible_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cuda_path = path
                break
    
    if not cuda_path:
        print("Could not find CUDA installation. Please ensure CUDA is installed.")
        return False
    
    print(f"Found CUDA at: {cuda_path}")
    
    # Set environment variables
    os.environ['CUDA_PATH'] = cuda_path
    os.environ['PATH'] = f"{cuda_path}\\bin;{os.environ['PATH']}"
    os.environ['CUDA_HOME'] = cuda_path
    
    # Create Triton cache directory
    triton_cache = Path("./triton_cache")
    triton_cache.mkdir(exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(triton_cache.absolute())
    
    # Verify CUDA installation
    try:
        nvcc_version = subprocess.check_output([f"{cuda_path}\\bin\\nvcc.exe", "--version"]).decode('utf-8')
        print("\nCUDA Compiler Version:")
        print(nvcc_version)
    except Exception as e:
        print(f"Could not verify CUDA compiler: {e}")
        return False
    
    # Check if CUDA is accessible to Python
    try:
        import torch
        print("\nPyTorch CUDA Information:")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")
        return False

    # Check Whisper installation
    print("\nChecking Whisper installation...")
    try:
        # Try to import whisper directly
        import whisper
        print("Whisper module found!")
        
        # Try to get version from different sources
        try:
            version = importlib.metadata.version('whisper')
            print(f"Whisper Version (metadata): {version}")
        except:
            try:
                version = whisper.__version__
                print(f"Whisper Version (__version__): {version}")
            except:
                print("Could not determine Whisper version")
        
        # Check where whisper is installed
        print(f"Whisper location: {whisper.__file__}")
        
        # Try to load a small model to test CUDA
        print("\nTesting Whisper CUDA...")
        model = whisper.load_model("tiny", device="cuda")
        print("Successfully loaded Whisper model with CUDA")
        
        # Check if Triton is available
        try:
            import triton
            print(f"\nTriton Version: {triton.__version__}")
            print(f"Triton CUDA Backend Available: {triton.backends.cuda.is_available()}")
        except ImportError:
            print("Triton not installed")
        except Exception as e:
            print(f"Error checking Triton: {e}")
            
    except Exception as e:
        print(f"Error checking Whisper: {e}")
        print("\nTrying to find Whisper in installed packages...")
        try:
            for package in pkg_resources.working_set:
                if 'whisper' in package.key.lower():
                    print(f"Found related package: {package.key} {package.version}")
        except Exception as e:
            print(f"Error searching packages: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = setup_cuda_environment()
    if success:
        print("\nCUDA environment setup completed successfully!")
    else:
        print("\nCUDA environment setup failed. Please check the errors above.") 