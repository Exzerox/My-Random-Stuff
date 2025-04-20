import outetts
import torch
import os
from pathlib import Path

def setup_environment():
    # Set up CUDA environment
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path:
        os.environ['PATH'] = f"{cuda_path}\\bin;{os.environ['PATH']}"
        os.environ['CUDA_HOME'] = cuda_path
    
    # Set up Triton cache
    triton_cache = Path("./triton_cache")
    triton_cache.mkdir(exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(triton_cache.absolute())
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    setup_environment()
    
    # Initialize the interface with GPU-optimized configuration
    interface = outetts.Interface(
        config=outetts.ModelConfig(
            model_path="OuteAI/Llama-OuteTTS-1.0-1B",
            tokenizer_path="OuteAI/Llama-OuteTTS-1.0-1B",
            interface_version=outetts.InterfaceVersion.V3,
            backend=outetts.Backend.HF,
            additional_model_config={
                "device_map": "auto"  # Automatically use available GPU
            },
            device="cuda",
            dtype=torch.float16  # Use float16 for better GPU memory efficiency
        )
    )

    # Create a speaker profile from the provided audio file
    speaker = interface.create_speaker("CleanedUpVoiceShort.wav")

    # Generate speech with a simple test message
    output = interface.generate(
        config=outetts.GenerationConfig(
            text="Hey, I hope you have a wonderful day. I'm a voice assistant created by OuteAI. I'm here to help you with your questions and tasks.",
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(
                temperature=0.4
            ),
            additional_gen_config={
                "use_cache": True,  # Enable KV cache for faster generation
            }
        )
    )

    # Save the output to a file
    output.save("test_output.wav")
    print("Voice generation complete! Check test_output.wav for the result.")

if __name__ == "__main__":
    main() 