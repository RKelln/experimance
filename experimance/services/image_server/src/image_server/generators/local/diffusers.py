import torch
import time
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    LCMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.models.attention_processor import AttnProcessor2_0
import numpy as np
from PIL import Image
import gc

"""

https://huggingface.co/docs/diffusers/en/optimization/xformers
$ pip install xformers

https://huggingface.co/docs/diffusers/en/optimization/deepcache
$ pip install DeepCache

"""

# from DeepCache import DeepCacheSDHelper
# helper = DeepCacheSDHelper(pipe=pipe)
# helper.set_params(
#     cache_interval=3,  # higher numbers are faster
#     cache_branch_id=0, # lower numbers are faster
# )
# helper.enable()


class OptimizedSDXLGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_id: str = "diffusers/controlnet-depth-sdxl-1.0",
        use_lightning: bool = True,
        lightning_lora_path: str = None,
        style_lora_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        enable_xformers: bool = True,
        compile_unet: bool = True
    ):
        """
        Initialize the optimized SDXL pipeline with speed optimizations.
        
        Args:
            model_id: SDXL base model or lightning model
            controlnet_id: ControlNet model for depth conditioning
            use_lightning: Whether to use Lightning optimizations
            lightning_lora_path: Path to Lightning LoRA (if not using lightning base model)
            style_lora_path: Path to style LoRA
            device: Device to run on
            dtype: Data type for models
            enable_cpu_offload: Enable CPU offloading for VRAM savings
            enable_attention_slicing: Enable attention slicing
            enable_xformers: Enable xFormers memory efficient attention
            compile_unet: Compile UNet with torch.compile for speed
        """
        self.device = device
        self.dtype = dtype
        self.use_lightning = use_lightning
        
        print("Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        print("Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        # Apply optimizations
        self._apply_optimizations(
            enable_cpu_offload, enable_attention_slicing, 
            enable_xformers, compile_unet
        )
        
        # Load LoRAs
        if lightning_lora_path:
            print("Loading Lightning LoRA...")
            self.pipe.load_lora_weights(lightning_lora_path, adapter_name="lightning")
        
        if style_lora_path:
            print("Loading Style LoRA...")
            self.pipe.load_lora_weights(style_lora_path, adapter_name="style")
        
        # Set optimal scheduler
        self._setup_scheduler()

        print("Pipeline ready!")
    
    def _apply_optimizations(self, cpu_offload, attention_slicing, xformers, compile_unet):
        """Apply various speed and memory optimizations."""
        
        if cpu_offload:
            print("Enabling CPU offload...")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        
        if attention_slicing:
            print("Enabling attention slicing...")
            self.pipe.enable_attention_slicing(1)
        
        if xformers:
            try:
                print("Enabling xFormers...")
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"xFormers not available: {e}")
        
        # Use efficient attention processors
        print("Setting efficient attention processors...")
        self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        
        if compile_unet:
            try:
                print("Compiling UNet with torch.compile...")
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead", 
                    fullgraph=True
                )
            except Exception as e:
                print(f"UNet compilation failed: {e}")
        
        # Enable VAE slicing for memory efficiency
        self.pipe.enable_vae_slicing()
        
        # Reduce VRAM usage
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        # https://huggingface.co/docs/diffusers/en/optimization/fp16#tensorfloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        
    
    def _setup_scheduler(self):
        """Setup the optimal scheduler based on model type."""
        if self.use_lightning:
            # Lightning models work best with LCM scheduler at low steps
            self.pipe.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            self.default_steps = 4
            self.guidance_scale = 1.0
        else:
            # For regular models, use DPM++ 2M for speed
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            self.default_steps = 20
            self.guidance_scale = 7.5
    
    def prepare_depth_image(self, depth_array: np.ndarray) -> Image.Image:
        """
        Prepare depth image for ControlNet.
        
        Args:
            depth_array: Numpy array of depth values
            
        Returns:
            PIL Image of depth map
        """
        # Normalize depth to 0-255 range
        depth_normalized = ((depth_array - depth_array.min()) / 
                          (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        
        return Image.fromarray(depth_normalized).convert("RGB").resize((1024, 1024))
    
    def generate(
        self,
        prompt: str,
        depth_image: Image.Image,
        negative_prompt: str = "",
        seed: int = None,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        controlnet_conditioning_scale: float = 1.0,
        style_lora_scale: float = 1.0,
        lightning_lora_scale: float = 1.0,
        width: int = 1024,
        height: int = 1024
    ) -> tuple[Image.Image, float]:
        """
        Generate an image with the optimized pipeline.
        
        Args:
            prompt: Text prompt
            depth_image: Depth conditioning image
            negative_prompt: Negative prompt
            seed: Random seed
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet influence strength
            style_lora_scale: Style LoRA influence
            lightning_lora_scale: Lightning LoRA influence
            width: Output width
            height: Output height
            
        Returns:
            Tuple of (generated_image, generation_time)
        """
        # Set defaults
        if num_inference_steps is None:
            num_inference_steps = self.default_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Set LoRA scales
        lora_scale = {}
        if hasattr(self.pipe, 'get_active_adapters'):
            if "lightning" in self.pipe.get_active_adapters():
                lora_scale["lightning"] = lightning_lora_scale
            if "style" in self.pipe.get_active_adapters():
                lora_scale["style"] = style_lora_scale
        
        # Warm up GPU if first run
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                width=width,
                height=height,
                cross_attention_kwargs={"scale": lora_scale} if lora_scale else None,
                output_type="pil"
            ).images[0]
        
        generation_time = time.time() - start_time
        
        # Clean up
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        return image, generation_time
    
    def benchmark(self, prompt: str, depth_image: Image.Image, num_runs: int = 5):
        """Run benchmark to measure average generation time."""
        times = []
        
        print(f"Running benchmark with {num_runs} generations...")
        
        for i in range(num_runs):
            _, gen_time = self.generate(
                prompt=prompt,
                depth_image=depth_image,
                seed=42 + i
            )
            times.append(gen_time)
            print(f"Run {i+1}: {gen_time:.2f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nBenchmark Results:")
        print(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"Fastest time: {min(times):.2f}s")
        print(f"Slowest time: {max(times):.2f}s")
        
        return avg_time, std_time

# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = OptimizedSDXLGenerator(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",  # or use a lightning model
        controlnet_id="diffusers/controlnet-depth-sdxl-1.0",
        use_lightning=True,
        lightning_lora_path="ByteDance/SDXL-Lightning",  # Example Lightning LoRA
        style_lora_path=None,  # Add your style LoRA path here
        compile_unet=True
    )
    
    # Create a dummy depth image for testing
    dummy_depth = np.random.rand(512, 512) * 255
    depth_img = generator.prepare_depth_image(dummy_depth)
    
    # Generate an image
    prompt = "a beautiful landscape with mountains and a lake, photorealistic, detailed"
    negative_prompt = "blurry, low quality, distorted"
    
    image, gen_time = generator.generate(
        prompt=prompt,
        depth_image=depth_img,
        negative_prompt=negative_prompt,
        seed=42
    )
    
    print(f"Generated image in {gen_time:.2f} seconds")
    image.save("output.png")
    
    # Run benchmark
    generator.benchmark(prompt, depth_img, num_runs=3)