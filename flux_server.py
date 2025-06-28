from io import BytesIO
from fastapi import Response, HTTPException
from pydantic import BaseModel, Field
import torch
import time
import os
import litserve as ls
from optimum.quanto import freeze, qfloat8, quantize
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import logging
from typing import Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    num_inference_steps: int = Field(default=4, ge=1, le=50, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0, description="Guidance scale")
    width: int = Field(default=1024, ge=512, le=2048, description="Image width")
    height: int = Field(default=1024, ge=512, le=2048, description="Image height")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class FluxLitAPI(ls.LitAPI):
    def __init__(self, quantized_cache_dir: str = "./quantized_models"):
        super().__init__()
        self.quantized_cache_dir = Path(quantized_cache_dir)
        self.quantized_cache_dir.mkdir(exist_ok=True)
        
    def _get_quantized_model_path(self, model_name: str) -> Path:
        """Get the path for cached quantized model"""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return self.quantized_cache_dir / f"{safe_name}_quantized"
    
    def _save_quantized_model(self, model, model_name: str):
        """Save quantized model to cache"""
        try:
            cache_path = self._get_quantized_model_path(model_name)
            cache_path.mkdir(exist_ok=True)
            model.save_pretrained(cache_path)
            logger.info(f"Saved quantized {model_name} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save quantized {model_name}: {e}")
    
    def _load_quantized_model(self, model_class, model_name: str, **kwargs):
        """Load quantized model from cache or create and cache it"""
        cache_path = self._get_quantized_model_path(model_name)
        
        if cache_path.exists() and any(cache_path.iterdir()):
            try:
                logger.info(f"Loading cached quantized {model_name} from {cache_path}")
                model = model_class.from_pretrained(cache_path, **kwargs)
                # Models loaded from cache should already be quantized and frozen
                return model
            except Exception as e:
                logger.warning(f"Failed to load cached {model_name}, will recreate: {e}")
        
        # Load original model, quantize, and cache
        logger.info(f"Loading and quantizing {model_name}...")
        model = model_class.from_pretrained(model_name, **kwargs)
        
        # Quantize and freeze
        quantize(model, weights=qfloat8)
        freeze(model)
        
        # Cache for future use
        self._save_quantized_model(model, model_name)
        
        return model

    def setup(self, device):
        try:
            logger.info("Loading Flux model components...")
            
            # Load scheduler (no quantization needed)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                subfolder="scheduler"
            )
            
            # Load CLIP text encoder (no quantization for this one)
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", 
                torch_dtype=torch.bfloat16
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            
            # Load VAE (no quantization needed)
            vae = AutoencoderKL.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                subfolder="vae", 
                torch_dtype=torch.bfloat16
            )
            
            # Load tokenizer_2 (no quantization needed)
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                subfolder="tokenizer_2"
            )
            
            # Load T5 text encoder with caching
            text_encoder_2 = self._load_quantized_model(
                T5EncoderModel,
                "black-forest-labs/FLUX.1-dev",
                subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16
            )
            
            # Load transformer with caching
            transformer = self._load_quantized_model(
                FluxTransformer2DModel,
                "black-forest-labs/FLUX.1-dev", 
                subfolder="transformer",
                torch_dtype=torch.bfloat16
            )

            logger.info("Creating pipeline...")
            self.pipe = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=transformer,
            )
            
            self.pipe.enable_model_cpu_offload()
            logger.info("Model setup complete!")
            
        except Exception as e:
            logger.error(f"Error during model setup: {str(e)}")
            raise e

    def decode_request(self, request):
        try:
            if isinstance(request, dict):
                flux_request = FluxRequest(**request)
            else:
                flux_request = request
            
            logger.info(f"Processing request: {flux_request.prompt[:50]}...")
            return flux_request
            
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    def predict(self, flux_request: FluxRequest):
        try:
            logger.info(f"Generating image with steps={flux_request.num_inference_steps}, guidance={flux_request.guidance_scale}")
            
            generator = torch.Generator()
            if flux_request.seed is not None:
                generator.manual_seed(flux_request.seed)
            else:
                generator.manual_seed(int(time.time()))
            
            image = self.pipe(
                prompt=flux_request.prompt,
                width=flux_request.width,
                height=flux_request.height,
                num_inference_steps=flux_request.num_inference_steps,
                generator=generator,
                guidance_scale=flux_request.guidance_scale,
            ).images[0]

            logger.info("Image generation complete!")
            return image
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    def encode_response(self, image):
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            buffered.seek(0)
            
            return Response(
                content=buffered.getvalue(), 
                media_type="image/png",
                headers={
                    "Content-Type": "image/png",
                    "Content-Disposition": "inline; filename=generated_image.png"
                }
            )
            
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

# Starting the server
if __name__ == "__main__":
    # You can specify a custom cache directory
    api = FluxLitAPI(quantized_cache_dir="./flux_quantized_cache")
    server = ls.LitServer(api, timeout=False)
    server.run(port=8000)