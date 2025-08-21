"""
Flux Image Generation API using LitServe

This module implements a FastAPI-based image generation service using the FLUX.1-dev model
from Black Forest Labs. The service uses 8-bit quantization to optimize memory usage
and enable deployment on lower-end GPUs like the T4 or L4.

Key Features:
- FLUX.1-dev model for high-quality image generation
- 8-bit quantization for memory efficiency
- Customizable generation parameters
- RESTful API interface via LitServe
- Comprehensive error handling and logging
"""

from io import BytesIO
from fastapi import Response, HTTPException
from pydantic import BaseModel, Field
import torch
import time
import litserve as ls
from optimum.quanto import freeze, qfloat8, quantize
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import logging
from typing import Optional

# Configure logging to provide detailed information about the API operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxRequest(BaseModel):
    """
    Pydantic model defining the request structure for image generation.
    
    This model validates incoming requests and provides default values
    for optional parameters. All fields include validation constraints
    to ensure reasonable values are provided.
    """
    
    # Main text prompt that describes the desired image
    prompt: str = Field(..., description="Text prompt for image generation")
    
    # TODO: Uncomment and implement negative prompts if needed for better control
    # negative_prompt: str = Field(default=None, description="Negative text prompt for image generation")
    
    # Number of denoising steps - higher values generally produce better quality but take longer
    # Range: 1-50 steps (4 is optimized for speed while maintaining quality)
    num_inference_steps: int = Field(default=4, ge=1, le=50, description="Number of inference steps")
    
    # Guidance scale controls how closely the model follows the prompt
    # Higher values = more adherence to prompt, lower values = more creative freedom
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0, description="Guidance scale")
    
    # Image dimensions - constrained to reasonable ranges for memory and quality
    # Both width and height must be between 512-2048 pixels
    width: int = Field(default=1024, ge=512, le=2048, description="Image width")
    height: int = Field(default=1024, ge=512, le=2048, description="Image height")
    
    # Optional seed for reproducible generation - if None, uses current timestamp
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class FluxLitAPI(ls.LitAPI):
    """
    Main API class implementing the LitServe interface for FLUX image generation.
    
    This class handles the complete lifecycle of the image generation service:
    - Model loading and quantization during setup
    - Request validation and processing
    - Image generation using the FLUX pipeline
    - Response encoding and error handling
    """
    
    def setup(self, device):
        """
        Initialize and configure the FLUX model pipeline with memory-efficient loading.
        
        This method is called once when the server starts. It loads all required
        model components, applies 8-bit quantization for memory efficiency,
        and sets up the complete generation pipeline. This version loads large
        components sequentially to avoid VRAM overflow on GPUs like the T4.
        
        Args:
            device: The device to run the model on (automatically handled by LitServe)
            
        Raises:
            Exception: If any component fails to load or configure
        """
        try:
            logger.info("Loading Flux model components sequentially to conserve VRAM...")
            
            # Define the main model ID and data type
            model_id = "black-forest-labs/FLUX.1-Krea-dev"
            dtype = torch.bfloat16

            # --- Load, Quantize, and Offload Large Components One-by-One ---

            # 1. Load and process the main Transformer
            logger.info("Loading and quantizing transformer...")
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=dtype
            )
            quantize(transformer, weights=qfloat8)
            freeze(transformer)
            transformer.to("cpu") # Move to CPU to free up VRAM
            torch.cuda.empty_cache() # Clear cache

            # 2. Load and process the T5 Text Encoder
            logger.info("Loading and quantizing text_encoder_2...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder_2", torch_dtype=dtype
            )
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)
            text_encoder_2.to("cpu") # Move to CPU
            torch.cuda.empty_cache()

            # --- Load Smaller Components ---
            logger.info("Loading remaining components...")
            
            # VAE can also be large, so we load it last before creating the pipeline
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)

            # Smaller components that should fit in memory
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")

            logger.info("All components loaded. Creating pipeline...")
            
            # Create the complete FLUX pipeline with all components
            self.pipe = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=transformer,
            )
            
            # Now that the pipeline is built, enable model offloading for inference
            self.pipe.enable_model_cpu_offload()
            
            logger.info("Model setup complete and ready for inference!")
            
        except Exception as e:
            logger.error(f"Error during model setup: {str(e)}")
            raise e

    def decode_request(self, request):
        """
        Parse and validate incoming requests.
        
        Converts the raw request data into a validated FluxRequest object,
        ensuring all parameters are within acceptable ranges and properly formatted.
        
        Args:
            request: Raw request data (dict or FluxRequest object)
            
        Returns:
            FluxRequest: Validated request object
            
        Raises:
            HTTPException: If request validation fails
        """
        try:
            # Handle both dictionary and object inputs
            if isinstance(request, dict):
                flux_request = FluxRequest(**request)
            else:
                flux_request = request
            
            # Log the first 50 characters of the prompt for debugging
            logger.info(f"Processing request: {flux_request.prompt[:50]}...")
            return flux_request
            
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid request format: {str(e)}")

    def predict(self, flux_request: FluxRequest):
        """
        Generate an image based on the provided request parameters.
        
        This method handles the core image generation process, including:
        - Setting up the random number generator for reproducibility
        - Configuring generation parameters
        - Running the FLUX pipeline to create the image
        
        Args:
            flux_request (FluxRequest): Validated request containing generation parameters
            
        Returns:
            PIL.Image: Generated image object
            
        Raises:
            HTTPException: If image generation fails
        """
        try:
            logger.info(f"Generating image with steps={flux_request.num_inference_steps}, guidance={flux_request.guidance_scale}")
            
            # Set up the random number generator for reproducible results
            generator = torch.Generator()
            if flux_request.seed is not None:
                # Use provided seed for reproducible generation
                generator.manual_seed(flux_request.seed)
            else:
                # Use current timestamp as seed if none provided
                generator.manual_seed(int(time.time()))
            
            # Run the FLUX pipeline to generate the image
            # This is where the actual image generation happens
            image = self.pipe(
                prompt=flux_request.prompt,                      # Text description of desired image
                width=flux_request.width,                        # Output image width
                height=flux_request.height,                      # Output image height
                num_inference_steps=flux_request.num_inference_steps,  # Number of denoising steps
                generator=generator,                             # Random number generator
                guidance_scale=flux_request.guidance_scale,      # How closely to follow the prompt
            ).images[0]  # Get the first (and only) generated image

            logger.info("Image generation complete!")
            return image
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    def encode_response(self, image):
        """
        Convert the generated PIL image into an HTTP response.
        
        This method handles the final step of the API pipeline by:
        - Converting the PIL image to PNG format
        - Creating an HTTP response with appropriate headers
        - Setting proper content type and disposition
        
        Args:
            image (PIL.Image): The generated image to encode
            
        Returns:
            Response: FastAPI Response object containing the PNG image
            
        Raises:
            HTTPException: If image encoding fails
        """
        try:
            # Create an in-memory buffer to hold the PNG data
            buffered = BytesIO()
            
            # Save the image as PNG to the buffer
            # PNG format is chosen for lossless quality and broad compatibility
            image.save(buffered, format="PNG")
            buffered.seek(0)  # Reset buffer position to beginning
            
            # Create HTTP response with proper headers
            return Response(
                content=buffered.getvalue(),                      # PNG image data
                media_type="image/png",                           # MIME type
                headers={
                    "Content-Type": "image/png",                  # Explicit content type
                    "Content-Disposition": "inline; filename=generated_image.png"  # Suggest filename
                }
            )
            
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

# Server startup and configuration
if __name__ == "__main__":
    """
    Main entry point for the application.
    
    Creates the API instance and starts the LitServe server.
    The server will listen on port 8000 and handle incoming requests.
    """
    
    # Create an instance of the FLUX API
    api = FluxLitAPI()
    
    # Create LitServer with the API instance
    # timeout=False allows for longer generation times without timing out
    server = ls.LitServer(api, timeout=False)
    
    # Start the server on port 8000
    # The server will be accessible at http://localhost:8000
    server.run(port=8000)