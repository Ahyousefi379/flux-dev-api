import requests
import io
from PIL import Image
import base64
import asyncio
import aiohttp
from typing import Optional, Union, List
import logging
from datetime import datetime
import os

class FluxAPIClient:
    """
    A client class for integrating with the Flux API in your Python programs
    """
    url="https://8000-01jys6rf4f2pnt32wttv0rt57f.cloudspaces.litng.ai/predict"
    url="https://8000-01jys6rf4f2pnt32wttv0rt57f.cloudspaces.litng.ai/predict"
  
    def __init__(self, api_url: str = url):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def generate_image(
        self,
        prompt: str,
#        negative_prompt:str = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1920,
        height: int = 1080,
        seed: Optional[int] = None,
        timeout: int = 300
    ) -> Optional[Image.Image]:
        """
        Generate an image and return it as a PIL Image object
        
        Args:
            prompt: Text description for image generation
            negative_prompt: negative Text description for image generation
            num_inference_steps: Number of denoising steps (1-50)
            guidance_scale: How closely to follow the prompt (1.0-20.0)
            width: Image width in pixels (512-2048)
            height: Image height in pixels (512-2048)
            seed: Random seed for reproducible results
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object or None if generation failed
        """
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            payload["seed"] = seed
#        if negative_prompt  is not None:
#            payload["negative_prompt"] =negative_prompt 
        
        try:
            self.logger.info(f"Generating image: {prompt[:50]}...")
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(response.content))
                self.logger.info("Image generated successfully")
                return image
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_and_save(
        self,
        prompt: str,
        filename: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image and save it to disk
        
        Returns:
            Filepath if successful, None if failed
        """
        image = self.generate_image(prompt, **kwargs)
        
        if image is None:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_generated_{timestamp}.png"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            image.save(filename)
            self.logger.info(f"Image saved to: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return None
   
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Optional[Image.Image]]:
        """
        Generate multiple images from a list of prompts
        
        Returns:
            List of PIL Images (None for failed generations)
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Generating image {i+1}/{len(prompts)}")
            image = self.generate_image(prompt, **kwargs)
            results.append(image)
        
        return results
    
    def is_server_healthy(self) -> bool:
        """
        Check if the API server is responding
        """
        try:
            # Try a simple test request
            response = self.session.post(
                self.api_url,
                json={"prompt": "test"},
                timeout=10
            )
            return response.status_code in [200, 400]  # 400 is also okay, means server is responding
        except:
            return False

# Async version for better performance in async applications
class AsyncFluxAPIClient:
    """
    Async version of the Flux API client for better performance in async applications
    """
    
    def __init__(self, api_url: str = "https://8000-01jys6rf4f2pnt32wttv0rt57f.cloudspaces.litng.ai/predict"):
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)
    
    async def generate_image(
        self,
        prompt: str,
        #negative_prompt :str = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        timeout: int = 300
    ) -> Optional[Image.Image]:
        """
        Async version of image generation
        """
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            payload["seed"] = seed
#       if negative_prompt is not None:
#           payload["negative_prompt"] = negative_prompt 
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status == 200:
                        content = await response.read()
                        image = Image.open(io.BytesIO(content))
                        return image
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            return None
    
    async def batch_generate_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Optional[Image.Image]]:
        """
        Generate multiple images concurrently
        """
        tasks = [self.generate_image(prompt, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Generation failed: {str(result)}")
                clean_results.append(None)
            else:
                clean_results.append(result)
        
        return clean_results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
