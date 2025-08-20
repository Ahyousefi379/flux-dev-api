import argparse
import requests
from datetime import datetime
import time
import sys
import os
from typing import Optional

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:8000/predict"

def show_progress(message: str):
    """Simple progress indicator"""
    print(f"⏳ {message}", end="", flush=True)
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

def send_request(
    prompt: str, 
    num_inference_steps: int = 4, 
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    output_dir: str = "outputs"
):
    """
    Send a request to the Flux server for image generation
    
    Args:
        prompt: Text prompt for image generation
        num_inference_steps: Number of inference steps (1-50)
        guidance_scale: Guidance scale (1.0-20.0)
        width: Image width (512-2048)
        height: Image height (512-2048)
        seed: Random seed for reproducibility
        output_dir: Directory to save generated images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare request payload
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height
    }
    
    if seed is not None:
        payload["seed"] = seed
    
    print(f"🎨 Generating image with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"📊 Parameters: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}")
    if seed is not None:
        print(f"🌱 Seed: {seed}")
    
    try:
        show_progress("Sending request to server")
        
        # Send request with timeout
        response = requests.post(
            API_URL, 
            json=payload, 
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            # Generate filename with timestamp and parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_{timestamp}_s{num_inference_steps}_g{guidance_scale:.1f}"
            if seed is not None:
                filename += f"_seed{seed}"
            filename += ".png"
            
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            with open(filepath, "wb") as image_file:
                image_file.write(response.content)
            
            print(f"✅ Image saved successfully!")
            print(f"📁 File: {filepath}")
            print(f"📦 Size: {len(response.content)} bytes")
            
            return filepath
        
        elif response.status_code == 400:
            print(f"❌ Bad request: {response.text}")
            print("Check your parameters and try again.")
            
        elif response.status_code == 500:
            print(f"❌ Server error: {response.text}")
            print("The server encountered an error during generation.")
            
        else:
            print(f"❌ Unexpected error: Status code {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server might be busy or the image is taking too long to generate.")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the server is running and accessible.")
        print(f"Server URL: {API_URL}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Flux server and receive the generated image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --prompt "A beautiful sunset over mountains"
  python client.py --prompt "A cat in space" --steps 8 --guidance 7.5
  python client.py --prompt "Abstract art" --width 1536 --height 1024 --seed 42
        """ 
    )
    
    parser.add_argument(
        "--prompt", 
        required=True, 
        help="Text prompt for image generation"
    )
    
    parser.add_argument(
        "--steps", 
        type=int, 
        default=4, 
        help="Number of inference steps (1-50, default: 4)"
    )
    
    parser.add_argument(
        "--guidance", 
        type=float, 
        default=3.5, 
        help="Guidance scale (1.0-20.0, default: 3.5)"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=1024, 
        help="Image width in pixels (512-2048, default: 1024)"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=1024, 
        help="Image height in pixels (512-2048, default: 1024)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed for reproducible results"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="outputs", 
        help="Directory to save generated images (default: outputs)"
    )
    
    parser.add_argument(
        "--server-url", 
        default="http://127.0.0.1:8000/predict", 
        help="Server URL (default: http://127.0.0.1:8000/predict)"
    )

    args = parser.parse_args()
    
    # Update global API URL if provided
    global API_URL
    API_URL = args.server_url
    
    # Validate parameters
    if not (1 <= args.steps <= 50):
        print("❌ Error: Number of steps must be between 1 and 50")
        sys.exit(1)
        
    if not (1.0 <= args.guidance <= 20.0):
        print("❌ Error: Guidance scale must be between 1.0 and 20.0")
        sys.exit(1)
        
    if not (512 <= args.width <= 2048):
        print("❌ Error: Width must be between 512 and 2048 pixels")
        sys.exit(1)
        
    if not (512 <= args.height <= 2048):
        print("❌ Error: Height must be between 512 and 2048 pixels")
        sys.exit(1)
    
    # Send the request
    result = send_request(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    if result:
        print("🎉 Generation completed successfully!")
    else:
        print("💔 Generation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()