# Example 1: Simple integration in your existing program

from flux_api_integration import FluxAPIClient, AsyncFluxAPIClient
import numpy as np


# Method 1: Using the client class
def flux_generator():
    # Initialize the Flux client
    flux = FluxAPIClient()
    
    # Generate an image during your program execution
    prompt = "zeus, realistic, sitting on his throne, eyes blasting with blue lightning, muscular, long white beard, epic, realistic, 4k, hgihly detailed"
    filename="testpng308.png" 
    num_inference_steps=25
    guidance_scale=4
    width=1280
    height=720
    #seed = np.random.random_integers(low=1,high=18446744073709552000)
    seed=150995
    timeout=600
    
    print("Generating image...")
    flux.generate_and_save(prompt=prompt,
                                #negative_prompt =negative_prompt,
                                filename=filename,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                #seed=seed,
                                timeout=timeout
                                )
    
my_program_with_flux()



