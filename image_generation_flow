from flux_api_integration import FluxAPIClient
import os 
from tqdm.auto import tqdm
from random import randint

json={
  "1": ["human walking on the street, rainy night, hyper realistic, 4k, cinematic, city in the background",
        "a man is running in rain, newyork city night life, 4k, high quality, cinematic, hyper realistic, 2d art"],
  "2": ["a black cat, walking on the edge of a building, an airplane in the background, anime style, bokeh, 4k, HDR, artistic",
        "a black cat, walking on the roof of a sky scraper, playing with a ball, anime style, bokeh, 4k, HDR, artisti"],
  "3":["a stunning view of the mount everest, cloudy weather, snowy, 4k wallpaper, HDR, photorealistic, vivid colours,",
       "a stunning view of the mount everest, sunny, sketch black pencil, drawing, on paper,HD, masterpiece, high resolution"] 
}

flux = FluxAPIClient()


def batch_generator(prompts_list: dict):
    json = prompts_list.copy()
    width = 1280
    height = 720
    guidance_scale = [3.5, 2]
    num_inference_steps = 28
    path = "outputs/teststory"
    os.makedirs(path, exist_ok=True)
    left_prompts = []

    for i, prompts in tqdm(json.items()):
        scene_has_failure = False  # Track if any prompt in this scene fails
        
        for prompt in prompts:

            try:
                for g in guidance_scale:

                    seed = randint(42,18446744073709552000)
                    filename = f"{path}/{i}-{prompt} cfg{g}28 seed{seed}.png"
                
                    try:
                        # Check the return value to see if generation succeeded
                        result = flux.generate_and_save(
                            prompt=prompt,
                            width=width, 
                            height=height, 
                            num_inference_steps=num_inference_steps,  
                            filename=filename,
                            guidance_scale=g,
                            timeout=600,
                            seed= seed
                        )
                        
                        # Only print success if result is not None
                        if result is not None:
                            print(f"✅scene {i}, {prompt}, cfg {g} created successfully")
                        else:
                            print(f"❌scene {i}, {prompt}, cfg {g} generation failed!!")
                            scene_has_failure = True
                                
                    except Exception as e:
                        print(f"❌scene {i}, {prompt}, cfg {g} generation failed!!")
                        print(f"Error: {e}")
                        scene_has_failure = True

            except Exception as e:
                print(f"Error processing scene {i}: {e}")
                scene_has_failure = True
        
        # Add scene index to left_prompts if any generation failed
        if scene_has_failure :
            left_prompts.append(i)

    # Write failed scene indices to file
    with open(f"{path}/left_prompts.txt", "w") as f:
        for failed_scene in left_prompts:
            f.write(f"{failed_scene} ")  # Fixed: now writes the actual failed scene index
    
    # Print summary
    print(f"\nGeneration Summary:")
    print(f"Total failed scenes: {len(left_prompts)}")
    if left_prompts:
        print(f"Failed scene indices: {left_prompts}")
        print(f"Failed scenes written to: {path}/left_prompts.txt")

batch_generator(json)