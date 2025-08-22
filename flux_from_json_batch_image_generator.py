from flux_image_generation_api import FluxAPIClient
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


def batch_generator(prompts_list: dict,
                    width = 1280, 
                    height = 720, 
                    guidance_scale=[3.5, 2], 
                    num_inference_steps = 28, 
                    output_dir = "video_generation/image_generation/stories/test",
                      is_regeneration=False):
    
    json = prompts_list.copy()

    failed_prompts = []
    os.makedirs(output_dir, exist_ok=True) 
    failed_prompts_file = f"{output_dir}/failed_prompts.txt"
    if not os.path.exists(failed_prompts_file):
        with open(failed_prompts_file, "w") as f:
            f.write("")
        
    if is_regeneration:
        with open(f"{output_dir}/failed_prompts.txt", "r",) as f:
            previously_failed_prompts = f.read().strip().split(" ")
            previously_succeeded_prompts= [i for i in json.keys() if i not in previously_failed_prompts]
    else:
        previously_succeeded_prompts = []

    


    for i, prompts in tqdm(json.items()):

        scene_has_failure = False  # Track if any prompt in this scene fails        

        if i not in previously_succeeded_prompts:

            for prompt in prompts:

                try:
                    for g in guidance_scale:

                        seed = randint(42,18446744073709552000)
                        filename = f"{output_dir}/{i}-{prompt} cfg{g} {num_inference_steps}steps seed{seed}.png"

                        try:
                            # Check the return value to see if generation succeeded
                            result = flux.generate_and_save(
                                prompt=prompt,
                                width=width, 
                                height=height, 
                                num_inference_steps=num_inference_steps,  
                                filename=filename,
                                guidance_scale=g,
                                timeout=6,
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
        
            # Add scene index to failed_prompts if any generation failed
            if scene_has_failure :
                failed_prompts.append(i)

    # Write failed scene indices to file
    with open(f"{output_dir}/failed_prompts.txt", "w") as f:
        for failed_scene in failed_prompts:
            f.write(f"{failed_scene} ")  # Fixed: now writes the actual failed scene index
    
    # Print summary
    print(f"\nGeneration Summary:")
    print(f"Total failed scenes: {len(failed_prompts)}")
    if failed_prompts:
        print(f"Failed scene indices: {failed_prompts}")
        print(f"Failed scenes written to: {output_dir}/failed_prompts.txt")

batch_generator(json,is_regeneration=True)