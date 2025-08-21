import os
from flux_from_json_batch_image_generator import batch_generator

#from json_prompt_validator import validate_and_fix_json, save_json



# json validation will be done using online larger models

#-----------------------------------------------------------------------------------------------------------------------------
# reading prompts
#file_path = os.path.join("video_generation_tools", "image_generation", "files", "raw_prompts_json.json")
#with open(file_path, "r", encoding="utf-8") as f:
#    raw_prompts_json = f.read()
#
## validating prompts json and saving
#json_result = validate_and_fix_json(raw_prompts_json)
#if json_result.get("success"):
#    save_json(json_result["data"], filename="video_generation_tools//image_generation//files//fixed_prompts_json.json")
#
#print("\n" + "="*40 + "\n")
#-----------------------------------------------------------------------------------------------------------------------------


# batch generation
story_name="test"
output_dir = f"video_generation/image_generation/outputs/{story_name}"



# reading prompts
file_path = os.path.join("video_generation_tools", "image_generation",output_dir, "prompts.json")
with open(file_path, "r", encoding="utf-8") as f:
    prompts_json = f.read()


is_regeneration=False



batch_generator(prompts_json,
                width=1280,
                height=720,
                guidance_scale=[3.5,2],
                num_inference_steps=28,
                output_dir=output_dir,
                is_regeneration= is_regeneration)