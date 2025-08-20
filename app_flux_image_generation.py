import os
from json_prompt_validator import validate_and_fix_json, save_json

# reading prompts
file_path = os.path.join("video_generation_tools", "image_generation", "files", "raw_prompts_json.json")
with open(file_path, "r", encoding="utf-8") as f:
    raw_prompts_json = f.read()

# validating prompts json and saving
json_result = validate_and_fix_json(raw_prompts_json)
if json_result.get("success"):
    save_json(json_result["data"], filename="video_generation_tools//image_generation//files//fixed_prompts_json.json")

print("\n" + "="*40 + "\n")


