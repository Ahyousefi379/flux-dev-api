import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def fix_json_with_llm(malformed_json):
    """Fix malformed JSON using LLM"""
    api_key = os.environ.get("GROQ_API_KEY_1")
    if not api_key:
        raise ValueError("GROQ_API_KEY_1 not found")
    
    # Simple LLM setup
    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0)
    
    prompt = f"""Fix this JSON to match the exact format. Return ONLY the JSON, no explanations:

{{
  "1": ["prompt1", "prompt2"],
  "2": ["prompt1", "prompt2"], 
  "3": ["prompt1", "prompt2"]
}}

Rules:
- Keys must be "1", "2", "3", etc.
- Each key has exactly 2 string prompts
- Keep original prompt text where possible
- If only 1 prompt exists, place "blue screen" instead of the missing one
- Convert non-string prompts to strings

Broken JSON to fix:
{malformed_json}

Fixed JSON:"""

    response = llm.invoke(prompt)
    raw_output = response.content
    
    # Clean the output - extract JSON from response
    cleaned = raw_output.strip()
    
    # Remove common prefixes/suffixes LLMs add
    prefixes_to_remove = ["```json", "```", "Fixed JSON:", "Here's the fixed JSON:", "The fixed JSON is:"]
    suffixes_to_remove = ["```", "\n```"]
    
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    
    return cleaned

def validate_and_fix_json(json_string):
    """Validate JSON format and fix if needed"""
    
    def is_valid_format(data):
        """Check if JSON matches required format"""
        if not isinstance(data, dict):
            return False
        
        for key, value in data.items():
            if not key.isdigit():
                return False
            if not isinstance(value, list) or len(value) != 2:
                return False
            if not all(isinstance(prompt, str) and prompt.strip() for prompt in value):
                return False
        return True
    
    # Try to parse original JSON
    try:
        data = json.loads(json_string)
        if is_valid_format(data):
            print("✅ JSON is already valid")
            return {"success": True, "data": data, "fixed": False}
    except json.JSONDecodeError:
        pass
    
    # Fix with LLM
    print("🔧 Fixing JSON with LLM...")
    try:
        fixed_json = fix_json_with_llm(json_string)
        
        # Try to parse the cleaned JSON
        try:
            fixed_data = json.loads(fixed_json)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return {"success": False, "error": f"Invalid JSON from LLM: {e}"}
        
        if is_valid_format(fixed_data):
            print("✅ JSON successfully fixed!")
            return {"success": True, "data": fixed_data, "fixed": True}
        else:
            print("❌ LLM output doesn't match required format")
            return {"success": False, "error": "Invalid format after LLM fix"}
    
    except Exception as e:
        print(f"❌ LLM fix failed: {e}")
        return {"success": False, "error": str(e)}

def save_json(data, filename="fixed_prompts.json"):
    """Save JSON to file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved to {filename}")

# Test
if __name__ == "__main__":
    # Valid JSON test
    valid_json = '''{"1": ["sunset over mountains", "lake at dawn"], "2": ["city street", "coffee shop"]}'''
    
    print("Testing valid JSON:")
    result = validate_and_fix_json(valid_json)
    if result["success"]:
        save_json(result["data"])
    
    print("\n" + "="*40 + "\n")
    
    # Malformed JSON test
    malformed_json = '''{
        "scene_1": ["Only one prompt"],
        "2": ["Two prompts", "Good format"],
        "3": [123, "Mixed types"],
        "4": "Not a list"
    }'''
    
    print("Testing malformed JSON:")
    result = validate_and_fix_json(malformed_json)
    if result["success"]:
        save_json(result["data"], "fixed_malformed.json")
    
    print("\n" + "="*40 + "\n")
    
    # Broken JSON test
    broken_json = "This is not JSON at all!"
    
    print("Testing broken JSON:")
    result = validate_and_fix_json(broken_json)
    if result["success"]:
        save_json(result["data"], "fixed_broken.json")