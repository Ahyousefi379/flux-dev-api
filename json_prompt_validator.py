import json
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# Load API keys
load_dotenv()

def setup_llm():
    """Setup LLM for JSON fixing"""
    groq_api = os.environ.get("GROQ_API_KEY_1")
    
    llama_8b = ChatGroq(
        api_key=groq_api,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        model_kwargs={"top_p": 0.5},
        max_retries=5,
        max_tokens=100000,
        streaming=False
    )
    
    system_prompt = '''You are a JSON repair specialist. Fix the provided malformed JSON to match this format:
    {
      "1": ["prompt1", "prompt2"],
      "2": ["prompt1", "prompt2"],
      "3": ["prompt1", "prompt2"],
      ...
    }
    there can be more numbers of items but all should follow this pattern
    Requirements:
    - Keys must be numeric strings ("1", "2", "3", etc.)
    - Each key must have image generation prompts as strings
    - Output only valid JSON, no additional text
    - Preserve the original prompt content. don't change, add, remove, modify the content
    - If prompts are missing, just write (blue screen) instead'''
    
    human_prompt = "Fix this malformed JSON:\n{malformed_json}"
    
    prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    parser = JsonOutputParser()
    chain = prompt_template | llama_8b | parser
    
    return chain

def fix_json_with_llm(malformed_json_string):
    """Use LLM to fix malformed JSON"""
    try:
        chain = setup_llm()
        response = chain.invoke({"malformed_json": malformed_json_string})
        return json.dumps(response, indent=2,)
    except Exception as e:
        print(f"Error fixing JSON with LLM: {str(e)}")
        return None

def validate_prompt_json(json_string, auto_fix=True):
    """
    Validates if the output is a valid JSON in the expected format for image prompts.
    If validation fails and auto_fix is True, attempts to fix using LLM.
    
    Expected format:
    {
        "1": ["prompt1", "prompt2"],
        "2": ["prompt1", "prompt2"],
        "3": ["prompt1", "prompt2"]
    }
    
    Args:
        json_string (str): The JSON string to validate
        auto_fix (bool): Whether to attempt LLM-based fixing if validation fails
    
    Returns:
        dict: Contains validation results with 'is_valid', 'data', 'errors', 'fixed', and 'original' keys
    """
    result = {
        'is_valid': False,
        'data': None,
        'errors': [],
        'fixed': False,
        'original': json_string
    }
    
    def _validate_structure(data):
        """Internal function to validate JSON structure"""
        errors = []
        
        # Check if it's a dictionary
        if not isinstance(data, dict):
            errors.append("Output must be a JSON object/dictionary")
            return errors
        
        # Check if all keys are numeric strings
        for key in data.keys():
            if not key.isdigit():
                errors.append(f"Key '{key}' should be a numeric string")
        
        # Check if all values are lists with exactly 2 prompts
        for key, value in data.items():
            if not isinstance(value, list):
                errors.append(f"Value for key '{key}' must be a list")
            elif len(value) != 2:
                errors.append(f"Key '{key}' must contain exactly 2 prompts, found {len(value)}")
            else:
                # Check if all prompts are strings
                for i, prompt in enumerate(value):
                    if not isinstance(prompt, str):
                        errors.append(f"Prompt {i+1} in key '{key}' must be a string")
                    elif len(prompt.strip()) == 0:
                        errors.append(f"Prompt {i+1} in key '{key}' cannot be empty")
        
        return errors
    
    try:
        # Parse JSON
        data = json.loads(json_string)
        result['data'] = data
        
        # Validate structure
        result['errors'] = _validate_structure(data)
        
        # If no errors found, mark as valid
        if not result['errors']:
            result['is_valid'] = True
            return result
            
    except json.JSONDecodeError as e:
        result['errors'].append(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")
    
    # If validation failed and auto_fix is enabled, try to fix with LLM
    if not result['is_valid'] and auto_fix:
        print("⚠️  JSON validation failed. Attempting to fix with LLM...")
        
        fixed_json = fix_json_with_llm(json_string)
        
        if fixed_json:
            try:
                # Validate the fixed JSON
                fixed_data = json.loads(fixed_json)
                fixed_errors = _validate_structure(fixed_data)
                
                if not fixed_errors:
                    print("✅ JSON successfully fixed by LLM!")
                    result['is_valid'] = True
                    result['data'] = fixed_data
                    result['errors'] = []
                    result['fixed'] = True
                else:
                    print("❌ LLM fix attempt failed validation")
                    result['errors'].extend([f"LLM fix error: {error}" for error in fixed_errors])
                    
            except Exception as e:
                result['errors'].append(f"Error in LLM-fixed JSON: {str(e)}")
        else:
            result['errors'].append("LLM failed to fix the JSON")

    # returning valid data or errors 
    if result['is_valid']:
        print('✅ JSON is valid')
        print("=="*10)
        for error in result['errors']:
            print(f"  - {error}")
        return result['data']
    else:
        print("❌ JSON validation failed!")
        for error in result['errors']:
            print(f"  - {error}")
        return None
    





def save_fixed_json(data, filename="fixed_prompts.json"):
    """Save the fixed JSON to a file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Valid JSON saved to {filename}")
    return True

# Example usage
if __name__ == "__main__":
    # Test with valid JSON
    valid_json = '''
    {
        "1": ["A beautiful sunset over mountains", "A serene lake at dawn"],
        "2": ["A bustling city street", "A quiet coffee shop interior"],
        "3": ["A magical forest", "An ancient castle on a hill"]
    }
    '''
    
    print("Testing valid JSON:")
    result = validate_prompt_json(valid_json, auto_fix=False)
    print_validation_result(result)
    
    print("\n" + "="*50 + "\n")
    
    # Test with malformed JSON that LLM can fix
    malformed_json = '''
    {
        "scene_1": ["Only one prompt here"],
        "2": ["Two prompts", "But scene_1 has wrong format"],
        "3": [123, "Non-string prompt mixed with string"]
        "4": "Not even a list"
    }
    '''
    
    print("Testing malformed JSON with auto-fix:")
    result = validate_prompt_json(malformed_json, auto_fix=True)
    print_validation_result(result)
    
    if result['is_valid']:
        save_fixed_json(result)
    
    print("\n" + "="*50 + "\n")
    
    # Test completely broken JSON
    broken_json = "This is not JSON at all!"
    
    print("Testing completely broken JSON with auto-fix:")
    result = validate_prompt_json(broken_json, auto_fix=True)
    print_validation_result(result)