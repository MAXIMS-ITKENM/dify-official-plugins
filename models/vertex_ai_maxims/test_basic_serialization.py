#!/usr/bin/env python3

import json

def test_json_serialization():
    """Test that we can serialize objects that would previously fail"""
    
    # Simulate what our _tool_to_dict method should produce
    test_tool_dict = {
        'function_declarations': [
            {
                'name': 'test_function',
                'description': 'A test function',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'param1': {
                            'type': 'string',
                            'description': 'Test parameter'
                        }
                    },
                    'required': ['param1']
                }
            }
        ]
    }
    
    try:
        json_str = json.dumps(test_tool_dict, sort_keys=True)
        print('SUCCESS: Basic tool dict serialization works')
        print(f'Serialized: {json_str}')
        return True
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}')
        return False

if __name__ == "__main__":
    test_json_serialization()
