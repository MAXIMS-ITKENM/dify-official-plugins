#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google.genai.types import Tool, FunctionDeclaration, Schema
import json

def test_tool_serialization():
    print("Testing tool serialization...")
    
    # Create a test tool
    try:
        # Create a Schema object first
        parameters_schema = Schema(
            type="object",
            properties={
                'param1': Schema(type="string", description="Test parameter")
            },
            required=['param1']
        )
        
        fd = FunctionDeclaration(
            name='test_function',
            description='A test function',
            parameters=parameters_schema
        )
        
        tool = Tool(function_declarations=[fd])
        print(f"Tool created successfully")
        print(f"Parameters type: {type(fd.parameters)}")
        print(f"Parameters: {fd.parameters}")
        
        # Test the _tool_to_dict method
        from models.llm.llm import VertexAiLargeLanguageModel
        
        # Create a mock model schemas for initialization
        model_schemas = {}
        llm = VertexAiLargeLanguageModel(model_schemas=model_schemas)
        
        tool_dict = llm._tool_to_dict(tool)
        print(f"Tool dict created: {tool_dict}")
        
        # Try to serialize it to JSON
        json_str = json.dumps(tool_dict, sort_keys=True)
        print('SUCCESS: Tool serialization works')
        print(f'Serialized tool: {json_str[:200]}...')
        
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tool_serialization()
