#!/usr/bin/env python3

print("Starting test...")

try:
    from models.llm.llm import VertexAiLargeLanguageModel
    print("Successfully imported VertexAiLargeLanguageModel")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

try:
    from google.genai.types import Tool, FunctionDeclaration
    print("Successfully imported Google GenAI types")
except Exception as e:
    print(f"Google GenAI import error: {e}")

import json

# Create a test instance
llm = VertexAiLargeLanguageModel()
print("VertexAiLargeLanguageModel instance created")

# Create a test tool without parameters first
print("Creating simple FunctionDeclaration...")
fd = FunctionDeclaration(
    name='test_function',
    description='A test function'
)
print(f"FunctionDeclaration created: {fd.name}")

tool = Tool(function_declarations=[fd])
print("Tool created")

# Test the tool_to_dict method
try:
    print("Testing _tool_to_dict method...")
    result = llm._tool_to_dict(tool)
    print("_tool_to_dict succeeded, result:", result)
    print("Trying JSON serialization...")
    json_str = json.dumps(result, indent=2)
    print('SUCCESS: Tool serialization worked!')
    print('Serialized tool:')
    print(json_str)
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
