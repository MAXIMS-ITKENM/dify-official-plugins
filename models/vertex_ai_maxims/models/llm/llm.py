import base64
import io
import json
import time
import hashlib
import logging
from collections.abc import Generator
from typing import Optional, Union, cast
import google.auth.transport.requests
import requests
from anthropic import AnthropicVertex, Stream
from anthropic.types import (
    ContentBlockDeltaEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
)
from dify_plugin.entities.model import PriceType
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from google.api_core import exceptions
from google.oauth2 import service_account
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, ThinkingConfig, FunctionDeclaration, FunctionCall, Content, Part, CreateCachedContentConfig, UpdateCachedContentConfig
from typing import Any
from PIL import Image


GLOBAL_ONLY_MODELS = ["gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-lite-preview-06-17"]

# Context caching minimum token requirements by model
CACHE_MINIMUM_TOKENS = {
    "gemini-2.5-pro": 2048,
    "gemini-2.5-flash": 1024,
    "gemini-2.5-flash-lite": 1024
}

# Set up logger for verbose messaging
logger = logging.getLogger(__name__)

# Configure logger if it hasn't been configured yet
if not logger.handlers:
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to see all our detailed logs
    logger.info("Logger configured for VertexAI LLM with DEBUG level")


class VertexAiLargeLanguageModel(LargeLanguageModel):
    # Class-level cache for storing cache mappings
    # Key: content hash, Value: {"cache_name": str, "timestamp": float, "user_id": str}
    _cache_mapping = {}
    _CACHE_CLEANUP_THRESHOLD = 100  # Clean up when we have more than 100 entries
    _CACHE_MAX_AGE = 3200  # Remove entries older than 1 hour
    
    # Instance-level client caching
    _CLIENT_CACHE_TTL = 3600  # Cache clients for 1 hour
    
    @property
    def _client_cache(self):
        """Lazy initialization of client cache to avoid __init__ issues."""
        if not hasattr(self, '_client_cache_dict'):
            self._client_cache_dict = {}
        return self._client_cache_dict
    
    def _cleanup_cache_mapping(self):
        """Clean up old cache mapping entries to prevent memory bloat."""
        if len(self._cache_mapping) <= self._CACHE_CLEANUP_THRESHOLD:
            return
            
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self._cache_mapping.items():
            if current_time - entry["timestamp"] > self._CACHE_MAX_AGE:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache_mapping[key]
            
        if keys_to_remove:
            logger.info(f"[CACHE] Cleaned up {len(keys_to_remove)} old cache entries, {len(self._cache_mapping)} remaining")
    
    def _get_credentials_hash(self, model: str, credentials: dict) -> str:
        """
        Generate a hash for the credentials to use as cache key.
        
        :param model: model name
        :param credentials: model credentials
        :return: hash string for caching
        """
        # Create a stable hash based on relevant credential components
        cache_key_components = {
            "model": model,
            "project_id": credentials.get("vertex_project_id", ""),
            "location": credentials.get("vertex_location", ""),
            "service_account_key": credentials.get("vertex_service_account_key", "")[:50] if credentials.get("vertex_service_account_key") else ""  # Use first 50 chars to avoid huge keys
        }
        
        # Handle global models and preview models location logic
        if model in GLOBAL_ONLY_MODELS:
            cache_key_components["location"] = "global"
        elif "preview" in model:
            cache_key_components["location"] = "us-central1"
        
        cache_key_str = json.dumps(cache_key_components, sort_keys=True)
        return hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()[:16]
    
    def _cleanup_client_cache(self):
        """Clean up expired client cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self._client_cache.items():
            if current_time - entry["timestamp"] > self._CLIENT_CACHE_TTL:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._client_cache[key]
            
        if keys_to_remove:
            logger.debug(f"[AUTH] Cleaned up {len(keys_to_remove)} expired client cache entries, {len(self._client_cache)} remaining")
    
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user_id: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param model_config: model config
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user_id: unique user id
        :return: full response or stream response chunk generator result
        """
        
        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user_id)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages using official Google GenAI API

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return: number of tokens
        """
        try:
            logger.debug(f"[TOKEN_COUNT] Starting token count for model {model} with {len(prompt_messages)} messages")
            
            # Create authenticated client
            client = self._create_authenticated_client(model, credentials)
            
            # Extract system instruction and convert other messages to genai format
            contents = []
            system_instruction = ""
            
            for msg in prompt_messages:
                if isinstance(msg, SystemPromptMessage):
                    system_instruction = msg.content if isinstance(msg.content, str) else str(msg.content)
                    logger.debug(f"[TOKEN_COUNT] System instruction found: {len(system_instruction)} chars")
                else:
                    content = self._format_message_to_genai_content(msg)
                    contents.append(content)
            
            logger.debug(f"[TOKEN_COUNT] Converted {len(contents)} messages to genai format")
            
            # Convert tools to genai format if provided
            genai_tools = None
            if tools:
                genai_tools = self._convert_tools_to_genai_tool(tools)
                logger.debug(f"[TOKEN_COUNT] Converted {len(genai_tools)} tools to genai format")
            
            # Use official API to count tokens with proper parameters
            count_params = {
                "model": model,
                "contents": contents,
            }
            
            # Add system instruction if present
            if system_instruction:
                count_params["system_instruction"] = system_instruction
            
            # Add tools if present
            if genai_tools:
                count_params["tools"] = genai_tools
            
            response = client.models.count_tokens(**count_params)
            
            token_count = response.total_tokens or 0
            logger.info(f"[TOKEN_COUNT] Official API returned {token_count} tokens")
            
            if hasattr(response, 'cached_content_token_count') and response.cached_content_token_count:
                logger.debug(f"[TOKEN_COUNT] Cached content tokens: {response.cached_content_token_count}")
            
            return token_count
            
        except Exception as e:
            logger.warning(f"[TOKEN_COUNT] Failed to get token count from official API: {str(e)}")
            logger.debug(f"[TOKEN_COUNT] Falling back to GPT-2 estimation")
            # Fallback to GPT-2 estimation
            prompt = self._convert_messages_to_prompt(prompt_messages)
            return self._get_num_tokens_by_gpt2(prompt)

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        """
        Format a list of messages into a full prompt for the Google model

        :param messages: List of PromptMessage to combine.
        :return: Combined string with necessary human_prompt and ai_prompt tags.
        """
        messages = messages.copy()
        text = "".join((self._convert_one_message_to_text(message) for message in messages))
        return text.rstrip()

    def _create_authenticated_client(self, model: str, credentials: dict):
        """
        Create an authenticated GenAI client for Vertex AI operations with caching.
        
        :param model: model name (used to determine location)
        :param credentials: model credentials containing service account key and project info
        :return: authenticated GenAI client
        :raises: Exception if authentication fails
        """
        logger.debug(f"[AUTH] Creating authenticated client for model {model}")
        
        # Clean up expired client cache entries periodically
        self._cleanup_client_cache()
        
        # Generate cache key for this set of credentials
        credentials_hash = self._get_credentials_hash(model, credentials)
        
        # Check if we have a cached client
        cached_entry = self._client_cache.get(credentials_hash)
        if cached_entry:
            current_time = time.time()
            if current_time - cached_entry["timestamp"] < self._CLIENT_CACHE_TTL:
                logger.debug(f"[AUTH] Using cached client for credentials hash {credentials_hash}")
                return cached_entry["client"]
            else:
                # Client expired, remove from cache
                logger.debug(f"[AUTH] Cached client expired for credentials hash {credentials_hash}")
                del self._client_cache[credentials_hash]
        
        # Create new client
        logger.debug(f"[AUTH] Creating new client for credentials hash {credentials_hash}")
        
        # Extract and decode service account information
        service_account_info = (
            json.loads(base64.b64decode(service_account_key))
            if (
                service_account_key := credentials.get("vertex_service_account_key", "")
            )
            else None
        )
        
        if not service_account_info:
            raise ValueError("No service account key found in credentials")
        
        # Determine project and location
        project_id = credentials["vertex_project_id"]
        if model in GLOBAL_ONLY_MODELS:
            location = "global"
        elif "preview" in model:
            location = "us-central1"
        else:
            location = credentials["vertex_location"]
        
        logger.debug(f"[AUTH] Using project_id={project_id}, location={location}")
        
        # Create authenticated credentials
        SCOPES = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/generative-language"
        ]
        credential = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )
        
        # Create and return authenticated client
        client = genai.Client(credentials=credential, project=project_id, location=location, vertexai=True)
        
        # Cache the client
        self._client_cache[credentials_hash] = {
            "client": client,
            "timestamp": time.time()
        }
        
        logger.debug(f"[AUTH] Client created and cached successfully (cache size: {len(self._client_cache)})")
        
        return client

    def _convert_tools_to_genai_tool(self, tools: list[PromptMessageTool]) -> list[Tool]:
        """
        Convert tool messages to genai tools

        :param tools: tool messages
        :return: genai tools
        """
        tool_declarations = []
        for tool_config in tools:
            # Ensure tool_config is a proper PromptMessageTool object
            if not hasattr(tool_config, 'parameters') or not hasattr(tool_config, 'name'):
                logger.warning(f"[TOOLS] Skipping invalid tool config: {type(tool_config)} - {tool_config}")
                continue
                
            properties_for_schema = {}
            
            # tool_config.parameters is guaranteed to be a dict by the Pydantic model
            parameters_input_dict = tool_config.parameters
            if not isinstance(parameters_input_dict, dict):
                logger.warning(f"[TOOLS] Skipping tool {tool_config.name} - parameters is not a dict: {type(parameters_input_dict)}")
                continue
                
            raw_properties = parameters_input_dict.get("properties", {})

            if isinstance(raw_properties, dict):
                for key, value_schema in raw_properties.items():
                    if not isinstance(value_schema, dict):
                        # Property schema must be a dictionary
                        continue

                    raw_type_str = str(value_schema.get("type", "string")).lower()
                    # Map types for genai compatibility
                    type_mapping = {
                        "select": "string",
                        "integer": "number",
                    }
                    final_type_for_prop = type_mapping.get(raw_type_str, raw_type_str)
                    
                    prop_details = {
                        "type": final_type_for_prop,
                        "description": value_schema.get("description", ""),
                    }
                    
                    enum_values = value_schema.get("enum")
                    # Add enum only if it's a non-empty list (OpenAPI recommendation)
                    if enum_values and isinstance(enum_values, list):
                        prop_details["enum"] = enum_values
                    
                    properties_for_schema[key] = prop_details

            # Schema for the 'parameters' object of the function declaration
            parameters_schema_for_declaration = {
                "type": "object", 
                "properties": properties_for_schema,
            }
            
            required_params = parameters_input_dict.get("required")
            # Add required only if it's a non-empty list of strings (OpenAPI recommendation)
            if required_params and isinstance(required_params, list) and all(isinstance(item, str) for item in required_params):
                parameters_schema_for_declaration["required"] = required_params

            # Create function declaration for genai
            function_declaration = FunctionDeclaration(
                name=tool_config.name,
                description=tool_config.description or "", 
                parameters=cast(Any, parameters_schema_for_declaration)
            )
            tool_declarations.append(function_declaration)

        return [Tool(function_declarations=tool_declarations)] if tool_declarations else []

    def _tool_to_dict(self, tool: Tool) -> dict:
        """
        Convert a Tool object to dictionary for caching purposes
        
        :param tool: Tool object to convert
        :return: Dictionary representation
        """
        tool_dict: dict[str, Any] = {}
        
        if hasattr(tool, 'function_declarations') and tool.function_declarations:
            tool_dict['function_declarations'] = []
            for func_decl in tool.function_declarations:
                func_dict: dict[str, Any] = {
                    'name': func_decl.name,
                    'description': func_decl.description,
                }
                if hasattr(func_decl, 'parameters') and func_decl.parameters:
                    # Convert parameters to dict safely for JSON serialization
                    try:
                        # For Google GenAI Schema objects, extract key attributes manually
                        params_dict = {}
                        
                        # Extract common schema attributes
                        if hasattr(func_decl.parameters, 'type'):
                            params_dict['type'] = str(func_decl.parameters.type)
                        if hasattr(func_decl.parameters, 'properties'):
                            params_dict['properties'] = self._convert_object_to_dict(func_decl.parameters.properties)
                        if hasattr(func_decl.parameters, 'required'):
                            required = func_decl.parameters.required
                            if required:
                                params_dict['required'] = list(required) if hasattr(required, '__iter__') and not isinstance(required, str) else [str(required)]
                        if hasattr(func_decl.parameters, 'description') and func_decl.parameters.description:
                            params_dict['description'] = str(func_decl.parameters.description)
                        if hasattr(func_decl.parameters, 'format') and func_decl.parameters.format:
                            params_dict['format'] = str(func_decl.parameters.format)
                        if hasattr(func_decl.parameters, 'enum') and func_decl.parameters.enum:
                            params_dict['enum'] = list(func_decl.parameters.enum)
                        
                        # If we extracted any attributes, use them
                        if params_dict:
                            func_dict['parameters'] = params_dict
                        else:
                            # Fallback: try to convert the object directly
                            if isinstance(func_decl.parameters, dict):
                                func_dict['parameters'] = self._convert_object_to_dict(func_decl.parameters)
                            else:
                                # Last resort: create a simplified dict based on string representation
                                func_dict['parameters'] = {'_repr': str(func_decl.parameters)}
                                
                    except Exception as e:
                        # If all else fails, use a string representation
                        logger.debug(f"[CACHE] Failed to serialize parameters for {func_decl.name}: {e}")
                        func_dict['parameters'] = {'_repr': str(func_decl.parameters) if func_decl.parameters else '{}'}
                        
                tool_dict['function_declarations'].append(func_dict)
        
        if hasattr(tool, 'google_search') and tool.google_search:
            tool_dict['google_search'] = True
            
        return tool_dict

    def _convert_object_to_dict(self, obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable dictionaries
        
        :param obj: Object to convert
        :return: JSON-serializable representation
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._convert_object_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                try:
                    result[str(key)] = self._convert_object_to_dict(value)
                except Exception as e:
                    logger.debug(f"[CACHE] Failed to convert dict key {key}: {e}")
                    result[str(key)] = str(value)
            return result
        elif hasattr(obj, '__dict__'):
            # Object has attributes, try to extract them safely
            try:
                obj_dict = {}
                for attr_name, attr_value in obj.__dict__.items():
                    if not attr_name.startswith('_'):  # Skip private attributes
                        try:
                            obj_dict[attr_name] = self._convert_object_to_dict(attr_value)
                        except Exception as e:
                            logger.debug(f"[CACHE] Failed to convert attribute {attr_name}: {e}")
                            obj_dict[attr_name] = str(attr_value)
                return obj_dict if obj_dict else str(obj)
            except Exception as e:
                logger.debug(f"[CACHE] Failed to convert object __dict__: {e}")
                return str(obj)
        else:
            # Fallback to string representation
            return str(obj)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            ping_message = SystemPromptMessage(content="ping")
            self._generate(model, credentials, [ping_message], {"max_tokens_to_sample": 5})
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _has_files_in_conversation(self, prompt_messages: list[PromptMessage]) -> bool:
        """
        Check if the conversation contains any uploaded files (images, documents, audio, video).
        
        :param prompt_messages: List of prompt messages to check
        :return: True if files are detected, False otherwise
        """
        logger.info(f"[CACHE] Checking for files in conversation with {len(prompt_messages)} messages")
        
        file_count = 0
        file_types = []
        
        for i, msg in enumerate(prompt_messages):
            if isinstance(msg, UserPromptMessage) and isinstance(msg.content, list):
                for j, content in enumerate(msg.content):
                    if content.type in [
                        PromptMessageContentType.IMAGE,
                        PromptMessageContentType.DOCUMENT,
                        PromptMessageContentType.AUDIO,
                        PromptMessageContentType.VIDEO
                    ]:
                        file_count += 1
                        file_types.append(content.type.name)
                        logger.debug(f"[CACHE] Found file #{file_count} of type {content.type.name} in message {i}, content {j}")
        
        has_files = file_count > 0
        if has_files:
            logger.info(f"[CACHE] File detection result: {file_count} files found with types: {', '.join(set(file_types))}")
        else:
            logger.info(f"[CACHE] File detection result: No files found in conversation")
            
        return has_files

    def _get_cache_minimum_tokens(self, model: str) -> int:
        """
        Get minimum token count required for caching based on the model.
        
        :param model: model name
        :return: minimum token count required for caching
        """
        # Check for exact model matches first
        for model_prefix, min_tokens in CACHE_MINIMUM_TOKENS.items():
            if model.startswith(model_prefix):
                logger.debug(f"[CACHE] Model {model} matched prefix {model_prefix}, min tokens: {min_tokens}")
                return min_tokens
        
        # Special case handling for preview models
        if "preview" in model.lower():
            if "pro" in model.lower():
                logger.debug(f"[CACHE] Preview Pro model {model}, using 2048 tokens")
                return 2048
            else:
                logger.debug(f"[CACHE] Preview Flash model {model}, using 1024 tokens")
                return 1024
        
        # Default for unknown models (assuming most restrictive)
        logger.warning(f"[CACHE] Unknown model {model}, using default 2048 tokens")
        return 2048

    def _check_cache_token_requirements(
        self,
        client,
        model: str,
        system_instruction: str,
        tools: list,
        conversation_contents: Optional[list] = None,
    ) -> tuple[bool, int]:
        """
        Check if content meets minimum token requirements for caching.
        
        :param client: authenticated genai client
        :param model: model name
        :param system_instruction: system instruction text
        :param tools: list of tools
        :param conversation_contents: optional conversation content list
        :return: tuple of (meets_requirements, actual_token_count)
        """
        try:
            # Prepare content for token counting
            contents = []
            
            # Add conversation contents if provided
            if conversation_contents:
                contents.extend(conversation_contents)
            
            # For token counting, we need to simulate the cache content
            # Create content that represents what would be cached
            # if system_instruction or tools:
            #     parts = []
                
            #     if system_instruction:
            #         parts.append(Part(text=f"System: {system_instruction}"))
                
            #     if tools:
            #         tools_text = f"Tools: {len(tools)} tools available"
            #         parts.append(Part(text=tools_text))
                
            #     cache_simulation_content = Content(parts=parts)
            #     contents.append(cache_simulation_content)
            
            # Count tokens using the official API
            count_params = {
                "model": model,
                "contents": contents,
            }
            
            # Add system instruction if present
            if system_instruction:
                count_params["system_instruction"] = system_instruction
            
            # Add tools if present
            if tools:
                count_params["tools"] = tools
            
            response = client.models.count_tokens(**count_params)
            
            actual_tokens = response.total_tokens or 0
            min_required = self._get_cache_minimum_tokens(model)
            
            meets_requirements = actual_tokens >= min_required
            
            logger.info(f"[CACHE] Token validation: {actual_tokens} tokens (min required: {min_required}, meets requirement: {meets_requirements})")
            
            return meets_requirements, actual_tokens
            
        except Exception as e:
            logger.warning(f"[CACHE] Failed to validate token requirements: {str(e)}")
            # If we can't count tokens, be conservative and assume it doesn't meet requirements
            return False, 0

    def _create_or_get_cache(
        self,
        client,
        model: str,
        credentials: dict,
        system_instruction: str,
        tools: list,
        conversation_id: Optional[str] = None,
        conversation_contents: Optional[list] = None,
    ) -> Optional[str]:
        """
        Create or retrieve a cached content for the given context using official genai client.
        
        For app-level caching: Uses system instruction + tools (no conversation_id or conversation_contents)
        For conversation-level caching: Uses system instruction + tools + conversation_id + conversation_contents
        
        If cache exists, updates its TTL. Returns the cache name if successful, None otherwise.
        """
        import time
        start_time = time.time()
        
        # Cleanup old cache entries periodically
        self._cleanup_cache_mapping()
        
        try:
            # Determine cache type and log operation start
            is_conversation_cache = conversation_id and conversation_contents
            cache_type = "conversation-level" if is_conversation_cache else "app-level"
            logger.info(f"[CACHE] Starting {cache_type} cache operation for model {model}")
            
            # Create cache ID based on context type
            if is_conversation_cache:
                # Conversation-level cache with full conversation history
                conversation_hash = hashlib.sha256(json.dumps([c.to_dict() for c in conversation_contents or []], sort_keys=True).encode('utf-8')).hexdigest()[:12]
                cache_content = f"conv-{conversation_id}-{model}-{system_instruction}-{json.dumps([self._tool_to_dict(t) for t in tools], sort_keys=True)}-history-{conversation_hash}"
                cache_id_prefix = "dify-conv-cache"
                cache_display_name = f"dify-conversation-cache-{int(time.time())}"
                ttl = "1800s"  # 30 minutes for conversation-level cache
                logger.info(f"[CACHE] Conversation cache: conversation_id={conversation_id}, history_items={len(conversation_contents or [])}")
                logger.debug(f"[CACHE] Conversation hash: {conversation_hash}")
            else:
                # App-level cache (no conversation history)
                cache_content = f"app-{model}-{system_instruction}-{json.dumps([self._tool_to_dict(t) for t in tools], sort_keys=True)}"
                cache_id_prefix = "dify-app-cache"
                cache_display_name = f"dify-app-cache-{int(time.time())}"
                ttl = "3600s"  # 1 hour for app-level cache
                logger.info(f"[CACHE] App-level cache: tools_count={len(tools) if tools else 0}")
            
            # Create content hash for cache lookup - include user_id for better cache isolation
            if is_conversation_cache:
                # Conversation-level cache with full conversation history
                conversation_hash = hashlib.sha256(json.dumps([c.to_dict() for c in conversation_contents or []], sort_keys=True).encode('utf-8')).hexdigest()[:12]
                cache_content = f"conv-{conversation_id}-{model}-{system_instruction}-{json.dumps([self._tool_to_dict(t) for t in tools], sort_keys=True)}-history-{conversation_hash}"
                cache_display_name = f"dify-conversation-cache-{int(time.time())}"
                ttl = "1800s"  # 30 minutes for conversation-level cache
                logger.info(f"[CACHE] Conversation cache: conversation_id={conversation_id}, history_items={len(conversation_contents or [])}")
                logger.debug(f"[CACHE] Conversation hash: {conversation_hash}")
            else:
                # App-level cache (no conversation history) - include basic context to isolate per application
                # Use a combination of system instruction, tools, and model to create app-level context
                app_context = f"sys:{len(system_instruction) if system_instruction else 0}"
                tools_context = f"tools:{len(tools) if tools else 0}"
                cache_content = f"app-{model}-{app_context}-{tools_context}-{hashlib.sha256(system_instruction.encode('utf-8') if system_instruction else b'').hexdigest()[:8]}"
                if tools:
                    tools_hash = hashlib.sha256(json.dumps([self._tool_to_dict(t) for t in tools], sort_keys=True).encode('utf-8')).hexdigest()[:8]
                    cache_content += f"-{tools_hash}"
                cache_display_name = f"dify-app-cache-{int(time.time())}"
                ttl = "3600s"  # 1 hour for app-level cache
                logger.info(f"[CACHE] App-level cache: tools_count={len(tools) if tools else 0}")
            
            # Create content hash as our local cache key
            content_hash = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
            logger.info(f"[CACHE] Generated content hash: {content_hash}")
            logger.debug(f"[CACHE] Cache config: display_name={cache_display_name}, ttl={ttl}")
            
            # Check if we have this cache in our local mapping
            cache_lookup_start = time.time()
            cached_entry = self._cache_mapping.get(content_hash)
            cached_google_name = cached_entry.get("cache_name") if cached_entry else None
            
            if cached_google_name:
                # We have a cached name, try to verify it still exists
                try:
                    content_cache = client.caches.get(name=cached_google_name)
                    cache_lookup_time = time.time() - cache_lookup_start
                    
                    logger.info(f"[CACHE] Cache HIT: Found existing cache via mapping (lookup took {cache_lookup_time:.3f}s)")
                    logger.debug(f"[CACHE] Current expire time: {content_cache.expire_time}")
                    logger.debug(f"[CACHE] Using cached Google name: {cached_google_name}")
                    
                    # Update the cache entry timestamp and TTL
                    self._cache_mapping[content_hash]["timestamp"] = time.time()
                    
                    # Update the cache TTL to extend its life
                    ttl_update_start = time.time()
                    try:
                        updated_cache = client.caches.update(
                            name=cached_google_name, 
                            config=UpdateCachedContentConfig(ttl=ttl)
                        )
                        ttl_update_time = time.time() - ttl_update_start
                        total_time = time.time() - start_time
                        
                        logger.info(f"[CACHE] Cache TTL updated in {ttl_update_time:.3f}s, new expire time: {updated_cache.expire_time}")
                        logger.info(f"[CACHE] Cache retrieval and update completed in {total_time:.3f}s")
                        
                        return cached_google_name
                    except Exception as update_error:
                        logger.warning(f"[CACHE] Failed to update cache TTL: {str(update_error)}")
                        # Return the cache name anyway, it's still valid
                        total_time = time.time() - start_time
                        logger.info(f"[CACHE] Cache retrieval completed in {total_time:.3f}s (TTL update failed)")
                        return cached_google_name
                        
                except Exception as verify_error:
                    # Cache no longer exists, remove from mapping
                    logger.info(f"[CACHE] Cached name no longer valid, removing from mapping: {str(verify_error)}")
                    del self._cache_mapping[content_hash]
                    cached_google_name = None
            
            if not cached_google_name:
                cache_lookup_time = time.time() - cache_lookup_start
                logger.info(f"[CACHE] Cache MISS: No valid cache found (lookup took {cache_lookup_time:.3f}s)")
                logger.info(f"[CACHE] Creating new {cache_type} cache")
                
                # Check token requirements before attempting cache creation
                token_check_start = time.time()
                meets_requirements, actual_tokens = self._check_cache_token_requirements(
                    client, model, system_instruction, tools, 
                    conversation_contents if is_conversation_cache else None
                )
                token_check_time = time.time() - token_check_start
                
                if not meets_requirements:
                    min_required = self._get_cache_minimum_tokens(model)
                    total_time = time.time() - start_time
                    logger.warning(f"[CACHE] Token validation FAILED: {actual_tokens} tokens < {min_required} required (check took {token_check_time:.3f}s)")
                    logger.warning(f"[CACHE] Skipping cache creation due to insufficient tokens (total time: {total_time:.3f}s)")
                    return None
                
                logger.info(f"[CACHE] Token validation PASSED: {actual_tokens} tokens meet requirements (check took {token_check_time:.3f}s)")
                
                cache_creation_start = time.time()
                try:
                    # Prepare cache configuration
                    if is_conversation_cache:
                        # For conversation-level cache with full conversation history
                        cache_config = CreateCachedContentConfig(
                            contents=conversation_contents or [],
                            system_instruction=system_instruction,
                            tools=tools if tools else None,
                            display_name=cache_display_name,
                            ttl=ttl,
                        )
                        logger.debug(f"[CACHE] Conversation cache config: contents={len(conversation_contents or [])} items, tools={len(tools) if tools else 0}")
                    else:
                        # For app-level cache without conversation history
                        cache_config = CreateCachedContentConfig(
                            system_instruction=system_instruction,
                            tools=tools if tools else None,
                            display_name=cache_display_name,
                            ttl=ttl,
                        )
                        logger.debug(f"[CACHE] App-level cache config: tools={len(tools) if tools else 0}")
                    
                    # Create the cache using official client
                    content_cache = client.caches.create(
                        model=model,
                        config=cache_config
                    )
                    
                    cache_creation_time = time.time() - cache_creation_start
                    total_time = time.time() - start_time
                    
                    # Store the mapping between our hash and Google's cache name with metadata
                    google_cache_name = content_cache.name
                    self._cache_mapping[content_hash] = {
                        "cache_name": google_cache_name,
                        "timestamp": time.time(),
                        "cache_type": cache_type,
                        "model": model
                    }
                    
                    logger.info(f"[CACHE] Cache CREATE SUCCESS: {google_cache_name} created in {cache_creation_time:.3f}s")
                    logger.debug(f"[CACHE] Stored mapping: {content_hash} -> {google_cache_name}")
                    logger.info(f"[CACHE] Cache mapping size: {len(self._cache_mapping)} entries")
                    if hasattr(content_cache, 'usage_metadata'):
                        logger.debug(f"[CACHE] Cache usage metadata: {content_cache.usage_metadata}")
                    logger.info(f"[CACHE] Total cache creation completed in {total_time:.3f}s")
                    
                    return google_cache_name
                    
                except Exception as create_error:
                    cache_creation_time = time.time() - cache_creation_start
                    total_time = time.time() - start_time
                    logger.error(f"[CACHE] Cache CREATE FAILED after {cache_creation_time:.3f}s: {str(create_error)}")
                    logger.debug(f"[CACHE] Cache creation error details: {type(create_error).__name__}: {str(create_error)}")
                    logger.error(f"[CACHE] Total failed cache operation took {total_time:.3f}s")
                    return None
                
        except Exception as e:
            # Log the error but don't fail the request
            operation_time = time.time() - start_time
            logger.error(f"[CACHE] Cache operation FAILED after {operation_time:.3f}s: {str(e)}")
            logger.debug(f"[CACHE] Cache operation error details: {type(e).__name__}: {str(e)}")
            return None

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user_id: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: credentials kwargs
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user_id: unique user id
        :return: full response or stream response chunk generator result
        """
        import time
        generation_start = time.time()
        
        logger.info(f"[GENERATION] Starting generation for model {model}")
        logger.info(f"[GENERATION] Parameters: messages={len(prompt_messages)}, tools={len(tools) if tools else 0}, stream={stream}, user_id={user_id}")
        logger.debug(f"[GENERATION] Model parameters: {model_parameters}")
        
        config_kwargs = model_parameters.copy()
        config_kwargs["max_output_tokens"] = config_kwargs.pop("max_tokens_to_sample", None)
        
        response_schema = None
        if "json_schema" in config_kwargs:
            response_schema = self._convert_schema_for_vertex(config_kwargs.pop("json_schema"))
            logger.debug(f"[GENERATION] Using JSON schema from json_schema parameter")
        elif "response_schema" in config_kwargs:
            response_schema = self._convert_schema_for_vertex(config_kwargs.pop("response_schema"))
            logger.debug(f"[GENERATION] Using JSON schema from response_schema parameter")
            
        if "response_schema" in config_kwargs:
            config_kwargs.pop("response_schema")
            
        enable_context_caching = config_kwargs.pop("context_caching", False)
        dynamic_threshold = config_kwargs.pop("grounding", False)
        no_thinking = config_kwargs.pop("no_thinking", False)
        thinking_budget = config_kwargs.pop("thinking_budget", None)
        
        logger.info(f"[GENERATION] Special features: context_caching={enable_context_caching}, grounding={dynamic_threshold}, no_thinking={no_thinking}, thinking_budget={thinking_budget}")
        
        # Handle thinking configuration logic
        if no_thinking:
            thinking_budget = 0  # Disable thinking
            logger.debug(f"[GENERATION] Thinking disabled via no_thinking parameter")
        elif thinking_budget is None:
            thinking_budget = -1  # Let model control budget
            logger.debug(f"[GENERATION] Thinking budget set to model control (-1)")
        else:
            logger.debug(f"[GENERATION] Thinking budget set to {thinking_budget}")
        
        custom_labels_str = credentials.get("vertex_custom_labels")
        labels = None
        if custom_labels_str:
            try:
                labels = json.loads(custom_labels_str)
                logger.debug(f"[GENERATION] Custom labels parsed: {len(labels) if labels else 0} labels")
            except (json.JSONDecodeError, TypeError):
                labels = None
                logger.warning(f"[GENERATION] Failed to parse custom labels: {custom_labels_str}")

        # Convert messages to genai format
        message_conversion_start = time.time()
        history = []
        system_instruction = ""
        
        for i, msg in enumerate(prompt_messages):
            if isinstance(msg, SystemPromptMessage):
                system_instruction = msg.content if isinstance(msg.content, str) else str(msg.content)
                logger.debug(f"[GENERATION] System instruction found at message {i}: {len(system_instruction)} chars")
            else:
                content = self._format_message_to_genai_content(msg)
                history.append(content)
                logger.debug(f"[GENERATION] Converted message {i} ({type(msg).__name__}) to genai format")
        
        message_conversion_time = time.time() - message_conversion_start
        logger.debug(f"[GENERATION] Message conversion completed in {message_conversion_time:.3f}s: {len(history)} history items")

        # Set up authentication and client
        auth_start = time.time()
        client = self._create_authenticated_client(model, credentials)
        auth_time = time.time() - auth_start
        logger.debug(f"[GENERATION] Client authentication completed in {auth_time:.3f}s")

        # Prepare tools
        tools_start = time.time()
        
        # Note: Google AI/Vertex AI API doesn't support mixing Google Search tools with function tools
        # We prioritize function tools over Google Search when both are requested
        if tools:
            function_tools = self._convert_tools_to_genai_tool(tools)
            logger.debug(f"[GENERATION] Converted {len(function_tools)} function tools to genai format")
            if dynamic_threshold:
                logger.debug(f"[GENERATION] Google Search grounding disabled due to function tools being present (API limitation)")
        elif dynamic_threshold:
            google_search_tool = Tool(google_search=GoogleSearch())
            logger.debug(f"[GENERATION] Added Google Search tool for grounding")
        
        tools_time = time.time() - tools_start
        logger.info(f"[GENERATION] Tools preparation completed in {tools_time:.3f}s: {len(function_tools)} total tools")

        # Handle context caching
        caching_start = time.time()
        cached_content_name = None
        if enable_context_caching and (system_instruction or function_tools):
            logger.info(f"[GENERATION] Context caching enabled - checking conversation for files")
            
            # Check if conversation contains files
            has_files = self._has_files_in_conversation(prompt_messages)
            
            if has_files:
                # Conversation-level caching: include entire conversation history
                conversation_id = user_id or f"conv-{int(time.time())}"  # Use user_id as conversation ID or generate one
                
                # Include all conversation content for caching (users, assistant, and files)
                conversation_contents = []
                for msg in prompt_messages:
                    if not isinstance(msg, SystemPromptMessage):  # Exclude system message as it's handled separately
                        content = self._format_message_to_genai_content(msg)
                        conversation_contents.append(content)
                
                logger.info(f"[GENERATION] Files detected - using conversation-level caching with {len(conversation_contents)} content items, conversation_id={conversation_id}")
                cached_content_name = self._create_or_get_cache(
                    client, model, credentials, system_instruction, function_tools, 
                    conversation_id=conversation_id, conversation_contents=conversation_contents
                )
            else:
                # App-level caching: only system instruction and tools
                logger.info(f"[GENERATION] No files detected - using app-level caching")
                cached_content_name = self._create_or_get_cache(
                    client, model, credentials, system_instruction, function_tools
                )
        elif enable_context_caching:
            logger.warning(f"[GENERATION] Context caching enabled but no system instruction or tools available")
        else:
            logger.info(f"[GENERATION] Context caching disabled")
            
        caching_time = time.time() - caching_start
        cache_status = "SUCCESS" if cached_content_name else "NONE"
        logger.info(f"[GENERATION] Caching operation completed in {caching_time:.3f}s: {cache_status}")
        if cached_content_name:
            logger.debug(f"[GENERATION] Using cached content: {cached_content_name}")

        # Prepare generation config
        config_prep_start = time.time()
        config_dict = {}
        if thinking_budget == 0:
            # Thinking is disabled via no_thinking parameter
            thinking_config = None
            logger.debug(f"[GENERATION] Thinking disabled")
        elif thinking_budget != -1:
            thinking_config = ThinkingConfig(thinking_budget=thinking_budget)
            logger.debug(f"[GENERATION] Thinking enabled with budget {thinking_budget}")
        else:
            thinking_config = None
            logger.debug(f"[GENERATION] Thinking enabled with model-controlled budget")
            
        if response_schema:
            config_dict["response_schema"] = response_schema
            config_dict["response_mime_type"] = "application/json"
            logger.debug(f"[GENERATION] JSON response schema configured")
        elif "response_mime_type" in config_kwargs:
            config_dict["response_mime_type"] = config_kwargs.pop("response_mime_type")
            logger.debug(f"[GENERATION] Response MIME type: {config_dict['response_mime_type']}")
            
        if stop and isinstance(stop, list):
            config_dict["stop_sequences"] = stop
            logger.debug(f"[GENERATION] Stop sequences: {stop}")
        elif stop:
            # Handle case where stop might not be a list
            config_dict["stop_sequences"] = [stop] if isinstance(stop, str) else []
            logger.debug(f"[GENERATION] Stop sequences (converted): {config_dict['stop_sequences']}")
            
        # Copy other model parameters
        for key, value in config_kwargs.items():
            if key not in ["stop_sequences"]:  # Already handled
                config_dict[key] = value

        config_prep_time = time.time() - config_prep_start
        logger.debug(f"[GENERATION] Generation config prepared in {config_prep_time:.3f}s")

        # Generate content
        content_generation_start = time.time()
        generation_config = GenerateContentConfig(
            tools=function_tools if function_tools and not cached_content_name else None,
            response_modalities=["TEXT"],
            system_instruction=system_instruction if not cached_content_name else None,
            thinking_config=thinking_config,
            cached_content=cached_content_name,
            **config_dict
        )
        
        logger.info(f"[GENERATION] Starting content generation: stream={stream}")
        logger.debug(f"[GENERATION] Final config: tools={len(function_tools) if function_tools and not cached_content_name else 0}, cached_content={bool(cached_content_name)}")
        
        setup_time = time.time() - generation_start
        logger.info(f"[GENERATION] Setup completed in {setup_time:.3f}s - starting model call")
        
        if stream:
            response = client.models.generate_content_stream(
                model=model,
                contents=history,
                config=generation_config
            )
            content_generation_time = time.time() - content_generation_start
            total_time = time.time() - generation_start
            logger.info(f"[GENERATION] Stream response initiated in {content_generation_time:.3f}s (total: {total_time:.3f}s)")
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages, system_instruction)
        else:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=generation_config
            )
            content_generation_time = time.time() - content_generation_start
            total_time = time.time() - generation_start
            logger.info(f"[GENERATION] Non-stream response completed in {content_generation_time:.3f}s (total: {total_time:.3f}s)")
            return self._handle_generate_response(model, credentials, response, prompt_messages)

    def _handle_generate_response(
        self, model: str, credentials: dict, response, prompt_messages: list[PromptMessage]
    ) -> LLMResult:
        """
        Handle llm response

        :param model: model name
        :param credentials: credentials
        :param response: response from genai
        :param prompt_messages: prompt messages
        :return: llm response
        """
        import time
        response_start = time.time()
        
        logger.info(f"[RESPONSE] Processing non-stream response for model {model}")
        
        assistant_prompt_message = AssistantPromptMessage(content="", tool_calls=[])
        
        # Handle genai response format
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            logger.debug(f"[RESPONSE] Processing candidate with {len(candidate.content.parts) if hasattr(candidate, 'content') and candidate.content.parts else 0} parts")
            
            if hasattr(candidate, 'content') and candidate.content.parts:
                part = candidate.content.parts[0]
                if hasattr(part, 'function_call') and part.function_call:
                    tool_call = AssistantPromptMessage.ToolCall(
                        id=part.function_call.name,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=part.function_call.name,
                            arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                        ),
                    )
                    assistant_prompt_message.tool_calls.append(tool_call)
                    logger.debug(f"[RESPONSE] Function call detected: {part.function_call.name}")
                elif hasattr(part, 'text') and part.text:
                    assistant_prompt_message.content = part.text
                    logger.debug(f"[RESPONSE] Text response: {len(part.text)} characters")
        else:
            logger.warning(f"[RESPONSE] No candidates found in response")

        # Use actual token counts from Google API if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            logger.info(f"[RESPONSE] Token usage from API: prompt={prompt_tokens}, completion={completion_tokens}")
        else:
            # Fallback to estimation if usage_metadata is not available
            prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
            completion_tokens = self.get_num_tokens(model, credentials, [assistant_prompt_message])
            logger.info(f"[RESPONSE] Token usage estimated: prompt={prompt_tokens}, completion={completion_tokens}")
            
        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)
        
        response_time = time.time() - response_start
        result = LLMResult(model=model, prompt_messages=prompt_messages, message=assistant_prompt_message, usage=usage)
        
        logger.info(f"[RESPONSE] Non-stream response processed in {response_time:.3f}s")
        logger.debug(f"[RESPONSE] Final message: content_length={len(str(assistant_prompt_message.content))}, tool_calls={len(assistant_prompt_message.tool_calls)}")
        
        return result

    def _handle_generate_stream_response(
        self, model: str, credentials: dict, response, prompt_messages: list[PromptMessage], system_instruction: str
    ) -> Generator:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :param system_instruction: system instruction
        :return: llm response chunk generator result
        """
        import time
        stream_start = time.time()
        
        logger.info(f"[STREAM] Starting stream response processing for model {model}")
        
        index = -1
        is_first_gemini2_response = True
        chunk_count = 0
        total_text_length = 0
        
        for chunk in response:
            chunk_start = time.time()
            chunk_count += 1
            
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                logger.debug(f"[STREAM] Processing chunk #{chunk_count} with {len(candidate.content.parts)} parts")
                
                for part in candidate.content.parts:
                    assistant_prompt_message = AssistantPromptMessage(content="", tool_calls=[])

                    if hasattr(part, 'function_call') and part.function_call:
                        assistant_prompt_message.tool_calls.append(
                            AssistantPromptMessage.ToolCall(
                                id=part.function_call.name,
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=part.function_call.name,
                                    arguments=json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}",
                                ),
                            )
                        )
                        logger.debug(f"[STREAM] Function call in chunk #{chunk_count}: {part.function_call.name}")
                    elif hasattr(part, 'text') and part.text:
                        assistant_prompt_message.content += part.text
                        total_text_length += len(part.text)
                        logger.debug(f"[STREAM] Text in chunk #{chunk_count}: {len(part.text)} chars")

                    index += 1
                    if not hasattr(candidate, "finish_reason") or not candidate.finish_reason:
                        chunk_time = time.time() - chunk_start
                        logger.debug(f"[STREAM] Yielding intermediate chunk #{chunk_count} (processed in {chunk_time:.3f}s)")
                        
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(index=index, message=assistant_prompt_message),
                        )
                    else:
                        # Final chunk with usage and grounding information
                        logger.info(f"[STREAM] Processing final chunk #{chunk_count} with finish_reason: {candidate.finish_reason}")
                        
                        # Use actual token counts from Google API if available in streaming response
                        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                            prompt_tokens = chunk.usage_metadata.prompt_token_count
                            completion_tokens = chunk.usage_metadata.candidates_token_count
                            logger.info(f"[STREAM] Token usage from API: prompt={prompt_tokens}, completion={completion_tokens}")
                        else:
                            # Fallback to estimation if usage_metadata is not available
                            prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
                            completion_tokens = self.get_num_tokens(model, credentials, [assistant_prompt_message])
                            logger.info(f"[STREAM] Token usage estimated: prompt={prompt_tokens}, completion={completion_tokens}")
                        usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

                        reference_lines = []
                        grounding_chunks = []
                        try:
                            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                                grounding_chunks = candidate.grounding_metadata.grounding_chunks
                                logger.debug(f"[STREAM] Found {len(grounding_chunks)} grounding chunks")
                        except AttributeError:
                            grounding_chunks = []
                            logger.debug(f"[STREAM] No grounding metadata found")

                        if grounding_chunks:
                            for i, gc in enumerate(grounding_chunks):
                                try:
                                    title = gc.web.title
                                    uri = gc.web.uri
                                except AttributeError:
                                    web_info = gc.get("web", {}) if hasattr(gc, 'get') else {}
                                    title = web_info.get("title")
                                    uri = web_info.get("uri")
                                if title and uri:
                                    reference_lines.append(f"<li><a href='{uri}'>{title}</a></li>")
                                    logger.debug(f"[STREAM] Grounding source #{i+1}: {title}")

                        if reference_lines:
                            reference_lines.insert(0, "<ol>")
                            reference_lines.append("</ol>")
                            reference_section = "\n\nGrounding Sources\n" + "\n".join(reference_lines)
                            logger.info(f"[STREAM] Added {len(reference_lines)-2} grounding sources")
                        else:
                            reference_section = ""
                            logger.debug(f"[STREAM] No grounding sources to add")
                            
                        if is_first_gemini2_response and model.startswith("gemini-2.") and system_instruction:
                            integrated_text = f"{assistant_prompt_message.content}"
                            is_first_gemini2_response = False
                            logger.debug(f"[STREAM] First Gemini 2.x response with system instruction")
                        else:
                            integrated_text = f"{assistant_prompt_message.content}{reference_section}"
                        assistant_message_with_refs = AssistantPromptMessage(content=integrated_text, tool_calls=assistant_prompt_message.tool_calls)

                        chunk_time = time.time() - chunk_start
                        total_time = time.time() - stream_start
                        
                        logger.info(f"[STREAM] Final chunk #{chunk_count} processed in {chunk_time:.3f}s")
                        logger.info(f"[STREAM] Stream completed: {chunk_count} chunks, {total_text_length} total chars, {total_time:.3f}s total")

                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=index,
                                message=assistant_message_with_refs,
                                finish_reason=str(candidate.finish_reason),
                                usage=usage,
                            ),
                        )
            else:
                logger.warning(f"[STREAM] Chunk #{chunk_count} has no candidates")
                
        if chunk_count == 0:
            logger.warning(f"[STREAM] No chunks received from stream response")

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        """
        Convert a single message to a string.

        :param message: PromptMessage to convert.
        :return: String representation of the message.
        """
        human_prompt = "\n\nuser:"
        ai_prompt = "\n\nmodel:"
        content = message.content
        if isinstance(content, list):
            content = "".join((c.data for c in content if c.type != PromptMessageContentType.IMAGE))
        if isinstance(message, UserPromptMessage):
            message_text = f"{human_prompt} {content}"
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, SystemPromptMessage | ToolPromptMessage):
            message_text = f"{human_prompt} {content}"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_message_to_genai_content(self, message: PromptMessage) -> Content:
        """
        Format a single message into genai.Content for Google API

        :param message: one PromptMessage
        :return: genai Content representation of message
        """
        if isinstance(message, UserPromptMessage):
            parts = []
            if isinstance(message.content, str):
                parts.append(Part(text=message.content))
            elif isinstance(message.content, list):
                for c in message.content:
                    if c.type == PromptMessageContentType.TEXT:
                        parts.append(Part(text=c.data))
                    elif c.type in [
                        PromptMessageContentType.IMAGE,
                        PromptMessageContentType.DOCUMENT,
                        PromptMessageContentType.AUDIO,
                        PromptMessageContentType.VIDEO
                    ]:
                        # For media content, create inline data part
                        import base64
                        from google.genai.types import Blob
                        if hasattr(c, 'base64_data'):
                            data = base64.b64decode(c.base64_data)
                        else:
                            data = base64.b64decode(c.data) if isinstance(c.data, str) else c.data
                        mime_type = getattr(c, 'mime_type', 'image/jpeg')
                        parts.append(Part(inline_data=Blob(mime_type=mime_type, data=data)))
                    else:
                        raise ValueError(f"Unsupported content type: {c.type}")
            genai_content = Content(role="user", parts=parts)
            return genai_content
        elif isinstance(message, AssistantPromptMessage):
            if message.tool_calls:
                parts = []
                for tool_call in message.tool_calls:
                    parts.append(Part(
                        function_call=FunctionCall(
                            name=tool_call.function.name,
                            args=json.loads(tool_call.function.arguments)
                        )
                    ))
                genai_content = Content(role="model", parts=parts)
            else:
                content_text = message.content if isinstance(message.content, str) else str(message.content)
                genai_content = Content(role="model", parts=[Part(text=content_text)])
            return genai_content
        elif isinstance(message, ToolPromptMessage):
            from google.genai.types import FunctionResponse
            genai_content = Content(
                role="function",
                parts=[
                    Part(function_response=FunctionResponse(
                        name=message.name or "",
                        response={"result": message.content}
                    ))
                ],
            )
            return genai_content
        else:
            raise ValueError(f"Got unknown type {message}")

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the ermd = gml.GenerativeModel(model) error type thrown to the caller
        The value is the md = gml.GenerativeModel(model) error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke emd = gml.GenerativeModel(model) error mapping
        """
        return {
            InvokeConnectionError: [exceptions.RetryError],
            InvokeServerUnavailableError: [
                exceptions.ServiceUnavailable,
                exceptions.InternalServerError,
                exceptions.BadGateway,
                exceptions.GatewayTimeout,
                exceptions.DeadlineExceeded,
            ],
            InvokeRateLimitError: [exceptions.ResourceExhausted, exceptions.TooManyRequests],
            InvokeAuthorizationError: [
                exceptions.Unauthenticated,
                exceptions.PermissionDenied,
                exceptions.Unauthenticated,
                exceptions.Forbidden,
            ],
            InvokeBadRequestError: [
                exceptions.BadRequest,
                exceptions.InvalidArgument,
                exceptions.FailedPrecondition,
                exceptions.OutOfRange,
                exceptions.NotFound,
                exceptions.MethodNotAllowed,
                exceptions.Conflict,
                exceptions.AlreadyExists,
                exceptions.Aborted,
                exceptions.LengthRequired,
                exceptions.PreconditionFailed,
                exceptions.RequestRangeNotSatisfiable,
                exceptions.Cancelled,
            ],
        }

    def _convert_schema_for_vertex(self, schema):
        """
        Convert JSON schema to Vertex AI's expected format (uppercase types)
        and validate structure. Automatically converts specific 'type' arrays:
        - ["string", "null"] -> type: "STRING", nullable: true
        - ["number", "string"] or ["string", "number"] -> type: "STRING"

        :param schema: The original JSON schema (dict, list, string, etc.)
        :return: Converted schema for Vertex AI or raises ValueError for invalid structures.
        :raises ValueError: If the schema contains unsupported structures or types.
        """
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"Input schema string is not valid JSON: {e}") from e

        if isinstance(schema, dict):
            converted_schema = {}
            # Define keys that expect nested schemas (dict)
            nested_schema_keys = {"properties", "items"}
            # Define keys that expect lists
            list_keys = {"enum", "required"}
            # Define keys that expect strings
            string_keys = {"description", "format"} # Removed 'type' for special handling
            # Define keys that expect numbers
            number_keys = {"minimum", "maximum"}
            # Define keys that expect integers
            integer_keys = {"minItems", "maxItems"}
            # Define keys that expect booleans
            boolean_keys = {"nullable"}
            # Vertex AI specific key
            vertex_specific_keys = {"propertyOrdering"} # Expects a list

            # All known keys *except* 'type' which has special handling below
            known_keys_minus_type = (
                nested_schema_keys | list_keys | string_keys | number_keys |
                integer_keys | boolean_keys | vertex_specific_keys
            )

            # --- Special Handling for 'type' key ---
            if "type" in schema:
                value = schema["type"]
                if isinstance(value, str):
                    # Standard case: single string type
                    converted_schema["type"] = value.upper()
                elif isinstance(value, list):
                    # Handle specific list patterns
                    # Use lowercased set for order-insensitive comparison
                    type_set = set(item.lower() if isinstance(item, str) else item for item in value)

                    if type_set == {"string", "null"}:
                        # Convert ["string", "null"] to type: STRING, nullable: true
                        converted_schema["type"] = "STRING"
                        converted_schema["nullable"] = True
                    elif type_set == {"number", "string"}:
                         # Convert ["number", "string"] to type: STRING
                         converted_schema["type"] = "STRING"
                    # Add more elif conditions here for other list types if needed in the future
                    # Example: elif type_set == {"integer", "null"}:
                    #             converted_schema["type"] = "INTEGER"
                    #             converted_schema["nullable"] = True
                    else:
                        # It's a list, but not one we know how to auto-convert
                        raise ValueError(
                            f"Invalid schema: Unsupported list value for 'type' key: {value}. "
                            f"Vertex AI expects a single string type. "
                            f"Auto-conversion only supported for ['string', 'null'] and ['number', 'string']."
                        )
                else:
                    # It's not a string and not a list - definitely invalid for 'type'
                    raise ValueError(
                        f"Invalid schema: Value for 'type' key must be a string or a supported list "
                        f"(like ['string', 'null']), but got {type(value).__name__}. Schema snippet: {{'type': {value}}}"
                    )
            # --- End Special Handling for 'type' key ---


            # --- Process other keys ---
            for key, value in schema.items():
                if key == "type":
                    continue # Already handled above

                if key in nested_schema_keys:
                    if isinstance(value, dict):
                         if key == "properties":
                             converted_props = {}
                             for prop_name, prop_def in value.items():
                                 # Recursively convert property definitions
                                 converted_props[prop_name] = self._convert_schema_for_vertex(prop_def)
                             converted_schema[key] = converted_props
                         elif key == "items":
                              # Recursively convert item definition
                              converted_schema[key] = self._convert_schema_for_vertex(value)
                    else:
                         raise ValueError(
                             f"Invalid schema: Value for '{key}' key must be a dictionary, "
                             f"but got {type(value).__name__}. Schema snippet: {{'{key}': {value}}}"
                         )
                elif key in list_keys | vertex_specific_keys:
                     if isinstance(value, list):
                         if key == "required" and not all(isinstance(item, str) for item in value):
                             raise ValueError(f"Invalid schema: All items in 'required' list must be strings.")
                         # Copy list values directly for enum, required, propertyOrdering
                         converted_schema[key] = value
                     else:
                         raise ValueError(
                             f"Invalid schema: Value for '{key}' key must be a list, "
                             f"but got {type(value).__name__}. Schema snippet: {{'{key}': {value}}}"
                         )
                elif key in known_keys_minus_type:
                     # For other known keys, copy the value directly.
                     if key == "nullable" and not isinstance(value, bool):
                          # Allow nullable to be set by the type conversion logic above
                          if key not in converted_schema: # Only raise if not already set by type logic
                            raise ValueError(f"Invalid schema: Value for 'nullable' must be boolean.")
                     elif key == "nullable" and key in converted_schema:
                         # If type logic set nullable=True, don't overwrite with potentially false value from original schema
                         pass
                     else:
                        converted_schema[key] = value
                else:
                    # Handle unknown keys: Ignore them as they are likely unsupported by Vertex AI
                    # print(f"Warning: Unknown schema key '{key}' encountered. Ignoring.")
                    pass # Ignore unknown keys

            return converted_schema

        elif isinstance(schema, list):
            # Handle top-level lists (e.g., schema defining an array directly)
            return [self._convert_schema_for_vertex(item) for item in schema]

        else:
            # Handle primitive types (int, str, bool, None, float) - return as is
            if isinstance(schema, (int, str, bool, float)) or schema is None:
                return schema
            else:
                 raise ValueError(f"Invalid schema component type: {type(schema).__name__}")
