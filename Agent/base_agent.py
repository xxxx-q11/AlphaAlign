"""
Base Agent class - Unified multi-LLM API interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from dataclasses import dataclass
import json


class LLMProvider(str, Enum):
    """LLM provider enumeration"""
    QWEN = "qwen"  # Qwen (Tongyi Qianwen)
    CLAUDE = "claude"  # Anthropic Claude
    GOOGLE = "google"  # Google Gemini
    OPENAI = "openai"  # OpenAI
    THIRD_PARTY = "third_party"  # Third-party platform


@dataclass
class Message:
    """Message data class"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """LLM response data class"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """LLM Agent base class"""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Agent
        
        Args:
            api_key: API key
            model: Model name
            base_url: API base URL (optional)
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Send chat request
        
        Args:
            messages: List of messages
            stream: Whether to return stream
            **kwargs: Other parameters
            
        Returns:
            LLMResponse or stream response object
        """
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        pass
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Format messages to common format
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of formatted message dictionaries
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Convenient call method
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            stream: Whether to return stream
            **kwargs: Other parameters
            
        Returns:
            LLMResponse or stream response object
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, stream=stream, **kwargs)
    
    def parse_json_response(self, response: Union[LLMResponse, str]) -> Dict[str, Any]:
        """
        Parse JSON format response from call() to dictionary
        
        Args:
            response: Return value from call() method, can be LLMResponse object or JSON string
            
        Returns:
            Parsed dictionary
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
            ValueError: If response format is incorrect or empty
        """
        import re
        
        # If it's an LLMResponse object, extract the content field
        if isinstance(response, LLMResponse):
            json_str = response.content
        # If it's a string, use it directly
        elif isinstance(response, str):
            json_str = response
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
        print(json_str)
        # Check if empty or only contains whitespace
        if not json_str or not json_str.strip():
            raise ValueError(
                f"Response content is empty, cannot parse JSON. Response type: {type(response)}, "
                f"Response content: {repr(json_str[:100]) if json_str else 'None'}"
            )
        
        # Remove leading and trailing whitespace
        json_str = json_str.strip()
        
        # Try to extract JSON from markdown code block (handle ```json ... ``` format)
        json_code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_code_block_pattern, json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        
        # Try to parse JSON directly
        try:
            json_str = json_str.replace("\\'", "'")  # Remove invalid single quote escape
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Provide more detailed error information
            error_msg = (
                f"JSON parsing failed: {e.msg}\n"
                f"Response content preview (first 1000 chars): {repr(json_str[:1000])}\n"
                f"Response content length: {len(json_str)} characters"
            )
            raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e
        

