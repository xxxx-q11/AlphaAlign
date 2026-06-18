"""
Claude (Anthropic) Agent Implementation
"""
from typing import List, Union, Any, Optional
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class ClaudeAgent(BaseAgent):
    """Claude Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        **kwargs
    ):
        """
        Initialize Claude Agent
        
        Args:
            api_key: Anthropic API Key
            model: Model name, e.g. claude-3-5-sonnet-20241022, claude-3-opus-20240229
            base_url: API base URL (optional, for proxy)
            temperature: Temperature parameter
            max_tokens: Maximum number of generated tokens
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if Anthropic is None:
            raise ImportError("Please install the anthropic package: pip install anthropic")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = Anthropic(**client_kwargs)
    
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Send a chat request
        
        Args:
            messages: Message list
            stream: Whether to return as a stream
            **kwargs: Other parameters
            
        Returns:
            LLMResponse or streaming response object
        """
        # Claude requires separating system messages
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        user_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages if msg.role != "system"
        ]
        
        # Build parameters
        params = {
            "model": self.model,
            "messages": user_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens or 4096),
        }
        
        if system_messages:
            params["system"] = " ".join(system_messages)
        
        # Merge extra parameters
        params.update({k: v for k, v in self.extra_params.items() if k not in params})
        params.update({k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]})
        
        if stream:
            # Streaming call
            with self.client.messages.stream(**params) as stream:
                return stream
        else:
            # Non-streaming call
            response = self.client.messages.create(**params)
            
            content = ""
            if response.content:
                for block in response.content:
                    if block.type == "text":
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={"id": response.id}
            )
    
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        return LLMProvider.CLAUDE

