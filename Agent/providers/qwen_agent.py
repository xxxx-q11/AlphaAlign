"""
Tongyi Qianwen (Qwen) Agent Implementation
"""
from typing import List, Union, Any, Optional

# Lazy import dashscope to avoid making the entire module unavailable when not installed
try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class QwenAgent(BaseAgent):
    """Tongyi Qianwen Agent"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Qwen Agent
        
        Args:
            api_key: DashScope API Key
            model: Model name, e.g. qwen-turbo, qwen-plus, qwen-max
            base_url: Unused (DashScope uses a fixed URL)
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
        """
        if dashscope is None:
            raise ImportError(
                "dashscope package is not installed. Please run: pip install dashscope"
            )
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        dashscope.api_key = api_key
    
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Any]:
        """
        Send a chat request
        
        Args:
            messages: List of messages
            stream: Whether to return a streaming response
            **kwargs: Other parameters
            
        Returns:
            LLMResponse or streaming response object
        """
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Build parameters
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        # Merge extra parameters
        params.update(self.extra_params)
        params.update(kwargs)
        
        if stream:
            # Streaming call
            responses = Generation.call(**params, stream=True)
            return responses
        else:
            # Non-streaming call
            response = Generation.call(**params)
            
            if response.status_code == 200:
                result = response.output
                
                # Check if result is None
                if result is None:
                    raise Exception(f"Qwen API returned an empty response: {response}")
                
                # Handle different response formats
                # DashScope native format: result.choices[0].message.content
                # OpenAI-compatible format: result["choices"][0]["message"]["content"]
                if hasattr(result, 'choices'):
                    # DashScope native format
                    choices = result.choices
                    if choices and len(choices) > 0:
                        content = getattr(choices[0].message, 'content', '')
                        usage = getattr(result, 'usage', None)
                    else:
                        content = ''
                        usage = None
                elif isinstance(result, dict):
                    # OpenAI-compatible format or dict format
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                    else:
                        content = ""
                    usage = result.get("usage")
                else:
                    raise Exception(f"Unknown response format: {type(result)}, Content: {result}")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    metadata={"request_id": getattr(response, 'request_id', None)}
                )
            else:
                error_msg = getattr(response, 'message', f'Status code: {response.status_code}')
                raise Exception(f"Qwen API call failed: {error_msg}")
    
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        return LLMProvider.QWEN

