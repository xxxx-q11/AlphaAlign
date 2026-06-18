"""
OpenAI Agent Implementation
"""
from typing import List, Union, Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class OpenAIAgent(BaseAgent):
    """OpenAI Agent"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI Agent

        Args:
            api_key: OpenAI API Key
            model: Model name, e.g. gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview
            base_url: API base URL (optional, for proxy or compatible API)
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if OpenAI is None:
            raise ImportError("Please install the openai package: pip install openai")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

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
            stream: Whether to use streaming
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
            response = self.client.chat.completions.create(**params, stream=True)
            return response
        else:
            # Non-streaming call
            response = self.client.chat.completions.create(**params)

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={"finish_reason": choice.finish_reason}
            )

    def get_provider(self) -> LLMProvider:
        """Get the provider type"""
        return LLMProvider.OPENAI
