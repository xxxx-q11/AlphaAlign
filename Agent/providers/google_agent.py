"""
Google Gemini Agent Implementation
"""
from typing import List, Union, Any, Optional
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from Agent.base_agent import BaseAgent, Message, LLMResponse, LLMProvider


class GoogleAgent(BaseAgent):
    """Google Gemini Agent"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Google Agent

        Args:
            api_key: Google API Key
            model: Model name, e.g. gemini-pro, gemini-pro-vision
            base_url: Not used (Google uses a fixed URL)
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__(api_key, model, base_url, temperature, max_tokens, **kwargs)
        if genai is None:
            raise ImportError("Please install google-generativeai package: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model)

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
            **kwargs: Additional parameters

        Returns:
            LLMResponse or streaming response object
        """
        # Format messages (Gemini uses a different format)
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })

        # Build generation configuration
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens

        # Merge additional parameters
        generation_config.update({k: v for k, v in self.extra_params.items() if k in ["top_p", "top_k"]})
        generation_config.update({k: v for k, v in kwargs.items() if k in ["top_p", "top_k", "temperature", "max_output_tokens"]})

        if stream:
            # Streaming call
            response = self.client.generate_content(
                formatted_messages,
                generation_config=generation_config,
                stream=True
            )
            return response
        else:
            # Non-streaming call
            response = self.client.generate_content(
                formatted_messages,
                generation_config=generation_config
            )

            content = ""
            if response.text:
                content = response.text

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else None,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else None,
                },
                metadata={"finish_reason": response.candidates[0].finish_reason if response.candidates else None}
            )

    def get_provider(self) -> LLMProvider:
        """Get the provider type"""
        return LLMProvider.GOOGLE
