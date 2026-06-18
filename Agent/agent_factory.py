"""
Agent factory functions - Create corresponding Agent instances based on configuration
"""
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from Agent.base_agent import BaseAgent, LLMProvider
from Agent.providers import (
    QwenAgent,
    ClaudeAgent,
    GoogleAgent,
    OpenAIAgent,
    ThirdPartyAgent,
)


def load_env_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Read env.yaml configuration file from config folder
    
    Args:
        config_path: Configuration file path (optional). If not provided, search in the following order:
                    1. config/env.yaml (repository root)
                    2. current working directory's config/env.yaml
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: Configuration file does not exist
        ValueError: YAML file parsing error
        IOError: Error occurred while reading file
        
    Example:
        >>> config = load_env_config()
        >>> api_key = config.get('api_key')
        >>> provider = config.get('provider')
    """
    if config_path is None:
        # Try multiple possible configuration file paths
        possible_paths = [
            # Repository local path
            Path(__file__).parent.parent / "config" / "env.yaml",
            # Relative path (from workspace directory)
            Path("config/env.yaml"),
        ]
        
        config_path = None
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
        
        if config_path is None:
            raise FileNotFoundError(
                f"Configuration file does not exist, tried the following paths: {[str(p) for p in possible_paths]}"
            )
    else:
        config_path = Path(config_path)
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    # Read YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"YAML file parsing error: {e}")
    except Exception as e:
        raise IOError(f"Error occurred while reading configuration file: {e}")


def create_agent(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseAgent:
    """
    Create Agent instance
    
    Args:
        provider: Provider name ("qwen", "claude", "google", "openai", "third_party")
        api_key: API key
        model: Model name (optional, uses default value)
        base_url: API base URL (optional, required for third-party platforms)
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens to generate
        **kwargs: Other parameters
        
    Returns:
        BaseAgent instance
        
    Examples:
        >>> # Create Qwen Agent
        >>> agent = create_agent("qwen", api_key="your-key", model="qwen-turbo")
        
        >>> # Create OpenAI Agent
        >>> agent = create_agent("openai", api_key="your-key", model="gpt-4")
        
        >>> # Create third-party platform Agent
        >>> agent = create_agent(
        ...     "third_party",
        ...     api_key="your-key",
        ...     model="custom-model",
        ...     base_url="https://api.example.com/v1"
        ... )
    """
    provider = provider.lower()
    
    # Default model configuration
    default_models = {
        "qwen": "qwen-turbo",
        "claude": "claude-3-5-sonnet-20241022",
        "google": "gemini-pro",
        "openai": "gpt-3.5-turbo",
    }
    
    if model is None:
        model = default_models.get(provider)
        if model is None:
            raise ValueError(f"Model name not specified, and {provider} has no default model")
    
    if provider == "qwen":
        if QwenAgent is None:
            raise ImportError("QwenAgent unavailable, please install dashscope: pip install dashscope")
        return QwenAgent(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "claude":
        if ClaudeAgent is None:
            raise ImportError("ClaudeAgent unavailable, please install anthropic: pip install anthropic")
        return ClaudeAgent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "google":
        if GoogleAgent is None:
            raise ImportError("GoogleAgent unavailable, please install google-generativeai: pip install google-generativeai")
        return GoogleAgent(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "openai":
        if OpenAIAgent is None:
            raise ImportError("OpenAIAgent unavailable, please install openai: pip install openai")
        return OpenAIAgent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "third_party":
        if ThirdPartyAgent is None:
            raise ImportError("ThirdPartyAgent unavailable")
        if not base_url:
            raise ValueError("Third-party platform Agent requires base_url")
        return ThirdPartyAgent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: qwen, claude, google, openai, third_party")


def create_agent_from_config(config: Dict[str, Any]) -> BaseAgent:
    """
    Create Agent from configuration dictionary
    
    Args:
        config: Configuration dictionary, must contain "provider" and "api_key"
        
    Returns:
        BaseAgent instance
        
    Example:
        >>> config = {
        ...     "provider": "openai",
        ...     "api_key": "sk-xxx",
        ...     "model": "gpt-4",
        ...     "temperature": 0.8
        ... }
        >>> agent = create_agent_from_config(config)
    """
    provider = config.pop("provider")
    api_key = config.pop("api_key")
    return create_agent(provider=provider, api_key=api_key, **config)
