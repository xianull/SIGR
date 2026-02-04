"""
LLM Client for SIGR Framework

Provides OpenAI-compatible API client for LLM interactions.
Supports concurrent requests with connection pooling.

Dual-model architecture:
- Fast model: High-frequency tasks (gene descriptions)
- Strong model: Low-frequency reasoning tasks (reflection, strategy)
"""

import os
import logging
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client using OpenAI-compatible API.

    Supports any OpenAI-compatible endpoint (OpenAI, Azure, local, etc.)
    Thread-safe with connection pooling for concurrent requests.
    """

    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_connections: int = 20,
        timeout: float = 60.0
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: API base URL
            api_key: API key (falls back to OPENAI_API_KEY env var)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_connections: Maximum concurrent connections
            timeout: Request timeout in seconds
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install with: pip install openai>=1.0.0"
            )

        if httpx is None:
            raise ImportError(
                "httpx package is required. Install with: pip install httpx>=0.24.0"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        # Create HTTP client with connection pooling
        self._http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections
            ),
            timeout=httpx.Timeout(timeout)
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            http_client=self._http_client
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0

    def close(self):
        """关闭 HTTP 客户端，释放连接资源"""
        if hasattr(self, '_http_client') and self._http_client:
            try:
                self._http_client.close()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")

    def __del__(self):
        """析构时关闭连接"""
        self.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时关闭连接"""
        self.close()
        return False

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        self.call_count += 1

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Generate text with system and user prompts.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Generated text
        """
        self.call_count += 1

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise


class DualModelClient:
    """
    Dual-model LLM client for SIGR framework.

    Uses different models for different tasks:
    - Fast model: High-frequency tasks (gene description generation)
    - Strong model: Low-frequency reasoning (reflection, strategy optimization)

    This architecture balances cost, latency, and quality.
    """

    def __init__(
        self,
        # API configuration
        base_url: str = "https://yunwu.ai/v1",
        api_key: Optional[str] = None,
        # Fast model for generation (high frequency)
        fast_model: str = "gpt-4o-mini",
        fast_temperature: float = 0.7,
        fast_max_tokens: int = 1500,
        # Strong model for reasoning (low frequency)
        strong_model: str = "gemini-3-pro-preview",
        strong_temperature: float = 0.5,
        strong_max_tokens: int = 4000,
        # Connection settings
        max_connections: int = 20,
        timeout: float = 120.0
    ):
        """
        Initialize the dual-model client.

        Args:
            base_url: API base URL (should support both models)
            api_key: API key
            fast_model: Model for high-frequency generation tasks
            fast_temperature: Temperature for fast model
            fast_max_tokens: Max tokens for fast model
            strong_model: Model for reasoning/reflection tasks
            strong_temperature: Temperature for strong model
            strong_max_tokens: Max tokens for strong model
            max_connections: Maximum concurrent connections
            timeout: Request timeout
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install with: pip install openai>=1.0.0"
            )

        if httpx is None:
            raise ImportError(
                "httpx package is required. Install with: pip install httpx>=0.24.0"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        # Create HTTP client with connection pooling
        self._http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections
            ),
            timeout=httpx.Timeout(timeout)
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            http_client=self._http_client
        )

        # Model configurations
        self.fast_model = fast_model
        self.fast_temperature = fast_temperature
        self.fast_max_tokens = fast_max_tokens

        self.strong_model = strong_model
        self.strong_temperature = strong_temperature
        self.strong_max_tokens = strong_max_tokens

        # Statistics
        self.fast_call_count = 0
        self.strong_call_count = 0

        logger.info(
            f"DualModelClient initialized: "
            f"fast={fast_model}, strong={strong_model}"
        )

    def close(self):
        """关闭 HTTP 客户端，释放连接资源"""
        if hasattr(self, '_http_client') and self._http_client:
            try:
                self._http_client.close()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")

    def __del__(self):
        """析构时关闭连接"""
        self.close()

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时关闭连接"""
        self.close()
        return False

    def generate(self, prompt: str) -> str:
        """
        Generate text using fast model (default for description generation).

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        return self.generate_fast(prompt)

    def generate_fast(self, prompt: str) -> str:
        """
        Generate text using fast model (for high-frequency tasks).

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        self.fast_call_count += 1

        try:
            response = self.client.chat.completions.create(
                model=self.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.fast_temperature,
                max_tokens=self.fast_max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Fast model generation error: {e}")
            raise

    def generate_strong(self, prompt: str) -> str:
        """
        Generate text using strong model (for reasoning tasks).

        Use this for:
        - Actor reflection
        - Strategy optimization
        - Complex analysis

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        self.strong_call_count += 1

        try:
            response = self.client.chat.completions.create(
                model=self.strong_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.strong_temperature,
                max_tokens=self.strong_max_tokens
            )
            logger.debug(f"Strong model call #{self.strong_call_count}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Strong model generation error: {e}")
            raise

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        use_strong: bool = False
    ) -> str:
        """
        Generate text with system and user prompts.

        Args:
            system_prompt: System message
            user_prompt: User message
            use_strong: Whether to use strong model

        Returns:
            Generated text
        """
        model = self.strong_model if use_strong else self.fast_model
        temperature = self.strong_temperature if use_strong else self.fast_temperature
        max_tokens = self.strong_max_tokens if use_strong else self.fast_max_tokens

        if use_strong:
            self.strong_call_count += 1
        else:
            self.fast_call_count += 1

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error ({model}): {e}")
            raise

    def get_stats(self) -> dict:
        """Get call statistics."""
        return {
            'fast_calls': self.fast_call_count,
            'strong_calls': self.strong_call_count,
            'total_calls': self.fast_call_count + self.strong_call_count,
            'fast_model': self.fast_model,
            'strong_model': self.strong_model,
        }


def get_llm_client(
    base_url: str = "https://yunwu.ai/v1",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_connections: int = 20,
    timeout: float = 60.0
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        base_url: API base URL
        api_key: API key
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        max_connections: Maximum concurrent connections
        timeout: Request timeout in seconds

    Returns:
        LLMClient instance
    """
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_connections=max_connections,
        timeout=timeout
    )


def get_dual_model_client(
    base_url: str = "https://yunwu.ai/v1",
    api_key: Optional[str] = None,
    fast_model: str = "gpt-4o-mini",
    strong_model: str = "gemini-3-pro-preview",
    max_connections: int = 20,
    timeout: float = 120.0
) -> DualModelClient:
    """
    Factory function to create a dual-model LLM client.

    Args:
        base_url: API base URL
        api_key: API key
        fast_model: Model for high-frequency tasks
        strong_model: Model for reasoning tasks
        max_connections: Maximum concurrent connections
        timeout: Request timeout

    Returns:
        DualModelClient instance
    """
    return DualModelClient(
        base_url=base_url,
        api_key=api_key,
        fast_model=fast_model,
        strong_model=strong_model,
        max_connections=max_connections,
        timeout=timeout
    )


# Recommended model configurations
MODEL_CONFIGS = {
    'fast': {
        'gpt-4o-mini': {'temperature': 0.7, 'max_tokens': 1500},
        'gpt-4o': {'temperature': 0.7, 'max_tokens': 2000},
        'claude-3-haiku-20240307': {'temperature': 0.7, 'max_tokens': 1500},
    },
    'strong': {
        'gemini-3-pro-preview': {'temperature': 0.5, 'max_tokens': 4000},
        'gemini-2.5-pro-preview-05-06': {'temperature': 0.5, 'max_tokens': 4000},
        'gpt-4o': {'temperature': 0.5, 'max_tokens': 4000},
        'claude-3-5-sonnet-20241022': {'temperature': 0.5, 'max_tokens': 4000},
        'claude-sonnet-4-20250514': {'temperature': 0.5, 'max_tokens': 4000},
        'deepseek-reasoner': {'temperature': 0.5, 'max_tokens': 8000},
    }
}

