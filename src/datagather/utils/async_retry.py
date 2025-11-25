"""Async retry utilities with exponential backoff."""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retryable exceptions for async HTTP
DEFAULT_ASYNC_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.HTTPError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
    OSError,
    asyncio.TimeoutError,
)

# HTTP status codes that should trigger retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    """Exception for HTTP errors that should be retried."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


def async_with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types to retry on

    Example:
        @async_with_retry(max_attempts=3, delay=1.0)
        async def fetch_data(url: str) -> str:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
    """
    exceptions = retryable_exceptions or DEFAULT_ASYNC_RETRYABLE_EXCEPTIONS

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=delay,
                    min=delay,
                    max=max_delay,
                    exp_base=exponential_base,
                ),
                retry=retry_if_exception_type(exceptions),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator


async def async_retry_call(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> T:
    """Call an async function with retry logic.

    Args:
        func: Async function to call
        args: Positional arguments
        kwargs: Keyword arguments
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Function result

    Raises:
        Exception: The last exception if all retries fail

    Example:
        result = await async_retry_call(
            client.get,
            args=("https://api.example.com/data",),
            kwargs={"timeout": 30},
            max_attempts=3,
        )
    """
    kwargs = kwargs or {}
    exceptions = retryable_exceptions or DEFAULT_ASYNC_RETRYABLE_EXCEPTIONS

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=delay,
            min=delay,
            max=max_delay,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return await func(*args, **kwargs)

    raise RuntimeError("Retry loop exited unexpectedly")


class AsyncRetryConfig:
    """Configuration for async retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay in seconds
            exponential_base: Base for exponential backoff
            max_delay: Maximum delay between retries
            retryable_exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.retryable_exceptions = (
            retryable_exceptions or DEFAULT_ASYNC_RETRYABLE_EXCEPTIONS
        )

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call an async function with this retry configuration.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        return await async_retry_call(
            func,
            args=args,
            kwargs=kwargs,
            max_attempts=self.max_attempts,
            delay=self.delay,
            exponential_base=self.exponential_base,
            max_delay=self.max_delay,
            retryable_exceptions=self.retryable_exceptions,
        )

    def decorator(self) -> Callable:
        """Return a decorator with this retry configuration."""
        return async_with_retry(
            max_attempts=self.max_attempts,
            delay=self.delay,
            exponential_base=self.exponential_base,
            max_delay=self.max_delay,
            retryable_exceptions=self.retryable_exceptions,
        )


async def simple_async_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    delay: float = 1.0,
    **kwargs: Any,
) -> T:
    """Simple async retry wrapper with fixed delay.

    Args:
        func: Async function to call
        *args: Positional arguments
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    last_exception: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop exited unexpectedly")


def check_response_for_retry(response: httpx.Response) -> None:
    """Check HTTP response and raise RetryableHTTPError if should retry.

    Args:
        response: httpx Response object

    Raises:
        RetryableHTTPError: If status code indicates retry should occur
    """
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise RetryableHTTPError(
            response.status_code,
            f"Retryable HTTP error from {response.url}",
        )
    response.raise_for_status()
