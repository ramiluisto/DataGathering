"""Retry utilities with exponential backoff."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

import httpx
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retryable exceptions
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.HTTPError,
    httpx.TimeoutException,
    ConnectionError,
    TimeoutError,
    OSError,
)


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types to retry on

    Example:
        @with_retry(max_attempts=3, delay=1.0)
        def fetch_data(url: str) -> str:
            response = httpx.get(url)
            response.raise_for_status()
            return response.text
    """
    exceptions = retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in Retrying(
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
                    return func(*args, **kwargs)
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator


def retry_call(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> T:
    """Call a function with retry logic.

    Args:
        func: Function to call
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
        result = retry_call(
            requests.get,
            args=("https://api.example.com/data",),
            kwargs={"timeout": 30},
            max_attempts=3,
        )
    """
    kwargs = kwargs or {}
    exceptions = retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS

    for attempt in Retrying(
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
            return func(*args, **kwargs)

    raise RuntimeError("Retry loop exited unexpectedly")


class RetryConfig:
    """Configuration for retry behavior."""

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
        self.retryable_exceptions = retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call a function with this retry configuration.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        return retry_call(
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
        return with_retry(
            max_attempts=self.max_attempts,
            delay=self.delay,
            exponential_base=self.exponential_base,
            max_delay=self.max_delay,
            retryable_exceptions=self.retryable_exceptions,
        )


def simple_retry(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> Callable[..., T]:
    """Simple retry wrapper with fixed delay.

    Args:
        func: Function to wrap
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop exited unexpectedly")

    return wrapper
