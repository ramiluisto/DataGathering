"""Rate limiting utilities for API requests."""

import time
from collections import deque
from threading import Lock
from typing import Optional


class RateLimiter:
    """Thread-safe rate limiter using sliding window.

    Example:
        limiter = RateLimiter(requests_per_minute=30)

        for url in urls:
            limiter.acquire()  # Blocks if rate limit exceeded
            response = requests.get(url)
    """

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self._timestamps: deque = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        """Acquire a request slot, blocking if necessary."""
        with self._lock:
            now = time.monotonic()
            minute_ago = now - 60

            # Remove timestamps older than 1 minute
            while self._timestamps and self._timestamps[0] < minute_ago:
                self._timestamps.popleft()

            # If at capacity, wait for oldest to expire
            if len(self._timestamps) >= self.requests_per_minute:
                sleep_time = self._timestamps[0] - minute_ago
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Recursive call after sleeping
                    return self.acquire()

            self._timestamps.append(now)

    def try_acquire(self) -> bool:
        """Try to acquire a request slot without blocking.

        Returns:
            True if slot acquired, False if rate limited
        """
        with self._lock:
            now = time.monotonic()
            minute_ago = now - 60

            # Remove timestamps older than 1 minute
            while self._timestamps and self._timestamps[0] < minute_ago:
                self._timestamps.popleft()

            # Check if at capacity
            if len(self._timestamps) >= self.requests_per_minute:
                return False

            self._timestamps.append(now)
            return True

    def wait_time(self) -> float:
        """Get time to wait until next slot available.

        Returns:
            Seconds to wait, 0 if slot available immediately
        """
        with self._lock:
            now = time.monotonic()
            minute_ago = now - 60

            # Remove timestamps older than 1 minute
            while self._timestamps and self._timestamps[0] < minute_ago:
                self._timestamps.popleft()

            if len(self._timestamps) < self.requests_per_minute:
                return 0.0

            return max(0.0, self._timestamps[0] - minute_ago)

    @property
    def current_rate(self) -> int:
        """Get current number of requests in the window."""
        with self._lock:
            now = time.monotonic()
            minute_ago = now - 60

            # Remove timestamps older than 1 minute
            while self._timestamps and self._timestamps[0] < minute_ago:
                self._timestamps.popleft()

            return len(self._timestamps)

    def reset(self) -> None:
        """Reset the rate limiter."""
        with self._lock:
            self._timestamps.clear()


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on response status.

    Backs off when hitting rate limits, speeds up when successful.
    """

    def __init__(
        self,
        requests_per_minute: int,
        min_rpm: int = 1,
        max_rpm: Optional[int] = None,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
    ):
        """Initialize adaptive rate limiter.

        Args:
            requests_per_minute: Initial requests per minute
            min_rpm: Minimum requests per minute
            max_rpm: Maximum requests per minute (default: 2x initial)
            backoff_factor: Factor to reduce rate on limit hit
            recovery_factor: Factor to increase rate on success
        """
        super().__init__(requests_per_minute)
        self._initial_rpm = requests_per_minute
        self._min_rpm = min_rpm
        self._max_rpm = max_rpm or (requests_per_minute * 2)
        self._backoff_factor = backoff_factor
        self._recovery_factor = recovery_factor
        self._consecutive_successes = 0

    def report_success(self) -> None:
        """Report a successful request."""
        with self._lock:
            self._consecutive_successes += 1
            if self._consecutive_successes >= 10:
                # Gradually increase rate
                new_rpm = min(
                    self._max_rpm,
                    int(self.requests_per_minute * self._recovery_factor),
                )
                if new_rpm != self.requests_per_minute:
                    self.requests_per_minute = new_rpm
                    self._consecutive_successes = 0

    def report_rate_limited(self) -> None:
        """Report a rate limit hit."""
        with self._lock:
            self._consecutive_successes = 0
            new_rpm = max(
                self._min_rpm,
                int(self.requests_per_minute * self._backoff_factor),
            )
            self.requests_per_minute = new_rpm

    def report_error(self) -> None:
        """Report a non-rate-limit error."""
        with self._lock:
            self._consecutive_successes = 0


class DelayedRateLimiter:
    """Simple rate limiter using fixed delays between requests.

    Useful for APIs that require a minimum delay (like arXiv's 3s requirement).
    """

    def __init__(self, delay_seconds: float):
        """Initialize delayed rate limiter.

        Args:
            delay_seconds: Minimum seconds between requests
        """
        self.delay_seconds = delay_seconds
        self._last_request: Optional[float] = None
        self._lock = Lock()

    def acquire(self) -> None:
        """Acquire a request slot, waiting for delay if necessary."""
        with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                elapsed = now - self._last_request
                if elapsed < self.delay_seconds:
                    time.sleep(self.delay_seconds - elapsed)
            self._last_request = time.monotonic()

    def reset(self) -> None:
        """Reset the rate limiter."""
        with self._lock:
            self._last_request = None
