"""Checkpoint system for resumable data gathering."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import orjson


class CheckpointManager:
    """Manages checkpoints for resumable data gathering.

    Checkpoints store:
    - Source name and status
    - Number of completed samples
    - Last processed ID
    - Source-specific cursor/state
    - Error log
    - Basic statistics

    Example:
        manager = CheckpointManager(Path("./data/checkpoints"))

        # Load existing checkpoint or start fresh
        state = manager.load("wikipedia") or {"completed_samples": 0}

        # Save progress periodically
        manager.save("wikipedia", {
            "completed_samples": 50,
            "last_processed_id": "wiki_12345",
            "seen_ids": ["id1", "id2", ...]
        })

        # Check if should save
        if manager.should_save(count=55, interval=50):
            manager.save("wikipedia", state)
    """

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, source: str) -> Path:
        """Get checkpoint file path for a source.

        Args:
            source: Source name

        Returns:
            Path to checkpoint file
        """
        return self.checkpoint_dir / f"{source}.checkpoint.json"

    def load(self, source: str) -> Optional[dict[str, Any]]:
        """Load checkpoint for a source.

        Args:
            source: Source name

        Returns:
            Checkpoint state dictionary, or None if not exists
        """
        path = self.get_path(source)
        if not path.exists():
            return None
        try:
            return orjson.loads(path.read_bytes())
        except (orjson.JSONDecodeError, IOError):
            return None

    def save(self, source: str, state: dict[str, Any]) -> None:
        """Save checkpoint state.

        Args:
            source: Source name
            state: State dictionary to save
        """
        state = {
            **state,
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if "started_at" not in state:
            state["started_at"] = state["updated_at"]

        path = self.get_path(source)
        path.write_bytes(
            orjson.dumps(state, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    def should_save(self, completed: int, interval: int) -> bool:
        """Check if checkpoint should be saved based on interval.

        Args:
            completed: Number of completed items
            interval: Save interval

        Returns:
            True if should save
        """
        return completed > 0 and completed % interval == 0

    def reset(self, source: Optional[str] = None) -> int:
        """Reset checkpoint(s).

        Args:
            source: Source name to reset, or None for all

        Returns:
            Number of checkpoints reset
        """
        count = 0
        if source:
            path = self.get_path(source)
            if path.exists():
                path.unlink()
                count = 1
        else:
            for path in self.checkpoint_dir.glob("*.checkpoint.json"):
                path.unlink()
                count += 1
        return count

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with their status.

        Returns:
            List of checkpoint summaries
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.checkpoint.json"):
            source = path.stem.replace(".checkpoint", "")
            state = self.load(source)
            if state:
                checkpoints.append({
                    "source": source,
                    "completed_samples": state.get("completed_samples", 0),
                    "target_samples": state.get("target_samples"),
                    "status": state.get("status", "unknown"),
                    "updated_at": state.get("updated_at"),
                    "error_count": len(state.get("errors", [])),
                })
        return checkpoints

    def get_status(self, source: str) -> dict[str, Any]:
        """Get detailed status for a source.

        Args:
            source: Source name

        Returns:
            Status dictionary
        """
        state = self.load(source)
        if not state:
            return {
                "source": source,
                "status": "not_started",
                "completed_samples": 0,
            }

        completed = state.get("completed_samples", 0)
        target = state.get("target_samples")

        if target:
            progress = (completed / target) * 100
            status = "completed" if completed >= target else "in_progress"
        else:
            progress = None
            status = state.get("status", "in_progress")

        return {
            "source": source,
            "status": status,
            "completed_samples": completed,
            "target_samples": target,
            "progress_percent": progress,
            "started_at": state.get("started_at"),
            "updated_at": state.get("updated_at"),
            "error_count": len(state.get("errors", [])),
            "stats": state.get("stats", {}),
        }

    def mark_started(self, source: str, target_samples: int) -> None:
        """Mark a source as started.

        Args:
            source: Source name
            target_samples: Target number of samples
        """
        state = self.load(source) or {}
        state.update({
            "status": "in_progress",
            "target_samples": target_samples,
            "completed_samples": state.get("completed_samples", 0),
        })
        self.save(source, state)

    def mark_completed(self, source: str) -> None:
        """Mark a source as completed.

        Args:
            source: Source name
        """
        state = self.load(source) or {}
        state["status"] = "completed"
        self.save(source, state)

    def mark_failed(self, source: str, error: str) -> None:
        """Mark a source as failed.

        Args:
            source: Source name
            error: Error message
        """
        state = self.load(source) or {}
        state["status"] = "failed"
        errors = state.get("errors", [])
        errors.append({
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        state["errors"] = errors[-100]  # Keep last 100 errors
        self.save(source, state)

    def add_error(self, source: str, doc_id: str, error: str) -> None:
        """Add an error to the checkpoint.

        Args:
            source: Source name
            doc_id: Document ID that failed
            error: Error message
        """
        state = self.load(source) or {}
        errors = state.get("errors", [])
        errors.append({
            "id": doc_id,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        state["errors"] = errors[-100]  # Keep last 100 errors
        self.save(source, state)

    def update_stats(self, source: str, stats: dict[str, Any]) -> None:
        """Update statistics for a source.

        Args:
            source: Source name
            stats: Statistics dictionary
        """
        state = self.load(source) or {}
        existing_stats = state.get("stats", {})
        existing_stats.update(stats)
        state["stats"] = existing_stats
        self.save(source, state)
