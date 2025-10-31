"""Configuration helpers for AirSim drone control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


# Mapping between model target labels and semantic directions.
TARGET_TO_DIRECTION: Dict[int, str] = {
    0: "forward",
    1: "left",
    2: "right",
    3: "backward",
}

# Reverse lookup from direction name back to target label.
DIRECTION_TO_TARGET: Dict[str, int] = {name: label for label, name in TARGET_TO_DIRECTION.items()}

# Set of allowed direction keywords for quick membership checks.
ALLOWED_DIRECTIONS = frozenset(TARGET_TO_DIRECTION.values())


@dataclass(slots=True)
class DroneConfig:
    """Holds tunable parameters for drone motion in the AirSim simulator."""

    forward: float = 5.0
    left: float = 5.0
    right: float = 5.0
    backward: float = 5.0
    speed: float = 3.0
    hover_height: float = 10.0
    min_move_duration: float = 1.0

    def displacement_for_direction(self, direction: str) -> Tuple[float, float, float]:
        """Return the (dx, dy, dz) displacement for the provided direction keyword."""

        key = direction.lower()
        if key not in ALLOWED_DIRECTIONS:
            raise ValueError(f"Unsupported direction '{direction}'. Expected one of {sorted(ALLOWED_DIRECTIONS)}")

        if key == "forward":
            return (self.forward, 0.0, 0.0)
        if key == "backward":
            return (-self.backward, 0.0, 0.0)
        if key == "left":
            return (0.0, -self.left, 0.0)
        if key == "right":
            return (0.0, self.right, 0.0)

        raise AssertionError("Unexpected direction reached; this should be impossible.")

    @staticmethod
    def direction_from_target(target: int) -> str:
        """Translate a numeric target label into its corresponding direction name."""

        try:
            return TARGET_TO_DIRECTION[target]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Unknown target label '{target}'.") from exc

    @staticmethod
    def target_from_direction(direction: str) -> int:
        """Translate a direction name back into the numeric target label used by the model."""

        key = direction.lower()
        try:
            return DIRECTION_TO_TARGET[key]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported direction '{direction}'.") from exc


def validate_target_sequence(targets: Iterable[int]) -> None:
    """Validate that all target labels in *targets* map to known directions."""

    unknown = {t for t in targets if t not in TARGET_TO_DIRECTION}
    if unknown:
        raise ValueError(f"Encountered unknown targets: {sorted(unknown)}")


__all__ = [
    "DroneConfig",
    "TARGET_TO_DIRECTION",
    "DIRECTION_TO_TARGET",
    "ALLOWED_DIRECTIONS",
    "validate_target_sequence",
]