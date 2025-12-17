"""Configuration helpers for AirSim drone control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


# Mapping between model target labels and directions.
# 1: Left Hand -> "left" (Rotation)
# 2: Right Hand -> "right" (Rotation)
# 3: Feet -> "backward"
# 4: Tongue -> "forward"
TARGET_TO_DIRECTION: Dict[int, str] = {
    1: "left",
    2: "right",
    3: "backward",
    4: "forward",
}

DIRECTION_TO_TARGET: Dict[str, int] = {name: label for label, name in TARGET_TO_DIRECTION.items()}

ALLOWED_DIRECTIONS = frozenset(TARGET_TO_DIRECTION.values())

# Key mapping for manual input
KEY_TO_TARGET: Dict[str, int] = {
    'a': 1,  # Left
    'd': 2,  # Right
    's': 3,  # Backward
    'w': 4   # Forward
}

@dataclass(slots=True)
class DroneConfig:
    """Holds tunable parameters for drone motion in the AirSim simulator."""

    forward: float = 5.0
    backward: float = 5.0
    rotation_angle: float = 30.0 # degrees
    speed: float = 3.0
    hover_height: float = 10.0
    min_move_duration: float = 1.0

    def displacement_for_direction(self, direction: str) -> Tuple[float, float, float, float]:
        """Return the (dx, dy, dz, yaw_rate) displacement for the provided direction keyword."""

        key = direction.lower()
        if key not in ALLOWED_DIRECTIONS:
            raise ValueError(f"Unsupported direction '{direction}'. Expected one of {sorted(ALLOWED_DIRECTIONS)}")

        # dx, dy, dz, yaw
        # For rotations (left/right), we set dx,dy=0 and provide a yaw value
        # For movement (forward/backward), we set yaw=0 and provide dx,dy values
        
        if key == "forward":
            return (self.forward, 0.0, 0.0, 0.0)
        if key == "backward":
            return (-self.backward, 0.0, 0.0, 0.0)
        
        # Left/Right are now rotations
        if key == "left":
            return (0.0, 0.0, 0.0, -self.rotation_angle) 
        if key == "right":
            return (0.0, 0.0, 0.0, self.rotation_angle)

        raise AssertionError("Unexpected direction reached; this should be impossible.")

    @staticmethod
    def direction_from_target(target: int) -> str:
        try:
            return TARGET_TO_DIRECTION[target]
        except KeyError as exc:  
            raise ValueError(f"Unknown target label '{target}'.") from exc

    @staticmethod
    def target_from_direction(direction: str) -> int:
        key = direction.lower()
        try:
            return DIRECTION_TO_TARGET[key]
        except KeyError as exc:  
            raise ValueError(f"Unsupported direction '{direction}'.") from exc
            
    @staticmethod
    def target_from_key(key: str) -> Optional[int]:
        return KEY_TO_TARGET.get(key.lower())


def validate_target_sequence(targets: Iterable[int]) -> None:
    unknown = {t for t in targets if t not in TARGET_TO_DIRECTION}
    if unknown:
        raise ValueError(f"Encountered unknown targets: {sorted(unknown)}")


__all__ = [
    "DroneConfig",
    "TARGET_TO_DIRECTION",
    "DIRECTION_TO_TARGET",
    "ALLOWED_DIRECTIONS",
    "KEY_TO_TARGET",
    "validate_target_sequence",
]
