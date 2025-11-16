"""Thin wrapper around the AirSim Python API for multirotor control."""

from __future__ import annotations

import importlib
import math
from contextlib import AbstractContextManager
from typing import Any, Optional


from drone_config import DroneConfig


class AirSimUnavailableError(RuntimeError):
    """Raised when the AirSim Python package is not available in the environment."""


def _load_airsim() -> Any:
    try:
        return importlib.import_module("airsim")
    except ImportError as exc:  # pragma: no cover - exercised only when AirSim missing
        raise AirSimUnavailableError(
            "The 'airsim' package is required but not installed. "
            "Install it with 'pip install airsim==1.8.1 --no-build-isolation'."
        ) from exc


class AirSimDroneController(AbstractContextManager["AirSimDroneController"]):
    """Utility that encapsulates connection and simple motion commands for a drone."""

    def __init__(self, config: Optional[DroneConfig] = None) -> None:
        self._airsim = _load_airsim()
        self.config = config or DroneConfig()
        self.client = self._airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # AirSim uses NED coordinates with negative Z above ground.
        self._hover_z = -abs(self.config.hover_height)

    # ------------------------------------------------------------------
    # Context manager helpers
    def __enter__(self) -> "AirSimDroneController":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:  # pragma: no cover - cleanup
        self.shutdown()
        return None

    # ------------------------------------------------------------------
    # Drone motion commands
    def takeoff_and_hover(self) -> None:
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self._hover_z, self.config.speed).join()

    def move_direction(self, direction: str) -> None:
        dx, dy, dz = self.config.displacement_for_direction(direction)
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance == 0:
            return

        speed = max(self.config.speed, 1e-3)
        duration = max(distance / speed, self.config.min_move_duration)
        vx = dx / duration
        vy = dy / duration
        target_z = self._hover_z + dz

        self.client.moveByVelocityZAsync(vx, vy, target_z, duration).join()
        self.client.hoverAsync().join()

    def land(self) -> None:
        self.client.landAsync().join()

    def shutdown(self) -> None:
        try:
            self.client.armDisarm(False)
        finally:
            self.client.enableApiControl(False)


__all__ = ["AirSimDroneController", "AirSimUnavailableError"]

