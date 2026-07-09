from abc import ABC, abstractmethod
import importlib
from types import MappingProxyType
from typing import Callable
import numpy as np

from polaris.config import PolicyArgs


_CLIENT_MODULE_BY_NAME = MappingProxyType(
    {
        "DroidDeltaJointPosition": "polaris.policy.droid_delta_position_client",
        "DroidJointPos": "polaris.policy.droid_jointpos_client",
        "DroidJointVelocity": "polaris.policy.droid_jointvelocity_client",
        "EgoLAPEefPose": "polaris.policy.lap_eef_pose_client",
    }
)


class InferenceClient(ABC):
    REGISTERED_CLIENTS = {}

    # def __init_subclass__(cls, client_name: str, *args, **kwargs) -> None:
    #     super().__init_subclass__(*args, **kwargs)
    #     InferenceClient.REGISTERED_CLIENTS[client_name] = cls

    @staticmethod
    def register(client_name: str) -> Callable[[type], type]:
        def decorator(cls: type):
            InferenceClient.REGISTERED_CLIENTS[client_name] = cls
            return cls

        return decorator

    @staticmethod
    def load_client_class(client_name: str) -> type["InferenceClient"]:
        """Import exactly the module bound to ``client_name`` and verify it."""

        if type(client_name) is not str:
            raise ValueError("Client name must be one exact string")
        module_name = _CLIENT_MODULE_BY_NAME.get(client_name)
        if module_name is None:
            raise ValueError(
                f"Client {client_name} not found. Available clients: "
                f"{sorted(_CLIENT_MODULE_BY_NAME)}"
            )
        importlib.import_module(module_name)
        client_class = InferenceClient.REGISTERED_CLIENTS.get(client_name)
        if (
            not isinstance(client_class, type)
            or not issubclass(client_class, InferenceClient)
            or client_class.__module__ != module_name
        ):
            raise RuntimeError(
                f"Client {client_name} did not register from exact module {module_name}"
            )
        return client_class

    @staticmethod
    def get_client(policy_args: PolicyArgs) -> "InferenceClient":
        client_class = InferenceClient.load_client_class(policy_args.client)
        return client_class(policy_args)

    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @property
    def rerender(self) -> bool:
        """
        Policy requests a rerender of the visualization. Optimization for less splat rendering
        for chunked policies. Can default to always True if optimization is not desired.
        """
        return True

    @abstractmethod
    def infer(
        self, obs, instruction, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Does inference on observation and returns action and visualization. If visualization is not needed, return None.
        """

        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode. Useful if policy is stateful.
        """
        pass


class FakeClient(InferenceClient):
    """
    Fake client that returns a dummy action and visualization.
    """

    def __init__(self, *args, **kwargs) -> None:
        return

    def infer(
        self, obs, instruction, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        import cv2

        external = obs["splat"]["external_cam"]
        wrist = obs["splat"]["wrist_cam"]
        external = cv2.resize(external, (224, 224))
        wrist = cv2.resize(wrist, (224, 224))
        both = np.concatenate([external, wrist], axis=1)
        return np.zeros((8,)), both

    def reset(self, *args, **kwargs):
        return
