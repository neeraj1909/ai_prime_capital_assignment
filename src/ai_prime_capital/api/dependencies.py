from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..bootstrap import RuntimeResources
from ..config import Settings


def get_resources(request: Request) -> RuntimeResources:
    resources = getattr(request.app.state, "resources", None)

    if not isinstance(resources, RuntimeResources):
        raise RuntimeError("application resources are not initialized")

    return resources


ResourcesDependency = Annotated[
    RuntimeResources,
    Depends(get_resources),
]


def get_settings(resources: ResourcesDependency) -> Settings:
    if resources.settings is None:
        raise RuntimeError("settings are unavailable")

    return resources.settings


SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]
