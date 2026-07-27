"""Intent handling for mapping natural-language tasks to scene objects."""

from .intent_handler import (
    HIDDEN_TARGET_OCCLUDER_MODE,
    IntentResult,
    SceneObject,
    resolve_intent,
)

__all__ = [
    "HIDDEN_TARGET_OCCLUDER_MODE",
    "IntentResult",
    "SceneObject",
    "resolve_intent",
]
