"""Label conventions shared by every seg/ stage (masks, fusion, cleanup, training)."""

from __future__ import annotations

#: Mask value and 3D label meaning "no component" (bare body panel, background, unknown).
NONE_LABEL = -1
