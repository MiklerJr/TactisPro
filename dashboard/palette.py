"""Shared colors for TactisPro's light and dark themes."""

from dataclasses import dataclass

FONT = "system-ui, sans-serif"


@dataclass(frozen=True)
class Palette:
    BACKGROUND: str
    SURFACE: str
    BORDER: str
    TEXT: str
    MUTED: str
    OBSERVED: str
    PLAYER_A: str
    PLAYER_B: str
    FORECAST: str
    PAST: str
    FORECAST_UP: str
    FORECAST_STABLE: str
    FORECAST_DOWN: str


DARK = Palette(
    "#181714", "#211e1a", "#3b352d", "#f3ebe0", "#b9aea1",
    "#80c8b9", "#a7bdee", "#e6a375", "#df9a70", "#b2a494", "#83d5a1", "#efaf5f", "#ff9a9c",
)
LIGHT = Palette(
    "#faf8f3", "#ffffff", "#d8d0c3", "#25241f", "#665e53",
    "#126958", "#275a9e", "#a34c25", "#a34429", "#776858", "#246b43", "#a35a12", "#a62939",
)
