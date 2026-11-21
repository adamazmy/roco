"""pygame face: robot-style rounded-rectangle eyes with eyelid overlays.

pygame is imported lazily (inside methods), so importing this module is cheap and
works without the ``face`` extra; constructing/running a :class:`PygameFace` needs it.

Drawing model: each eye is drawn into its own small, supersampled surface: a rounded
rectangle in the eye colour, then background-coloured eyelid polygons on top to carve
the expression. The surface is smooth-scaled down (cheap anti-aliasing) and blitted.
The listening glow is a blurred copy added on top with ``BLEND_ADD``.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from roco.body.base import Face
from roco.body.expression import EyeShape, FaceAnimator, FaceFrame, clamp, lerp
from roco.bus import Emotion, State

log = logging.getLogger(__name__)

Color = tuple[int, int, int]

# Layout at the reference resolution; everything scales with min(W/800, H/480).
REF_W, REF_H = 800, 480
EYE_W, EYE_H = 168.0, 196.0
EYE_SPACING = 316.0  # centre to centre
EYE_Y = 0.45  # eye centre line, fraction of height
GAZE_PX = 64.0  # full gaze deflection
MOUTH_Y = 176.0  # below the eye centre line
SUPERSAMPLE = 2


def parse_color(value: Any, default: Color = (0, 0, 0)) -> Color:
    """'#00E5FF', '00e5ff', (0, 229, 255) or [0, 229, 255] -> (0, 229, 255)."""
    try:
        if isinstance(value, str):
            text = value.strip().lstrip("#")
            if len(text) == 3:
                text = "".join(c * 2 for c in text)
            if len(text) != 6:
                raise ValueError(value)
            return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        r, g, b = (int(c) for c in value)
        return (clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255))  # type: ignore[return-value]
    except (TypeError, ValueError):
        log.warning("Invalid colour %r, using %s", value, default)
        return default


def mix(a: Color, b: Color, t: float) -> Color:
    t = clamp(t)
    return (round(lerp(a[0], b[0], t)), round(lerp(a[1], b[1], t)), round(lerp(a[2], b[2], t)))


@dataclass(frozen=True)
class FaceSettings:
    width: int = 800
    height: int = 480
    fullscreen: bool = False
    fps: int = 30
    eye_color: Color = (0x00, 0xE5, 0xFF)
    background: Color = (0, 0, 0)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None, **overrides: Any) -> FaceSettings:
        """Accepts the full config or just its ``face`` section."""
        cfg = cfg or {}
        section = cfg.get("face", cfg)
        section = section if isinstance(section, Mapping) else {}
        values = {
            "width": int(section.get("width", cls.width)),
            "height": int(section.get("height", cls.height)),
            "fullscreen": bool(section.get("fullscreen", cls.fullscreen)),
            "fps": max(1, int(section.get("fps", cls.fps))),
            "eye_color": parse_color(section.get("eye_color", "#00E5FF"), cls.eye_color),
            "background": parse_color(section.get("background", "#000000"), cls.background),
        }
        values.update(overrides)
        return cls(**values)


# --------------------------------------------------------------------------- geometry
# Pure helpers (no pygame) so lid shapes are unit-testable.


@dataclass(frozen=True)
class EyeBox:
    cx: float
    cy: float
    w: float
    h: float
    outer: int  # -1 when the eye's outer corner is on the left of the screen, +1 on the right


def eye_boxes(frame: FaceFrame, width: int, height: int) -> tuple[EyeBox, EyeBox]:
    s = min(width / REF_W, height / REF_H)
    p = frame.params
    gx = clamp(p.gaze_x + frame.offset_x, -1.5, 1.5) * GAZE_PX * s
    gy = clamp(p.gaze_y + frame.offset_y, -1.5, 1.5) * GAZE_PX * 0.7 * s
    openness = frame.eye_openness()
    boxes = []
    for shape, side in ((p.left, -1), (p.right, 1)):
        w = EYE_W * s * shape.width * p.scale
        full_h = EYE_H * s * shape.height * p.scale
        h = max(full_h * openness, 9 * s)  # closed eyes stay a thin bar
        cx = width / 2 + side * EYE_SPACING * s / 2 + gx
        cy = height * EYE_Y + (shape.dy + frame.bob) * EYE_H * s + gy
        if openness < 1:
            cy += (full_h - h) * 0.18  # closing eyes settle slightly downward
        boxes.append(EyeBox(cx, cy, w, h, side))
    return boxes[0], boxes[1]


def upper_lid_edge(box: EyeBox, shape: EyeShape, x: float) -> float:
    """y of the upper lid's lower edge at screen ``x`` (lid covers everything above)."""
    top = box.cy - box.h / 2
    y_mid = top + shape.upper * box.h
    return y_mid + math.tan(math.radians(shape.upper_tilt)) * (x - box.cx) * box.outer


def lower_lid_edge(box: EyeBox, shape: EyeShape, x: float) -> float:
    """y of the lower lid's upper edge at screen ``x`` (lid covers everything below)."""
    bottom = box.cy + box.h / 2
    u = clamp((x - (box.cx - box.w / 2)) / box.w) * 2 - 1  # -1..1 across the eye
    bulge = shape.lower_curve * box.h * (1 - u * u)
    y_mid = bottom - shape.lower * box.h
    return y_mid - bulge - math.tan(math.radians(shape.lower_tilt)) * (x - box.cx) * box.outer


# --------------------------------------------------------------------------- face


class PygameFace(Face):
    """The animated face window. Construct anywhere; :meth:`run` on the main thread."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        *,
        settings: FaceSettings | None = None,
        animator: FaceAnimator | None = None,
    ) -> None:
        self.settings = settings or FaceSettings.from_config(cfg)
        self.animator = animator or FaceAnimator()
        self._stop = threading.Event()
        self._label = ""
        self._label_lock = threading.Lock()
        self._fonts: dict[int, Any] = {}

    # -- Face contract (thread-safe, store targets only) -------------------------------
    def set_state(self, state: State) -> None:
        self.animator.set_state(state)

    def set_emotion(self, emotion: Emotion) -> None:
        self.animator.set_emotion(emotion)

    def set_mouth(self, amplitude: float) -> None:
        self.animator.set_mouth(amplitude)

    def stop(self) -> None:
        self._stop.set()

    def set_label(self, text: str) -> None:
        """Small caption at the bottom of the screen (used by ``roco demo-face``)."""
        with self._label_lock:
            self._label = text

    # -- main loop ----------------------------------------------------------------------
    def run(self) -> None:
        import pygame

        pygame.init()
        try:
            screen = self._open_window(pygame)
            clock = pygame.time.Clock()
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        self._stop.set()
                    elif event.type == pygame.KEYDOWN:
                        self.on_key(event.key)
                    elif event.type == pygame.VIDEORESIZE and not self.settings.fullscreen:
                        screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                self.render(screen, self.animator.update())
                pygame.display.flip()
                clock.tick(self.settings.fps)
        finally:
            pygame.quit()

    def on_key(self, key: int) -> None:
        """Hook for key presses other than Esc/Q (the demo uses it)."""

    def _open_window(self, pygame: Any) -> Any:
        pygame.display.set_caption("ROCO")
        if self.settings.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        else:
            size = (self.settings.width, self.settings.height)
            screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        return screen

    # -- drawing ------------------------------------------------------------------------
    def render(self, surface: Any, frame: FaceFrame) -> None:
        """Draw one frame onto ``surface`` (any size; works with a plain Surface)."""
        import pygame

        st = self.settings
        width, height = surface.get_size()
        s = min(width / REF_W, height / REF_H)
        surface.fill(st.background)
        p = frame.params
        brightness = p.brightness
        if frame.sleeping:
            brightness *= 0.85 + 0.15 * math.sin(frame.time * 2 * math.pi / 4.5)
        color = mix(st.background, st.eye_color, brightness)

        boxes = eye_boxes(frame, width, height)
        for box, shape in zip(boxes, (p.left, p.right), strict=True):
            self._draw_eye(pygame, surface, box, shape, color)

        glow = p.glow * (0.65 + 0.35 * frame.glow_pulse)
        if glow > 0.02:
            for box in boxes:
                self._draw_glow(pygame, surface, box, glow, s)

        if p.mouth > 0.02:
            self._draw_mouth(pygame, surface, frame, color, width, height, s)
        if frame.sleeping and p.openness < 0.2:
            self._draw_zzz(pygame, surface, boxes[1], frame.time, color, s)
        with self._label_lock:
            label = self._label
        if label:
            self._draw_label(pygame, surface, label, width, height, s)

    def _draw_eye(
        self, pygame: Any, surface: Any, box: EyeBox, shape: EyeShape, color: Color
    ) -> None:
        k = SUPERSAMPLE
        pad = 2
        w, h = max(1, round(box.w)), max(1, round(box.h))
        surf = pygame.Surface(((w + 2 * pad) * k, (h + 2 * pad) * k))
        bg = self.settings.background
        surf.fill(bg)
        radius = min(0.24 * box.w, box.h / 2)
        pygame.draw.rect(
            surf,
            color,
            pygame.Rect(pad * k, pad * k, w * k, h * k),
            border_radius=round(radius * k),
        )
        # Eyelids: background-coloured polygons in eye-local, supersampled coordinates.
        ox, oy = box.cx - w / 2 - pad, box.cy - h / 2 - pad  # screen pos of surf's top-left
        x0, x1 = box.cx - w / 2 - pad, box.cx + w / 2 + pad

        def local(x: float, y: float) -> tuple[float, float]:
            return ((x - ox) * k, (y - oy) * k)

        far_top, far_bottom = oy - 10, oy + h + 2 * pad + 10
        if shape.upper > 0.001 or abs(shape.upper_tilt) > 0.1:
            pts = [local(x0, far_top), local(x1, far_top)]
            pts += [
                local(x1, upper_lid_edge(box, shape, x1)),
                local(x0, upper_lid_edge(box, shape, x0)),
            ]
            pygame.draw.polygon(surf, bg, pts)
        if shape.lower > 0.001 or shape.lower_curve > 0.001 or abs(shape.lower_tilt) > 0.1:
            n = 24
            edge = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
            pts = [local(x, lower_lid_edge(box, shape, x)) for x in edge]
            pts += [local(x1, far_bottom), local(x0, far_bottom)]
            pygame.draw.polygon(surf, bg, pts)
        small = pygame.transform.smoothscale(surf, (w + 2 * pad, h + 2 * pad))
        surface.blit(small, (round(ox), round(oy)))

    def _draw_glow(self, pygame: Any, surface: Any, box: EyeBox, amount: float, s: float) -> None:
        spread = 26 * s
        w, h = box.w + 2 * spread, box.h + 2 * spread
        shrink = 8
        sw, sh = max(1, round(w / shrink)), max(1, round(h / shrink))
        small = pygame.Surface((sw, sh))
        small.fill((0, 0, 0))
        tint = mix((0, 0, 0), self.settings.eye_color, 0.55 * amount)
        inner = pygame.Rect(0, 0, max(1, round(box.w / shrink)), max(1, round(box.h / shrink)))
        inner.center = (sw // 2, sh // 2)
        pygame.draw.rect(small, tint, inner.inflate(2, 2), border_radius=max(1, sw // 5))
        blurred = pygame.transform.smoothscale(small, (max(1, sw // 2), max(1, sh // 2)))
        blurred = pygame.transform.smoothscale(blurred, (round(w), round(h)))
        surface.blit(
            blurred, (round(box.cx - w / 2), round(box.cy - h / 2)), special_flags=pygame.BLEND_ADD
        )

    def _draw_mouth(
        self,
        pygame: Any,
        surface: Any,
        frame: FaceFrame,
        color: Color,
        width: int,
        height: int,
        s: float,
    ) -> None:
        p = frame.params
        amp = frame.mouth_open
        mw = (46 + 30 * amp) * s
        mh = (7 + 40 * amp) * s
        gx = clamp(p.gaze_x + frame.offset_x, -1.5, 1.5) * GAZE_PX * s * 0.4
        cx = width / 2 + gx
        cy = height * EYE_Y + (MOUTH_Y + frame.bob * EYE_H) * s
        rect = pygame.Rect(0, 0, round(mw), round(mh))
        rect.center = (round(cx), round(cy))
        mouth_color = mix(self.settings.background, color, p.mouth)
        pygame.draw.rect(surface, mouth_color, rect, border_radius=round(min(mw, mh) / 2))

    def _draw_zzz(
        self, pygame: Any, surface: Any, box: EyeBox, t: float, color: Color, s: float
    ) -> None:
        period = 3.0
        for i in range(3):
            phase = ((t / period) + i / 3) % 1.0
            size = (10 + 10 * phase) * s
            x = box.cx + box.w * 0.55 + 40 * s * phase
            y = box.cy - 40 * s - 110 * s * phase
            fade = math.sin(phase * math.pi)
            c = mix(self.settings.background, color, 0.8 * fade)
            pts = [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]
            pygame.draw.lines(surface, c, False, pts, max(2, round(3 * s)))

    def _font(self, pygame: Any, size: int) -> Any:
        if size not in self._fonts:
            if not pygame.font.get_init():
                pygame.font.init()
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _draw_label(
        self, pygame: Any, surface: Any, label: str, width: int, height: int, s: float
    ) -> None:
        try:
            font = self._font(pygame, max(12, round(26 * s)))
        except Exception:  # pragma: no cover - font module unavailable
            return
        text = font.render(label, True, mix(self.settings.background, (230, 240, 245), 0.75))
        surface.blit(text, (round(16 * s), height - text.get_height() - round(12 * s)))
