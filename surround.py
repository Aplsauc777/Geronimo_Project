
from dataclasses import dataclass
from typing import List, Optional

import math

import cv2
import numpy as np

NUM_SECTORS = 12        
MAX_RANGE = 6.0        
BLOCK_DIST = 1.2       
CLEAR_DIST = 2.5       

@dataclass
class SurroundState:
    """What's around the robot right now, one entry per sector."""

    min_dist: List[float]        # nearest hit in each sector (metres)
    blocked: List[bool]          # True if that sector is impassable
    num_sectors: int
    max_range: float

    def front(self) -> float:
        return self.min_dist[0]

    def sector_angle_deg(self, i: int) -> float:
        """Centre bearing of sector i, in degrees, 0 = forward, +ve = left."""
        return (360.0 / self.num_sectors) * i

    def summary(self) -> str:
        names = {0: "front", 3: "left", 6: "rear", 9: "right"}
        parts = []
        for i, name in names.items():
            d = self.min_dist[i]
            parts.append(f"{name}={d:.1f}m{'!' if self.blocked[i] else ''}")
        return " | ".join(parts)


def analyze_surroundings(angles_rad, distances,
                         num_sectors: int = NUM_SECTORS,
                         max_range: float = MAX_RANGE,
                         block_dist: float = BLOCK_DIST) -> SurroundState:
    """Bin raw rays into sectors and flag which are blocked.

    angles_rad : array of ray bearings in the ROBOT frame.
                 0 = straight ahead, positive = counter-clockwise (left).
    distances  : matching array of hit distances in metres. Non-hits should be
                 inf or >= max_range.
    """
    angles = np.asarray(angles_rad, dtype=float)
    dists = np.asarray(distances, dtype=float)

    # Treat misses / NaNs as "nothing out there".
    dists = np.where(np.isfinite(dists), dists, max_range)
    dists = np.clip(dists, 0.0, max_range)

    # Normalise angles into [0, 2pi) so binning is simple.
    two_pi = 2.0 * math.pi
    a = np.mod(angles, two_pi)

    # Which sector each ray falls in.
    sector_width = two_pi / num_sectors
    idx = np.floor(np.mod(a + sector_width / 2, two_pi) / sector_width).astype(int) % num_sectors
    # Nearest hit per sector. Start at max_range so empty sectors read "open".
    min_dist = [max_range] * num_sectors
    for s in range(num_sectors):
        sel = dists[idx == s]
        if sel.size:
            min_dist[s] = float(sel.min())

    blocked = [d < block_dist for d in min_dist]
    return SurroundState(min_dist=min_dist, blocked=blocked,
                         num_sectors=num_sectors, max_range=max_range)


def choose_heading(state: SurroundState):
    """Pick a velocity command from the surround picture.

    Returns (forward_speed, yaw_rate). +yaw = turn left.
    Strategy: go straight if the front is open; otherwise rotate toward
    whichever nearby sector has the most room.
    """
    n = state.num_sectors
    front = state.min_dist[0]

    if front >= CLEAR_DIST:
        return 0.4, 0.0

    if not state.blocked[0]:
        return 0.15, 0.0

    span = n // 4                       # a quarter turn's worth of sectors
    left_best = max(state.min_dist[1:1 + span])
    right_best = max(state.min_dist[n - span:])

    if left_best >= right_best:
        return 0.0, +0.6                # rotate left in place
    return 0.0, -0.6                    # rotate right in place


def draw_radar(state: SurroundState, size: int = 320) -> np.ndarray:
    """Top-down 'Tesla display': robot at centre, sectors shaded by clearance.

    Returns a BGR image you can imshow, save, or paste into a bigger frame.
    Screen convention: UP on the image = FORWARD for the robot.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    radius = int(size * 0.42)

    for frac in (0.33, 0.66, 1.0):
        cv2.circle(img, (cx, cy), int(radius * frac), (45, 45, 45), 1)
        metres = state.max_range * frac
        cv2.putText(img, f"{metres:.0f}m", (cx + 4, cy - int(radius * frac) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)

    sector_deg = 360.0 / state.num_sectors

    for i in range(state.num_sectors):
        d = state.min_dist[i]
        is_blocked = state.blocked[i]

        # Wedge reaches as far as the nearest obstacle in that direction.
        frac = min(d / state.max_range, 1.0)
        r = max(int(radius * frac), 6)

        # Robot forward (+X) should point UP on screen, and positive angles
        # go counter-clockwise (left). OpenCV angles run clockwise from +X
        # right, so we negate and shift by -90 degrees.
        centre = state.sector_angle_deg(i)
        start = -(centre + sector_deg / 2) - 90.0
        end = -(centre - sector_deg / 2) - 90.0

        if is_blocked:
            colour = (60, 60, 220)          # red   - in the way
        elif d < CLEAR_DIST:
            colour = (60, 190, 230)         # yellow - tight
        else:
            colour = (90, 190, 90)          # green - open
        cv2.ellipse(img, (cx, cy), (r, r), 0.0, start, end, colour, -1)

    for i in range(state.num_sectors):
        ang = math.radians(-state.sector_angle_deg(i) - sector_deg / 2 - 90.0)
        x = int(cx + radius * math.cos(ang))
        y = int(cy + radius * math.sin(ang))
        cv2.line(img, (cx, cy), (x, y), (25, 25, 25), 1)

    cv2.circle(img, (cx, cy), 9, (240, 240, 240), -1)
    cv2.line(img, (cx, cy), (cx, cy - 22), (240, 240, 240), 2)   # nose = up
    cv2.putText(img, "FWD", (cx - 14, cy - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)

    return img


def overlay_radar(frame: np.ndarray, state: SurroundState,
                  size: int = 200, margin: int = 12) -> np.ndarray:
    """Paste the radar into the bottom-right of a camera frame, in place."""
    radar = draw_radar(state, size=size)
    h, w = frame.shape[:2]
    y0, x0 = h - size - margin, w - size - margin
    if y0 < 0 or x0 < 0:
        return frame                        # frame too small skip
    roi = frame[y0:y0 + size, x0:x0 + size]
    cv2.addWeighted(radar, 0.85, roi, 0.15, 0.0, dst=roi)
    return frame