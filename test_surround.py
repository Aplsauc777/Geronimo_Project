"""
Geronimo - test surround.py with FAKE scans. No Isaac Sim, no GPU, no robot.

Builds synthetic lidar readings for situations you can reason about by hand,
runs them through analyze_surroundings + choose_heading, and draws the radar
for each. If the pictures and the printed decisions match your intuition, the
logic is right and anything that goes wrong later is the SENSOR, not this code.

    python test_surround.py

Writes surround_tests.png and prints a decision per scenario.
"""

import math

import cv2
import numpy as np

from surround import analyze_surroundings, choose_heading, draw_radar

MAX_RANGE = 6.0
N_RAYS = 72                      # matches horizontal_res=5.0 over 360 deg


def make_scan(wall_fn):
    """Build (angles, distances) for one scenario.

    wall_fn(bearing_deg) -> distance in metres, where bearing 0 = straight
    ahead and POSITIVE = counter-clockwise (to the robot's left).
    """
    bearings = np.linspace(-180.0, 180.0, N_RAYS, endpoint=False)
    dists = np.array([wall_fn(b) for b in bearings], dtype=float)
    return np.radians(bearings), dists


def near(bearing, centre, halfwidth):
    """True if `bearing` is within +/- halfwidth of `centre` (degrees, wraps)."""
    d = (bearing - centre + 180.0) % 360.0 - 180.0
    return abs(d) <= halfwidth


# ---------------------------------------------------------------------------
# SCENARIOS - each returns a distance for a given bearing.
# Predict the decision yourself before you look at the output.
# ---------------------------------------------------------------------------
SCENARIOS = {
    # Nothing anywhere. Expect: drive forward.
    "open field":
        lambda b: MAX_RANGE,

    # Wall straight ahead at 0.8 m, everything else open.
    # Expect: front blocked -> rotate toward whichever side has more room.
    "wall ahead":
        lambda b: 0.8 if near(b, 0, 35) else MAX_RANGE,

    # Corridor: walls left and right, open ahead and behind.
    # Expect: drive forward (sides don't matter if the front is clear).
    "corridor":
        lambda b: 0.9 if (near(b, 90, 45) or near(b, -90, 45)) else MAX_RANGE,

    # Blocked ahead and on the right; open to the left.
    # Expect: turn LEFT (+yaw).
    "debris right, open left":
        lambda b: 0.7 if (near(b, 0, 30) or near(b, -60, 60)) else MAX_RANGE,

    # Blocked ahead and on the left; open to the right.
    # Expect: turn RIGHT (-yaw).
    "debris left, open right":
        lambda b: 0.7 if (near(b, 0, 30) or near(b, 60, 60)) else MAX_RANGE,

    # Dead end - walls on three sides, only the way back is open.
    # Expect: rotate in place (forward speed 0).
    "dead end":
        lambda b: 0.8 if not near(b, 180, 50) else MAX_RANGE,

    # Something ahead but not close - tight, not blocked.
    # Expect: creep forward slowly.
    "narrow gap ahead":
        lambda b: 1.8 if near(b, 0, 20) else MAX_RANGE,
}


def main():
    tiles = []
    print()
    for name, fn in SCENARIOS.items():
        angles, dists = make_scan(fn)
        state = analyze_surroundings(angles, dists, max_range=MAX_RANGE)
        fwd, yaw = choose_heading(state)

        turn = "straight" if abs(yaw) < 1e-6 else ("LEFT" if yaw > 0 else "RIGHT")
        print(f"{name:<26} fwd={fwd:.2f}  yaw={yaw:+.2f} ({turn})")
        print(f"{'':<26} {state.summary()}")

        # Render this scenario's radar with its name on top.
        img = draw_radar(state, size=260)
        cv2.putText(img, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        cv2.putText(img, f"fwd={fwd:.2f} yaw={yaw:+.2f}", (8, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(img)

    # --- lay the tiles out in a grid ---------------------------------------
    per_row = 4
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        while len(row) < per_row:                      # pad the last row
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    cv2.imwrite("surround_tests.png", sheet)
    print("\nwrote surround_tests.png")

    # Comment this out if you're running headless.
    cv2.imshow("surround tests", sheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()