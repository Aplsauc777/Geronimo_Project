import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import math
from ultralytics import YOLO


L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
TORSO_KPTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
KP_THRESH = 0.5
REAL_H = 1.82
FOCAL_PX = (287 * 1) / REAL_H



@dataclass
class RescueReport:
    detected: bool = False #bool the datatype is boolean
    confidence: Optional[float] = None #Optional means it can have "none" 
    tracking_active: bool = False
    track_id: Optional[int] = None

    posture: Optional[str] = None
    distance_m: Optional[float] = None
    distance_m2: Optional[float] = None

    left_clear: Optional[bool] = None
    right_clear: Optional[bool] = None
    scene_note: Optional[str] = None

    timestamp: str = field(
        default_factory = lambda: datetime.now().strftime("%H:%M:%S")
    )


    def render(self) -> str:
        if not self.detected:
            return "No person detected"
        
        conf_pct = f"{self.confidence * 100:.1f}%" if self.confidence is not None else "No person for confidence"
        posture = f"{self.posture.capitalize()}" if self.posture else "pending posture"
        dist = f"Between {self.distance_m:.1f}m and {self.distance_m2:.1f}m" if self.distance_m is not None and self.distance_m2 is not None else "pending distance"
        track = "Active" if self.tracking_active else "Inactive"

        return (
            "Person Detected\n"
            f"Confidence: {conf_pct}\n"
            f"Posture: {posture}\n"
            f"Est Distance From Robot: {dist}\n"
            f"Tracking Status: {track}\n"
            f"Time: {self.timestamp}"
        )   
    
def choose_target(id_to_box):
    if not id_to_box:
        return None
    
    return max(id_to_box, 
                key = lambda tid: id_to_box[tid][2] * id_to_box[tid][3]
    )


def estimate_posture(kpts_xy, kpts_conf, box, kp_thresh = 0.5):
    def pt(i):
        return kpts_xy[i] if kpts_conf[i] >= kp_thresh else None
    
    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)

    if all(p is not None for p in (ls, rs, lh, rh)):
        mid_sh = ((ls[0] + rs[0]) / 2), ((ls[1] + rs[1]) / 2)
        mid_hip = ((lh[0] + rh[0]) / 2), ((lh[1] + rh[1]) / 2)

        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])
        
        
        if(dy != None and dx != None):
            angle_from_dy = math.degrees(math.atan2(dx, dy))
            print(f"[DBG] posture dx={dx:.1f} dy={dy:.1f} angle={angle_from_dy:.1f}", flush=True)

            print(f"[DBG] torso angle={angle_from_dy:.1f}deg  "
                          f"kp_conf sh={kpts_conf[5]:.2f},{kpts_conf[6]:.2f} "
                          f"hip={kpts_conf[11]:.2f},{kpts_conf[12]:.2f}", flush=True)
            print(f"[DBG] ls={ls} rs={rs} lh={lh} rh={rh}", flush=True)
            print(f"[DBG] mid_sh={mid_sh} mid_hip={mid_hip}", flush=True)
        if angle_from_dy < 35:
            return "Standing"
        elif angle_from_dy < 55:
            return"Leaning"
        else:
            return "Lying"
        

    cx, cy, w, h = box
    return "Standing" if h >= w * 1.3 else "Lying"

def estimate_distance(box, posture, focal_px=FOCAL_PX):
    """Monocular distance from box size. Returns (low_m, high_m).
    Only gives up when there's no posture at all."""
    cx, cy, w, h = box
    if h < 10 or posture is None:
        return None

    p = posture.lower()
    if "stand" in p:
        span, margin = h, 0.20              # box height ~ full standing height
    elif "lying" in p or "lie" in p:
        span, margin = max(w, h), 0.35      # long side ~ body length
    else:                                    # leaning, kneeling, anything else
        span, margin = max(w, h), 0.35      # usable, just wider uncertainty

    d = (1.70 * focal_px) / span
    return (d * (1 - margin), d * (1 + margin))

def draw_torso_debug(frame, kpts_xy, kpts_conf):
    pts = {}
    for i in TORSO_KPTS:
        x, y = int(kpts_xy[i][0]), int(kpts_xy[i][1])
        good = kpts_conf[i] >= KP_THRESH
        cv2.circle(frame, (x, y), 5, (0, 255, 0) if good else (0, 0, 255), -1)
        cv2.putText(frame, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1)
        if good:
            pts[i] = (x, y)

    if len(pts) < 4:
        cv2.putText(frame , "Torso Incomplete", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness = 2)
        return

    mid_sh = ((pts[L_SHOULDER][0] + pts[R_SHOULDER][0]) // 2), ((pts[L_SHOULDER][1] + pts[R_SHOULDER][1]) // 2)
    mid_hip = ((pts[L_HIP][0]  + pts[R_HIP][0]) // 2), ((pts[L_HIP][1] + pts[R_HIP][1]) // 2)

    cv2.line(frame, pts[L_SHOULDER], pts[R_SHOULDER], (200, 200, 200), 1)
    cv2.line(frame, pts[L_HIP], pts[R_HIP], (200, 200, 200), 1)
    cv2.circle(frame, mid_sh, 6, (0, 165, 255), -1)
    cv2.circle(frame, mid_hip, 6, (0, 165, 255), -1)

    cv2.line(frame, mid_hip, mid_sh, (0, 200, 100), 3) #graphing works with 2 distinct "(x,y)" points
    corner = (mid_hip[0], mid_sh[1])
    cv2.line(frame, mid_hip, corner, (255, 120, 0), 1)
    cv2.line(frame, mid_sh, corner, (255, 120, 0), 1)

    dx = mid_sh[0] - mid_hip[0]
    dy = mid_sh[1] - mid_hip[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    cv2.putText(frame, f"dx = {dx} dy = {dy} angle = {angle:.1f}deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness = 2)

#TO LOOK
def analyze_clearance(depth, min_safe_m = 1.0, block_frac = 0.15):
    h, w = depth.shape

    band = depth[int(h * 0.4):int(h * 0.9), :]

    clear = []
    for third in np,array_split(band, 3, axis = 1):
        valid = np.isinfinite(third) & (third > 0)
        near = (third < min_safe_m) & valid
        blocked = near.sum() / max(valid.sum(), 1)
        clear.append(blocked < block_frac)

    return clear

def choose_heading(left_clear, centre_clear, right_clear):
    if centre_clear:
        return 0.4, 0
    if left_clear and not right_clear:
        return 0.1, + 0.6
    if right_clear and not left_clear:
        return 0.1, -0.6
    if left_clear and right_clear:
        return 0.1, +0.6
    return 0, +0.8


# def main():

#     model = YOLO("yolo11n-pose.pt")
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#     locked_id = None
#     last_print = 0.0

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
        
#         if not cap.isOpened():
#             print("Cannot open camera")
#             return
        
#         results = model.track(
#             frame, persist = True, classes=[0],
#             conf = 0.5, tracker = "bytetrack.yaml", verbose = False
#         )
        
#         boxes = results[0].boxes

#         id_to_box, id_to_conf = {}, {}
#         id_to_kpts_xy, id_to_kpts_conf = {}, {}
#         if boxes.id is not None and results[0].keypoints is not None:
#             xywh = boxes.xywh.cpu().numpy()
#             confs = boxes.conf.cpu().numpy()
#             ids = boxes.id.int().cpu().tolist()

#             kxy = results[0].keypoints.xy.cpu().numpy()
#             kcf = results[0].keypoints.conf.cpu().numpy()
#             for box, cf, tid, in zip(xywh, confs, ids):
#                 id_to_box[tid] = box #
#                 id_to_conf[tid] = float(cf)
            
#             for i, tid in enumerate(ids): #Enumerate returns both index and value of list (i being index)
#                 id_to_kpts_xy[tid] = kxy[i]
#                 id_to_kpts_conf[tid] = kcf[i]       
       
#         if locked_id not in id_to_box:
#             locked_id = choose_target(id_to_box)

#         if locked_id is not None:
#             report = RescueReport(
#                 detected = True,
#                 confidence = id_to_conf[locked_id],
#                 tracking_active = True,
#                 track_id = locked_id
#             )
#         else:
#             report = RescueReport(detected = False)

#         if locked_id is not None:
#             cx, cy, w, h = id_to_box[locked_id]
#             p1 = (int(cx - w / 2), int(cy - h / 2))
#             p2 = (int(cx + w / 2), int(cy + h / 2))
#             cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness = 2)
#             cv2.putText(frame, f"Person {locked_id}", (p1[0], p1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 0)
#             cv2.putText(frame, f"Height {h}", (p1[0], p1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 0)
#             report.posture = estimate_posture(
#                 id_to_kpts_xy[locked_id], id_to_kpts_conf[locked_id], id_to_box[locked_id]
#             )

#             dist_rng = estimate_distance(box, report.posture)
#             report.distance_m = dist_rng[0] if dist_rng else None
#             report.distance_m2 = dist_rng[1] if dist_rng else None


#         if(locked_id is not None and locked_id in id_to_kpts_xy):
#             draw_torso_debug(frame, id_to_kpts_xy[locked_id], id_to_kpts_conf[locked_id])
#         cv2.imshow("Geronimo's Report", frame)

#         if time.time() - last_print > 1.0:
#             print("\n" + report.render())
#             last_print = time.time()

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

