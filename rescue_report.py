import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import math

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
TORSO_KPTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
KP_THRESH, CONF_THRESH = 0.5
REAL_H = 1.82
FOCAL_PX = (287 * 1) / REAL_H

USE_AI_CAMERA = False 

if USE_AI_CAMERA:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from boxmot import ByteTrack
else:
    from ultralytics import YOLO

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
        dist = f"Between {self.distance_m:.1f}m and {self.distance_m2:.1f}m" if self.distance_m is not None else "pending distance"
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

def read_from_webcam(model, cap):
    ok, frame = cap.read()
    if not ok:
        return None, {}
    
    res = model.track(frame, persist=True, classes=[0], conf=0.4, tracker='bytetrack.yaml',verbose = False)[0]

    people = {}
    boxes = res.boxes
    if boxes is not None and boxes.id is not None:
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().tilist()
        kp = res.keypoints
        has_kp = kp is not None and kp.conf is not None
        kxy = kp.xy.cpu().numpy() if has_kp else None
        kcf = kp.conf.cpu().numpy if has_kp else None
        for i, tid in enumerate(ids):
            people[tid] = {
                "box": xywh[i], 
                "conf": float(confs[i]),
                "kxy": kxy[i] if has_kp else None,
                "kcf": kcf[i] if has_kp else None,
            }
        return frame, people

def read_from_sensor(imx500, picam2, tracker):
    req = picam2.capture_request()
    frame = req.make_array("main")
    meta = req.get_metadata()
    req.release()

    outputs = imx500.get_outputs(meta) #Raw detection output (not mode.track())
    dets = parse_outputs(outputs, meta, imx500, picam2)

    people = {}
    if len(dets):
        for t in tracker.update(dets, frame): #BoxMOT's ByteTrack
            x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
            people[tid] = {
                "box": np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]),
                "conf": float(t[5]) if len(t) > 5 else 1.0,
                "kxy": None, "kcf": None
            }
        return frame, people
    
def parse_outputs(outputs, meta, imx500, picam2):
    if outputs is None:
        return [], []
    boxes_raw, scores, kpts_raw = outputs[0], outputs[1], outputs[2]
    dets, kpts = [], []
    for b, s, k in zip (boxes_raw, scores, kpts_raw):
        if s < CONF_THRESH:
            continue
        x, y, w, h = imx500.convert_inference_coords(b, meta, picam2)
        dets.append([x ,y, x + w, y + h, float(s), 0])
        person_kpts = [(kx * w, ky * h, kc) for kx, ky, kc, in k]
        kpts.append(person_kpts)

    return dets, kpts


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
            angle_from_dy = math.degrees(math.atan2(dy, dx))
        if angle_from_dy < 35:
            return "Standing"
        elif angle_from_dy < 55:
            return"Lying"
        else:
            return "Leaning"
    
    cx, cy, w, h = box
    return "Standing" if h >= w else "Lying"

def estimate_distance(box, posture, focal_px = FOCAL_PX):
    cx, cy, w, h = box
    if h < 10:
        return None
    
    if posture == "Standing":
        real_h = REAL_H
    elif posture == "Lying":
        h = max(w, h)
        real_h = REAL_H
    else:
        return None

    d = (real_h * focal_px) / h # h/focal_px = real_h/d cross multiply to find d

    return (d * 0.8, d * 1.2)


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
    cv2.putText(frame, f"dx = {dx} dy = {dy} angle = {angle:.1f}deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness = 2)

def main():
    if USE_AI_CAMERA:
        imx500 = IMX500("network.rpk")
        picam2 = Picamera2(imx500.camera_num)
        picam2.start()
        tracker = ByteTrack
    else:
        model = YOLO("yolo11n-pose.pt")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    locked_id = None
    last_print = 0.0

    while True:

        if USE_AI_CAMERA:
            frame, people = read_from_sensor(imx500, picam2, tracker)
        else:
            frame, people = read_from_webcam(model, cap)
        if frame is None:
            break

        ok, frame = cap.read()
        if not ok:
            break
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        results = model.track(frame)
        
        boxes = results[0].boxes

        id_to_box, id_to_conf = {}, {}
        id_to_kpts_xy, id_to_kpts_conf = {}, {}
        if boxes.id is not None and results[0].keypoints is not None:
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()

            kxy = results[0].keypoints.xy.cpu().numpy()
            kcf = results[0].keypoints.conf.cpu().numpy()
            for box, cf, tid, in zip(xywh, confs, ids):
                id_to_box[tid] = box #
                id_to_conf[tid] = float(cf)
            
            for i, tid in enumerate(ids): #Enumerate returns both index and value of list (i being index)
                id_to_kpts_xy[tid] = kxy[i]
                id_to_kpts_conf[tid] = kcf[i]       
       
        if locked_id not in id_to_box:
            locked_id = choose_target(id_to_box)

        if locked_id is not None:
            report = RescueReport(
                detected = True,
                confidence = id_to_conf[locked_id],
                tracking_active = True,
                track_id = locked_id
            )
        else:
            report = RescueReport(detected = False)

        if locked_id is not None:
            cx, cy, w, h = id_to_box[locked_id]
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness = 2)
            cv2.putText(frame, f"Person {locked_id}", (p1[0], p1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 0)
            cv2.putText(frame, f"Height {h}", (p1[0], p1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 0)
            report.posture = estimate_posture(
                id_to_kpts_xy[locked_id], id_to_kpts_conf[locked_id], id_to_box[locked_id]
            )

            dist_rng = estimate_distance(box, report.posture)
            report.distance_m = dist_rng[0] if dist_rng else None
            report.distance_m2 = dist_rng[1] if dist_rng else None


        if(locked_id is not None and locked_id in id_to_kpts_xy):
            draw_torso_debug(frame, id_to_kpts_xy[locked_id], id_to_kpts_conf[locked_id])
        cv2.imshow("Geronimo's Report", frame)

        if time.time() - last_print > 1.0:
            print("\n" + report.render())
            last_print = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        

    if USE_AI_CAMERA:
        picam2.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

