import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
from ultralytics import YOLO

@dataclass
class RescueReport:
    detected: bool = False #bool the datatype is boolean
    confidence: Optional[float] = None #Optional means it can have "none" 
    tracking_active: bool = False
    track_id: Optional[int] = None

    posture: Optional[str] = None
    distance_m: Optional[float] = None

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
        dist = f"{self.distance_m}m" if self.distance_m is not None else "pending distance"
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

def main():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    locked_id = None
    last_print = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
    
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        results = model.track(
            frame, persist = True, classes = [0],
            conf = 0.4, tracker = 'bytetrack.yaml', verbose = False
        )
        boxes = results[0].boxes

        id_to_box, id_to_conf = {}, {}
        if boxes.id is not None:
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            for box, cf, tid, in zip(xywh, confs, ids):
                id_to_box[tid] = box
                id_to_conf[tid] = float(cf)

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
            cv2.putText(frame, f"Person {locked_id}", (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Geronimo's Report", frame)

        if time.time() - last_print > 1.0:
            print("\n" + report.render())
            last_print = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()