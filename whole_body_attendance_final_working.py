#!/usr/bin/env python3
"""
AI-Powered Face and Body Recognition Attendance System with Anti-Spoofing and Firebase Integration
"""

import os
import cv2
import time
import math
import numpy as np
import pandas as pd
import face_recognition
import mediapipe as mp
from collections import deque, defaultdict
from datetime import datetime, timezone

# Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# TensorFlow for Anti-Spoof model
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ------------------------- CONFIG -------------------------
KNOWN_FACES_DIR = r"Face_DataSet_Location"
FIREBASE_SERVICE_ACCOUNT = r"servicejson_location"
ANTI_SPOOF_MODEL_PATH = None  # set to path if using model

FRAME_WIDTH, FRAME_HEIGHT, FPS = 640, 480, 20
ATTENDANCE_DIR = "attendance_audit"
ATTENDANCE_CSV = os.path.join(ATTENDANCE_DIR, "attendance_log.csv")
ATTENDANCE_XLSX = os.path.join(ATTENDANCE_DIR, "attendance_log.xlsx")
FACE_MATCH_TOLERANCE = 0.6
BLINK_EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
POSE_VISIBILITY_THRESHOLD = 0.5
MIN_POSE_KEYPOINTS = 6
MICRO_MOTION_THRESHOLD = 2.5
ATTENDANCE_COOLDOWN = 10.0
AS_MODEL_INPUT_SIZE = (160, 160)
# -----------------------------------------------------------


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=120):
        self.next_id = 0
        self.objects, self.bboxes, self.disappeared = {}, {}, {}
        self.max_disappeared, self.max_distance = max_disappeared, max_distance

    def register(self, centroid, bbox):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid], self.bboxes[oid], self.disappeared[oid] = centroid, bbox, 0
        return oid

    def deregister(self, oid):
        for d in [self.objects, self.bboxes, self.disappeared]:
            d.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.bboxes

        input_centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2) in rects]

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids): self.register(c, rects[i])
        else:
            object_ids, object_centroids = list(self.objects.keys()), list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:,None,:] - np.array(input_centroids)[None,:,:], axis=2)
            rows, cols = D.min(axis=1).argsort(), D.argmin(axis=1)[D.min(axis=1).argsort()]
            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols or D[r,c] > self.max_distance: continue
                oid = object_ids[r]
                self.objects[oid], self.bboxes[oid], self.disappeared[oid] = input_centroids[c], rects[c], 0
                used_rows.add(r); used_cols.add(c)
            for i, c in enumerate(input_centroids):
                if i not in used_cols: self.register(c, rects[i])
            for i, oid in enumerate(object_ids):
                if i not in used_rows:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared: self.deregister(oid)
        return self.bboxes


mp_pose, mp_face_mesh = mp.solutions.pose, mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def euclidean(a,b): return np.linalg.norm(np.array(a)-np.array(b))
def eye_aspect_ratio(landmarks, eye_idxs, w, h):
    pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in eye_idxs]
    A,B,C = euclidean(pts[1],pts[5]), euclidean(pts[2],pts[4]), euclidean(pts[0],pts[3])
    return 0 if C==0 else (A+B)/(2*C)


def count_visible_pose_keypoints(pose_landmarks):
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    visible = sum(1 for r in required if pose_landmarks[r.value].visibility >= POSE_VISIBILITY_THRESHOLD)
    return visible, len(required)


def load_known_faces(path):
    '''
    encs, names = [], []
    for folder in os.listdir(path):
        full = os.path.join(path, folder)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.lower().endswith((".jpg",".png")):
                    img = face_recognition.load_image_file(os.path.join(full, f))
                    locs = face_recognition.face_locations(img)
                    if not locs: continue
                    encs.append(face_recognition.face_encodings(img, locs)[0])
                    names.append(folder)
    print(f"[INFO] Loaded {len(encs)} known faces.")
    return encs, names
    '''

    encs, names = [], []
    print(f"[INFO] Scanning known faces in: {path}")

    if not os.path.exists(path):
        print(f"[ERROR] Directory not found: {path}")
        return encs, names

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"[INFO] Loading faces for: {folder}")
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, file)
                img = face_recognition.load_image_file(img_path)
                locs = face_recognition.face_locations(img)
                if len(locs) > 0:
                    enc = face_recognition.face_encodings(img, locs)[0]
                    encs.append(enc)
                    names.append(folder)
                else:
                    print(f"[WARN] No face found in {file}")
    print(f"[INFO] Loaded {len(encs)} known faces total.")
    return encs, names
    

def init_firebase(sa):
    if not FIREBASE_AVAILABLE or not os.path.exists(sa): return None
    cred = credentials.Certificate(sa)
    firebase_admin.initialize_app(cred)
    print("[INFO] Firebase initialized.")
    return firestore.client()


class AntiSpoof:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.centroid_history = defaultdict(lambda: deque(maxlen=8))
        if model_path and TF_AVAILABLE:
            self.model = load_model(model_path)
            print("[INFO] Anti-spoof model loaded.")

    def predict_from_crop(self, crop):
        if self.model is None: return None
        try:
            img = cv2.resize(crop, AS_MODEL_INPUT_SIZE).astype(np.float32)/255.0
            p = self.model.predict(np.expand_dims(img,0))
            return float(p.ravel()[0]) >= 0.5
        except: return None

    def record_centroid(self, tid, c): self.centroid_history[tid].append(c)
    def fallback_check_motion(self, tid):
        h = self.centroid_history[tid]
        if len(h)<3: return False
        arr = np.array(h)
        return np.linalg.norm(arr-np.median(arr,axis=0),axis=1).max() >= MICRO_MOTION_THRESHOLD


class WholeBodyAttendanceApp:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.pose_proc = mp_pose.Pose(
                          static_image_mode=False,
                          model_complexity=1,
                          smooth_landmarks=True,
                          min_detection_confidence=0.4,
                          min_tracking_confidence=0.4
                          )

        


        self.face_mesh = mp_face_mesh.FaceMesh(
                          static_image_mode=False,
                          max_num_faces=2,
                          refine_landmarks=True,
                          min_detection_confidence=0.4,
                          min_tracking_confidence=0.4
                          )








        
        self.detector_hog = cv2.HOGDescriptor()
        self.detector_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker = CentroidTracker()
        self.known_encs, self.known_names = load_known_faces(KNOWN_FACES_DIR)
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        if not os.path.exists(ATTENDANCE_CSV): pd.DataFrame(columns=["track_id","name","timestamp","notes","attendance"]).to_csv(ATTENDANCE_CSV,index=False)
        self.db = init_firebase(FIREBASE_SERVICE_ACCOUNT)
        self.antispoof = AntiSpoof(ANTI_SPOOF_MODEL_PATH)
        self.last_marked, self.marked_names = {}, set()
        self.blink_counters, self.blinked = defaultdict(int), defaultdict(bool)

    def detect_people_hog(self, frame):
        rects,_=self.detector_hog.detectMultiScale(frame,winStride=(8,8))
        return [(x,y,x+w,y+h) for x,y,w,h in rects]

    def find_face_and_encoding(self, frame, bbox):
        x1,y1,x2,y2=bbox
        crop=frame[y1:y2,x1:x2]
        if crop.size==0: return None,None,None
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        locs=face_recognition.face_locations(rgb)
        if not locs: return None,None,None
        enc=face_recognition.face_encodings(rgb,known_face_locations=locs)[0]
        t,r,b,l=locs[0]
        return enc,(l+x1,t+y1,r+x1,b+y1),crop

    def pose_full_body_check(self, pose_landmarks):
        if pose_landmarks is None: return False
        v,_=count_visible_pose_keypoints(pose_landmarks)
        return v>=MIN_POSE_KEYPOINTS

    def mark_attendance(self, track_id, name, notes):
        if name=="Unknown": return False
        now=datetime.now(timezone.utc).astimezone().isoformat()
        if time.time()-self.last_marked.get(track_id,0)<ATTENDANCE_COOLDOWN: return False
        if name in self.marked_names: return False

        attendance_status = "P" if "face_match=True" in ";".join(notes) else "A"
        row={"track_id":track_id,"name":name,"timestamp":now,"notes":";".join(notes),"attendance":attendance_status}

        # CSV & XLSX save
        df=pd.read_csv(ATTENDANCE_CSV)
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
        df.to_csv(ATTENDANCE_CSV,index=False)
        df.to_excel(ATTENDANCE_XLSX,index=False)

        # Firebase log
        if self.db:
            try:
                self.db.collection("attendance").add(row)
            except Exception as e:
                print("[WARN] Firebase upload failed:", e)

        self.marked_names.add(name)
        self.last_marked[track_id]=time.time()
        print(f"[ATTEND] {name} | Attendance={attendance_status}")
        return True

    def run_once(self):
        ret,frame=self.cap.read()
        if not ret: return False
        frame=cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pose_res=self.pose_proc.process(rgb)
        face_res=self.face_mesh.process(rgb)
        pose_landmarks=pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
        face_landmarks=face_res.multi_face_landmarks[0].landmark if face_res.multi_face_landmarks else None
        ear=0
        if face_landmarks:
            ear=(eye_aspect_ratio(face_landmarks,LEFT_EYE_IDX,frame.shape[1],frame.shape[0])
                 +eye_aspect_ratio(face_landmarks,RIGHT_EYE_IDX,frame.shape[1],frame.shape[0]))/2
        boxes=self.detect_people_hog(frame)
        tracked=self.tracker.update(boxes)

        if ear>0 and ear<BLINK_EAR_THRESHOLD:
            for t in tracked: 
                self.blink_counters[t]+=1
                if self.blink_counters[t]>=BLINK_CONSEC_FRAMES: self.blinked[t]=True
        else:
            for t in tracked: self.blink_counters[t]=0

        vis=frame.copy()
        for tid,bbox in tracked.items():
            x1,y1,x2,y2=bbox
            enc,face_box,face_crop=self.find_face_and_encoding(frame,bbox)
            matched="Unknown"; match=False
            if enc is not None and len(self.known_encs)>0:
                d=face_recognition.face_distance(self.known_encs,enc)
                i=np.argmin(d)
                if d[i]<=FACE_MATCH_TOLERANCE:
                    matched=self.known_names[i]; match=True
            spoof=False
            if face_crop is not None:
                cx,cy=int((face_box[0]+face_box[2])/2),int((face_box[1]+face_box[3])/2)
                self.antispoof.record_centroid(tid,(cx,cy))
                res=self.antispoof.predict_from_crop(face_crop)
                if res is True: spoof=True
                elif res is None:
                    blink_ok=self.blinked.get(tid,False)
                    motion_ok=self.antispoof.fallback_check_motion(tid)
                    spoof=blink_ok or motion_ok
            full=self.pose_full_body_check(pose_landmarks)
            notes=[f"face_match={match}",f"anti_spoof={spoof}",f"blinked={self.blinked.get(tid,False)}",f"full_body={full}"]
            if match and spoof and full:
                self.mark_attendance(tid,matched,notes)
                label=f"{matched} (P)"
            else: label=f"{matched}"
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(vis,label,(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imshow("AI Attendance System (ESC to exit)",vis)
        if cv2.waitKey(1)&0xFF==27: return False
        return True

    def close(self): self.cap.release(); cv2.destroyAllWindows()


def main():
    app=WholeBodyAttendanceApp(0)
    print("[INFO] Running system...")
    while True:
        if not app.run_once(): break
    app.close()
    print("[INFO] Stopped.")


if __name__=="__main__":
    main()

