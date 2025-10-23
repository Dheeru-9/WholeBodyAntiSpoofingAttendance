#!/usr/bin/env python3
"""
whole_body_attendance6.py
Whole-body attendance: face recognition + blink liveness + full-body (limb) verification.
Stores attendance in CSV and optionally Firebase Firestore.

Requirements:
    pip install opencv-python mediapipe face_recognition pandas numpy firebase-admin
Run:
    python whole_body_attendance6.py
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import face_recognition
from collections import deque, defaultdict
from datetime import datetime, timezone
import mediapipe as mp

# Optional Firebase (set SERVICE_ACCOUNT below to enable)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

# -------------------------
# CONFIG
# -------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20

BUFFER_SECONDS = 3
BUFFER_FRAMES = int(BUFFER_SECONDS * FPS)

ATTENDANCE_DIR = "attendance_audit"
ATTENDANCE_CSV = os.path.join(ATTENDANCE_DIR, "attendance_log.csv")

# Path to your known faces folder (you provided earlier)
KNOWN_FACES_DIR = r"C:\Users\adapa\Desktop\Internship2-Antforge\Complete_FaceRecognition\Kaggle-Faces1\Dataset\Faces"

# Firebase service account JSON (set to None to disable Firebase upload)
# e.g. r"C:\path\to\serviceAccountKey.json"
FIREBASE_SERVICE_ACCOUNT = r"C:\Users\adapa\Desktop\Internship2-Antforge\Complete_FaceRecognition\serviceAccountKey.json"  # <-- set path string to enable

# Thresholds
FACE_MATCH_TOLERANCE = 0.5
BLINK_EAR_THRESHOLD = 0.20   # empirical; lower -> more strict
BLINK_CONSEC_FRAMES = 2      # consecutive frames EAR below threshold to count as blink
POSE_VISIBILITY_THRESHOLD = 0.5  # MediaPipe landmark visibility threshold
MIN_POSE_KEYPOINTS = 6       # require at least this many important joints to be visible
ATTENDANCE_COOLDOWN = 10.0   # seconds before same track can be re-marked

# -------------------------
# Helpers: centroid tracker (simple)
# -------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=120):
        self.next_id = 0
        self.objects = {}     # id -> centroid
        self.bboxes = {}      # id -> bbox (x1,y1,x2,y2)
        self.disappeared = {} # id -> frames missed
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.bboxes[oid]
            del self.disappeared[oid]

    def update(self, rects):
        # rects: list of (x1,y1,x2,y2)
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.bboxes

        input_centroids = []
        for (x1,y1,x2,y2) in rects:
            cX = int((x1 + x2)/2.0)
            cY = int((y1 + y2)/2.0)
            input_centroids.append((cX,cY))

        if len(self.objects) == 0:
            for i,c in enumerate(input_centroids):
                self.register(c, rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:,None,:] - np.array(input_centroids)[None,:,:], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = object_ids[row]
                self.objects[oid] = input_centroids[col]
                self.bboxes[oid] = rects[col]
                self.disappeared[oid] = 0
                used_rows.add(row); used_cols.add(col)

            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.register(input_centroids[i], rects[i])

            for i in range(len(object_ids)):
                if i not in used_rows:
                    oid = object_ids[i]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)
        return self.bboxes

# -------------------------
# MediaPipe helpers
# -------------------------
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh eye landmark indices (MediaPipe): we'll use typical indices for EAR
# Left eye: [33, 160, 158, 133, 153, 144], Right eye: [362, 385, 387, 263, 373, 380]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def euclidean(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def eye_aspect_ratio(landmarks, eye_idxs, img_w, img_h):
    # landmarks: list of normalized multi-point landmarks (x,y,z) objects with .x/.y
    # convert normalized to pixel coords for EAR distances
    pts = []
    for idx in eye_idxs:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    # indices mapping for EAR formula: use 6 points:
    # two horizontal: pts[0] & pts[3], vertical pairs: (1,5) and (2,4)
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def count_visible_pose_keypoints(pose_landmarks):
    # We consider shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24), knees(25,26), ankles(27,28)
    required_idxs = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    visible = 0
    for r in required_idxs:
        lm = pose_landmarks[r.value]
        if hasattr(lm, 'visibility'):
            if lm.visibility is not None and lm.visibility >= POSE_VISIBILITY_THRESHOLD:
                visible += 1
    return visible, len(required_idxs)

# -------------------------
# Face database loader
# -------------------------
def load_known_faces(known_dir):
    encs = []
    names = []
    if not os.path.exists(known_dir):
        raise FileNotFoundError(f"Known faces folder not found: {known_dir}")
    # support two formats: subfolders per person OR all images flat
    entries = sorted(os.listdir(known_dir))
    for entry in entries:
        path = os.path.join(known_dir, entry)
        if os.path.isdir(path):
            label = entry
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            for f in files:
                p = os.path.join(path, f)
                img = face_recognition.load_image_file(p)
                facelocs = face_recognition.face_locations(img, model='hog')
                if len(facelocs) == 0:
                    continue
                e = face_recognition.face_encodings(img, known_face_locations=facelocs)
                if len(e) > 0:
                    encs.append(e[0]); names.append(label)
        else:
            # flat images: use filename as label
            if entry.lower().endswith(('.jpg','.jpeg','.png')):
                label = os.path.splitext(entry)[0]
                img = face_recognition.load_image_file(path)
                facelocs = face_recognition.face_locations(img, model='hog')
                if len(facelocs) == 0:
                    continue
                e = face_recognition.face_encodings(img, known_face_locations=facelocs)
                if len(e) > 0:
                    encs.append(e[0]); names.append(label)
    print(f"[INFO] Loaded {len(encs)} known face encodings.")
    return encs, names

# -------------------------
# Firebase helper (optional)
# -------------------------
def init_firebase(service_account_path):
    if not FIREBASE_AVAILABLE:
        print("[WARN] firebase-admin not installed; Firebase upload disabled.")
        return None
    if not service_account_path or not os.path.exists(service_account_path):
        print("[WARN] Firebase service account JSON not provided / not found; Firebase disabled.")
        return None
    try:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[INFO] Firebase initialized.")
        return db
    except Exception as e:
        print("[WARN] Firebase init failed:", e)
        return None

# -------------------------
# Main application class
# -------------------------
class WholeBodyAttendance:
    def __init__(self, source=0):
        # video
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        # trackers / buffers
        self.detector_hog = cv2.HOGDescriptor()
        self.detector_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=120)
        self.buffers = defaultdict(lambda: {"frames": deque(maxlen=BUFFER_FRAMES), "times": deque(maxlen=BUFFER_FRAMES)})

        # mediapipe processors
        self.pose_proc = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                     enable_segmentation=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                               min_detection_confidence=0.4, min_tracking_confidence=0.4)

        # face DB
        self.known_encs, self.known_names = load_known_faces(KNOWN_FACES_DIR)

        # attendance storage
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        if not os.path.exists(ATTENDANCE_CSV):
            pd.DataFrame(columns=["track_id","name","timestamp","notes"]).to_csv(ATTENDANCE_CSV, index=False)

        # firebase
        self.db = init_firebase(FIREBASE_SERVICE_ACCOUNT)

        # state
        self.last_marked = {}  # track_id -> timestamp
        self.marked_names = set()  # to avoid duplicate marking per session
        self.blink_counters = defaultdict(int)
        self.blinked = defaultdict(bool)

    def detect_people_hog(self, frame):
        rects, _ = self.detector_hog.detectMultiScale(frame, winStride=(8,8))
        boxes = [(int(x), int(y), int(x+w), int(y+h)) for (x,y,w,h) in rects]
        return boxes

    def find_face_in_bbox(self, frame, bbox):
        # perform face detection inside bbox using face_recognition
        x1,y1,x2,y2 = bbox
        x1 = max(0,x1); y1 = max(0,y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # face_recognition returns (top,right,bottom,left) in crop coords
        locs = face_recognition.face_locations(rgb, model='hog')
        if not locs:
            return None, None
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
        if not encs:
            return None, None
        # convert loc to frame coords: (top+ y1, right + x1, bottom+y1, left+x1)
        top, right, bottom, left = locs[0]
        face_box = (left + x1, top + y1, right + x1, bottom + y1)  # x1,y1,x2,y2
        return encs[0], face_box

    def pose_full_body_check(self, pose_landmarks):
        if pose_landmarks is None:
            return False
        visible, total = count_visible_pose_keypoints(pose_landmarks)
        # require at least MIN_POSE_KEYPOINTS visible
        return visible >= MIN_POSE_KEYPOINTS

    def mark_attendance(self, track_id, name, notes):
        # only non-Unknown and only if not already marked this session
        if name is None or name == "Unknown":
            return False
        now = datetime.now(timezone.utc).astimezone().isoformat()
        # cooldown per track
        last = self.last_marked.get(track_id, 0.0)
        if time.time() - last < ATTENDANCE_COOLDOWN:
            return False
        # if name already marked in session, skip
        if name in self.marked_names:
            # still update last_marked for cooldown per track
            self.last_marked[track_id] = time.time()
            return False
        row = {"track_id": track_id, "name": name, "timestamp": now, "notes": notes}
        df = pd.read_csv(ATTENDANCE_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)
        self.marked_names.add(name)
        self.last_marked[track_id] = time.time()
        print(f"[ATTEND] {track_id} | {name} | {now} | {notes}")

        # Firebase write (optional)
        if self.db is not None:
            try:
                doc = {
                    "track_id": int(track_id),
                    "name": name,
                    "timestamp": now,
                    "notes": notes
                }
                self.db.collection("attendance").add(doc)
            except Exception as e:
                print("[WARN] Firebase upload failed:", e)
        return True

    def run_once(self):
        ret, frame = self.cap.read()
        if not ret:
            return False
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # detect persons (HOG fallback)
        boxes = self.detect_people_hog(frame)
        tracked = self.tracker.update(boxes)

        # run mediapipe pose & face mesh on whole frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose_proc.process(rgb)
        face_res = self.face_mesh.process(rgb)
        pose_landmarks = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
        face_landmarks = face_res.multi_face_landmarks[0].landmark if face_res.multi_face_landmarks else None

        # compute EAR if face landmarks exist
        ear = 0.0
        if face_landmarks:
            ear_left = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
            ear_right = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
            ear = (ear_left + ear_right) / 2.0

        # track blink counters (we check global blink for simplicity)
        blinked_now = False
        if ear > 0 and ear < BLINK_EAR_THRESHOLD:
            # increment counter
            for tid in tracked.keys():
                self.blink_counters[tid] += 1
                if self.blink_counters[tid] >= BLINK_CONSEC_FRAMES:
                    self.blinked[tid] = True
                    blinked_now = True
        else:
            # reset counters for all tracked if EAR above threshold
            for tid in tracked.keys():
                self.blink_counters[tid] = 0

        # evaluate each tracked person
        vis = frame.copy()
        for tid, bbox in tracked.items():
            x1,y1,x2,y2 = bbox
            # draw bbox
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 2)
            label = f"ID:{tid}"
            # 1) find face encoding inside bbox
            enc, face_box = self.find_face_in_bbox(frame, bbox)
            name = "Unknown"
            face_match_ok = False
            if enc is not None and len(self.known_encs) > 0:
                dists = face_recognition.face_distance(self.known_encs, enc)
                best_idx = np.argmin(dists)
                if dists[best_idx] <= FACE_MATCH_TOLERANCE:
                    name = self.known_names[best_idx]
                    face_match_ok = True

            # 2) check blink (per-track)
            blink_ok = bool(self.blinked.get(tid, False))

            # 3) check full body via pose
            pose_ok = self.pose_full_body_check(pose_landmarks)

            # compose notes
            notes = []
            notes.append(f"face_match={face_match_ok}")
            notes.append(f"blink={blink_ok}")
            notes.append(f"full_body={pose_ok}")

            # decide marking: require face match, blink, and full body
            if face_match_ok and blink_ok and pose_ok:
                marked = self.mark_attendance(tid, name, notes)
                if marked:
                    label += f" {name} LIVE"
                else:
                    label += f" {name}"
            else:
                label += " ???"

            # draw face box if available
            if face_box:
                fx1, fy1, fx2, fy2 = face_box
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)

            cv2.putText(vis, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

        cv2.imshow("Whole-body Attendance (ESC to quit)", vis)
        key = cv2.waitKey(1) & 0xFF
        # ESC to quit
        if key == 27:
            return False
        return True

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

# -------------------------
# Entrypoint
# -------------------------
def main():
    print("Starting whole-body attendance (face+blink+full-body).")
    app = WholeBodyAttendance(source=0)
    try:
        while True:
            ok = app.run_once()
            if not ok:
                break
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        print("Stopped.")

if __name__ == "__main__":
    main()
