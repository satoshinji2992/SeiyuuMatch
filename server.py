import os
import sys
import glob
import socket
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from email.parser import BytesParser
from email.policy import default as email_policy
import json
import urllib.parse
import time
import threading
import uuid
import re

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
FACES_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "faces_upload"
)
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
HISTORY_FILE = os.path.join(UPLOADS_DIR, "history.json")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FACES_UPLOAD_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

history_lock = threading.Lock()
recognition_lock = threading.Lock()
recognition_slots = threading.BoundedSemaphore(1)
MAX_IMAGE_DIM = 1280
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_FACE_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_FACE_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024
MAX_FEEDBACK_BYTES = 16 * 1024
DEFAULT_MTCNN_THRESHOLDS = [0.4, 0.5, 0.7]
LOW_MTCNN_THRESHOLDS = [0.2, 0.3, 0.5]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
EXTRA_GROUP_MEMBERS = {
    "sumimi": ["佐々木李子"],
}
EXTRA_PERSON_BANDS = {
    "佐々木李子": ["sumimi"],
}


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def safe_path_segment(value, fallback):
    value = (value or "").strip()
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:80] or fallback


def safe_filename(filename):
    name = os.path.basename(filename or "")
    stem, ext = os.path.splitext(name)
    stem = safe_path_segment(stem, "photo")
    ext = ext.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = ".jpg"
    return f"{stem}{ext}"


def directory_size(path):
    total = 0
    if not os.path.isdir(path):
        return total
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def load_face_groups():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
    groups = {}
    counts = {}
    if not os.path.isdir(base_dir):
        return groups, counts
    for band in sorted(os.listdir(base_dir)):
        band_path = os.path.join(base_dir, band)
        if not os.path.isdir(band_path) or band.startswith("."):
            continue
        roles = []
        counts[band] = {}
        for role in sorted(os.listdir(band_path)):
            role_path = os.path.join(band_path, role)
            if os.path.isdir(role_path) and not role.startswith("."):
                roles.append(role)
                counts[band][role] = sum(
                    1
                    for filename in os.listdir(role_path)
                    if os.path.splitext(filename)[1].lower()
                    in ALLOWED_IMAGE_EXTENSIONS
                )
        if roles:
            groups[band] = roles
        else:
            counts.pop(band, None)

    for band, people in EXTRA_GROUP_MEMBERS.items():
        groups.setdefault(band, [])
        counts.setdefault(band, {})
        for name in people:
            if name not in groups[band]:
                groups[band].append(name)
            if name not in counts[band]:
                count = 0
                for source_band, source_people in groups.items():
                    if name not in source_people:
                        continue
                    source_dir = os.path.join(base_dir, source_band, name)
                    if os.path.isdir(source_dir):
                        count = sum(
                            1
                            for filename in os.listdir(source_dir)
                            if os.path.splitext(filename)[1].lower()
                            in ALLOWED_IMAGE_EXTENSIONS
                        )
                        break
                counts[band][name] = count
        groups[band] = sorted(groups[band])
    return groups, counts


def parse_person_bands(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw = value
    else:
        raw = str(value).split(",")
    return {str(b).strip() for b in raw if str(b).strip()}


def encode_bands(bands):
    return ",".join(sorted(dict.fromkeys(b for b in bands if b)))


def infer_name_bands(names):
    groups, _ = load_face_groups()
    by_name = {}
    for band, people in groups.items():
        for name in people:
            by_name.setdefault(name, set()).add(band)
    return [encode_bands(by_name.get(name, set())) for name in names]


def normalize_feature_bands(names, bands):
    normalized = []
    for name, band_value in zip(names, bands):
        band_set = parse_person_bands(band_value)
        band_set.update(EXTRA_PERSON_BANDS.get(name, []))
        normalized.append(encode_bands(band_set))
    return normalized


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdaFace"))
import net as adaface_net
from face_alignment.mtcnn import MTCNN

FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
)


def load_adaface():
    model = adaface_net.build_model("ir_50")
    statedict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    model.load_state_dict(
        {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    )
    model.eval()
    return model


def load_mtcnn():
    m = MTCNN(device="cpu", crop_size=(112, 112))
    m.min_face_size = 12
    m.thresholds = DEFAULT_MTCNN_THRESHOLDS
    return m


def adaface_infer(model, face_aligned):
    bgr = ((face_aligned[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(np.array([bgr.transpose(2, 0, 1)])).float()
    with torch.no_grad():
        feature, _ = model(tensor)
    return feature[0].numpy()


def resize_for_recognition(img):
    height, width = img.shape[:2]
    longest = max(width, height)
    if longest <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / longest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def recognize(
    mtcnn,
    adaface,
    names,
    bands,
    feature_db,
    feature_norms,
    image_bytes,
    thresholds,
    selected_bands,
):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_raw is None:
        return []
    img_raw = resize_for_recognition(img_raw)
    pil_img = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    original_thresholds = list(mtcnn.thresholds)
    mtcnn.thresholds = thresholds
    try:
        boxes, aligned_faces = mtcnn.align_multi(pil_img)
    finally:
        mtcnn.thresholds = original_thresholds

    results = []
    person_band_sets = [parse_person_bands(band) for band in bands]
    band_mask = np.array(
        [bool(person_bands & selected_bands) for person_bands in person_band_sets],
        dtype=bool,
    )
    if not np.any(band_mask):
        band_mask = np.ones(len(names), dtype=bool)
    filtered_names = [name for name, keep in zip(names, band_mask) if keep]
    filtered_band_sets = [
        person_bands for person_bands, keep in zip(person_band_sets, band_mask) if keep
    ]
    filtered_features = feature_db[band_mask]
    filtered_norms = feature_norms[band_mask]
    img_w, img_h = pil_img.size
    has_boxes = isinstance(boxes, np.ndarray) and len(boxes) == len(aligned_faces)
    for i, face_pil in enumerate(aligned_faces):
        vec = adaface_infer(adaface, np.array(face_pil))
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            continue
        cos_results = filtered_features @ vec / (filtered_norms * vec_norm)
        max_idx = int(np.argmax(cos_results))
        top_indices = np.argsort(cos_results)[::-1][:5]
        top5 = [
            {
                "name": filtered_names[idx],
                "band": encode_bands(filtered_band_sets[idx]),
                "bands": sorted(filtered_band_sets[idx]),
                "similarity": round(float(cos_results[idx]), 4),
            }
            for idx in top_indices
        ]
        box = boxes[i] if has_boxes else None
        bbox = None
        if box is not None:
            bbox = [
                float(box[0]) / img_w,
                float(box[1]) / img_h,
                float(box[2]) / img_w,
                float(box[3]) / img_h,
            ]
        results.append(
            {
                "name": filtered_names[max_idx],
                "band": encode_bands(filtered_band_sets[max_idx]),
                "bands": sorted(filtered_band_sets[max_idx]),
                "similarity": round(float(cos_results[max_idx]), 4),
                "top5": top5,
                "bbox": bbox,
            }
        )
    return results


class FaceHandler(BaseHTTPRequestHandler):
    mtcnn = None
    adaface = None
    names = None
    bands = None
    feature_db = None
    feature_norms = None

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_face_upload(self, content_length):
        if content_length <= 0:
            self.send_json(400, {"error": "empty upload"})
            return
        if content_length > MAX_FACE_UPLOAD_BYTES:
            self.send_json(413, {"error": "upload too large"})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json(400, {"error": "multipart/form-data required"})
            return

        body = self.rfile.read(content_length)
        if len(body) > MAX_FACE_UPLOAD_BYTES:
            self.send_json(413, {"error": "upload too large"})
            return
        message = BytesParser(policy=email_policy).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {content_length}\r\n"
                "\r\n"
            ).encode("utf-8")
            + body
        )

        band_value = ""
        role_value = ""
        photos = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_param("filename", header="content-disposition")
            payload = part.get_payload(decode=True) or b""
            if name == "band":
                band_value = payload.decode(part.get_content_charset() or "utf-8", "replace")
            elif name == "role":
                role_value = payload.decode(part.get_content_charset() or "utf-8", "replace")
            elif name == "photos" and filename:
                photos.append((filename, payload))

        band = safe_path_segment(band_value, "unknown_band")
        role = safe_path_segment(role_value, "unknown_role")
        incoming_size = sum(len(payload) for _, payload in photos)
        used_size = directory_size(FACES_UPLOAD_DIR)
        if used_size + incoming_size > MAX_FACE_UPLOAD_TOTAL_BYTES:
            self.send_json(
                413,
                {
                    "error": "faces_upload storage limit exceeded",
                    "limit_mb": MAX_FACE_UPLOAD_TOTAL_BYTES // (1024 * 1024),
                    "used_mb": round(used_size / 1024 / 1024, 2),
                },
            )
            return

        saved = []
        target_dir = os.path.join(FACES_UPLOAD_DIR, band, role)
        os.makedirs(target_dir, exist_ok=True)
        for filename, payload in photos:
            if not payload:
                continue
            used_size += len(payload)
            if used_size > MAX_FACE_UPLOAD_TOTAL_BYTES:
                self.send_json(413, {"error": "faces_upload storage limit exceeded"})
                return
            original_name = safe_filename(filename)
            _, ext = os.path.splitext(original_name)
            file_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            output_name = f"{file_id}-{original_name}"
            output_path = os.path.join(target_dir, output_name)
            with open(output_path, "wb") as f:
                f.write(payload)
            saved.append(
                {
                    "filename": output_name,
                    "path": os.path.relpath(output_path, os.path.dirname(os.path.abspath(__file__))),
                    "ext": ext,
                }
            )

        if not saved:
            self.send_json(400, {"error": "no photos uploaded"})
            return

        print(
            f"[server] face upload band={band} role={role} files={len(saved)} bytes={content_length}",
            flush=True,
        )
        self.send_json(200, {"saved": saved, "band": band, "role": role})

    def handle_feedback_upload(self, content_length):
        if content_length <= 0:
            self.send_json(400, {"error": "empty feedback"})
            return
        if content_length > MAX_FEEDBACK_BYTES:
            self.send_json(413, {"error": "feedback too large"})
            return

        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_json(400, {"error": "invalid json"})
            return

        message = str(data.get("message", "")).strip()
        contact = str(data.get("contact", "")).strip()
        if not message:
            self.send_json(400, {"error": "feedback message required"})
            return
        if len(message) > 2000:
            self.send_json(413, {"error": "feedback message too long"})
            return

        item = {
            "message": message,
            "contact": contact[:200],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ip": self.headers.get("CF-Connecting-IP") or self.client_address[0],
            "user_agent": self.headers.get("User-Agent", ""),
        }
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("[server] feedback saved", flush=True)
        self.send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "index.html"
            )
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
        elif self.path.startswith("/avatar/"):
            name = urllib.parse.unquote(self.path[len("/avatar/"):])
            base_dir = os.path.dirname(os.path.abspath(__file__))
            faces_dir = None
            for band_dir in glob.glob(os.path.join(base_dir, "faces", "*")):
                candidate = os.path.join(band_dir, name)
                if os.path.isdir(candidate):
                    faces_dir = candidate
                    break
            if os.path.isdir(faces_dir):
                for ext in [".jpg", ".jpeg", ".png"]:
                    photo = os.path.join(faces_dir, "1" + ext)
                    if os.path.exists(photo):
                        self.send_response(200)
                        ct = "image/jpeg" if ext != ".png" else "image/png"
                        self.send_header("Content-Type", ct)
                        self.end_headers()
                        with open(photo, "rb") as f:
                            self.wfile.write(f.read())
                        return
            self.send_response(404)
            self.end_headers()
        elif self.path == "/people":
            self.send_json(200, {"names": self.names})
        elif self.path == "/face_groups":
            groups, counts = load_face_groups()
            self.send_json(200, {"groups": groups, "counts": counts})
        elif self.path == "/health":
            self.send_json(200, {"ok": True, "people": len(self.names or [])})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        started = time.time()
        parsed_path = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        if parsed_path.path == "/upload_faces":
            self.handle_face_upload(content_length)
            return
        if parsed_path.path == "/feedback":
            self.handle_feedback_upload(content_length)
            return

        params = urllib.parse.parse_qs(parsed_path.query)
        relaxed = params.get("mode", [""])[0] == "relaxed"
        requested_bands = [
            safe_path_segment(band, "")
            for band in params.get("bands", [""])[0].split(",")
            if band.strip()
        ]
        selected_bands = set(requested_bands or ["mygo", "avemujica"])
        thresholds = LOW_MTCNN_THRESHOLDS if relaxed else DEFAULT_MTCNN_THRESHOLDS
        if content_length <= 0:
            self.send_json(400, {"error": "empty image"})
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_json(413, {"error": "image too large"})
            return
        if not recognition_slots.acquire(blocking=False):
            self.send_json(429, {"error": "recognition busy, please retry"})
            return

        body = self.rfile.read(content_length)
        try:
            with recognition_lock:
                result = recognize(
                    self.mtcnn,
                    self.adaface,
                    self.names,
                    self.bands,
                    self.feature_db,
                    self.feature_norms,
                    body,
                    thresholds,
                    selected_bands,
                )
            photo_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.jpg"
            photo_path = os.path.join(UPLOADS_DIR, photo_name)
            nparr = np.frombuffer(body, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = resize_for_recognition(img)
                cv2.imwrite(photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, 10])
                with history_lock:
                    history = load_history()
                    history.append({
                        "photo": f"uploads/{photo_name}",
                        "faces": result,
                        "mode": "relaxed" if relaxed else "default",
                        "bands": sorted(selected_bands),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    save_history(history)
            face_names = [r["name"] for r in result]
            self.send_json(
                200,
                {
                    "faces": face_names,
                    "details": result,
                    "mode": "relaxed" if relaxed else "default",
                    "thresholds": thresholds,
                    "bands": sorted(selected_bands),
                },
            )
            elapsed = time.time() - started
            print(
                f"[server] POST {content_length} bytes mode={'relaxed' if relaxed else 'default'} bands={','.join(sorted(selected_bands))} -> {len(result)} faces in {elapsed:.2f}s",
                flush=True,
            )
        except Exception as e:
            self.send_json(500, {"error": str(e)})
        finally:
            recognition_slots.release()

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3724)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("-f", "--features", default=FEATURES_FILE)
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    print(f"Loading features from {args.features}...")
    data = np.load(args.features, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    if "bands" in data:
        bands = normalize_feature_bands(names, [str(b) for b in data["bands"]])
    else:
        bands = infer_name_bands(names)
    feature_db = data["features"]
    feature_norms = np.linalg.norm(feature_db, axis=1)
    print(f"Loaded {len(names)} people, feature dim: {feature_db.shape[1]}")

    FaceHandler.mtcnn = mtcnn
    FaceHandler.adaface = adaface
    FaceHandler.names = names
    FaceHandler.bands = bands
    FaceHandler.feature_db = feature_db
    FaceHandler.feature_norms = feature_norms

    server = ThreadingHTTPServer((args.host, args.port), FaceHandler, False)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.server_bind()
    server.server_activate()
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
