import os
import sys
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
import queue
import hashlib
import hmac
import secrets
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

_dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_dotenv_path):
    with open(_dotenv_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
AVATAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatar")
ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon")
FACES_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "faces_upload"
)
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
UPLOAD_PHOTOS_DIR = os.path.join(UPLOADS_DIR, "photos")
HISTORY_DIR = os.path.join(UPLOADS_DIR, "history")
HISTORY_LOCK_FILE = os.path.join(UPLOADS_DIR, ".history.lock")
JOBS_LOCK_FILE = os.path.join(JOBS_DIR, ".jobs.lock")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(UPLOAD_PHOTOS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(ICON_DIR, exist_ok=True)
os.makedirs(FACES_UPLOAD_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

history_lock = threading.Lock()
recognition_lock = threading.Lock()
recognition_slots = threading.BoundedSemaphore(1)
RECOGNITION_QUEUE_TIMEOUT = float(os.environ.get("RECOGNITION_QUEUE_TIMEOUT", "30"))
JOB_QUEUE_MAX_SIZE = int(os.environ.get("JOB_QUEUE_MAX_SIZE", "200"))
MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", "1280"))
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_FACE_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_FACE_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024
MAX_FEEDBACK_BYTES = 16 * 1024
DEFAULT_MTCNN_THRESHOLDS = [0.3, 0.7, 0.7]
LOW_MTCNN_THRESHOLDS = [0.2, 0.3, 0.5]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
STATIC_CACHE_SECONDS = 7 * 24 * 60 * 60
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
ADMIN_TOKEN_TTL = 24 * 60 * 60
job_queue = queue.Queue(maxsize=JOB_QUEUE_MAX_SIZE)
EASTER_EGG_BAND = "???"
EASTER_EGG_NAME = "liyuu"
EASTER_EGG_DISPLAY_NAME = "Liyuu"
EASTER_EGG_TRIGGER_SCORE = 65
EASTER_EGG_MESSAGE = "你引起了李嘉的注意"
BIG_BROTHER_NAME = "立希"
BIG_BROTHER_TRIGGER_SCORE = 65
BIG_BROTHER_MESSAGE = "老大哥正在看着你"
EXTRA_GROUP_MEMBERS = {
    "sumimi": ["佐々木李子"],
    "millsage": ["薬師寺李有", "千春", "結川あさき", "伊駒ゆりえ", "咲川ひなの"],
    "dumbrock": ["橘めい", "涼泉桜花", "花宮初奈", "菱川花菜", "遠野ひかる"],
    "mewtype": ["仲町あられ", "宮永ののか", "峰月律", "藤都子", "千石ユノ"],
}
EXTRA_PERSON_BANDS = {
    "佐々木李子": ["sumimi"],
}


@contextmanager
def file_lock(path):
    with open(path, "a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def daily_key(now=None):
    return time.strftime("%Y-%m-%d", now or time.localtime())


def daily_upload_dir(day):
    return os.path.join(UPLOAD_PHOTOS_DIR, safe_path_segment(day, "unknown_date"))


def daily_history_file(day):
    filename = f"{safe_path_segment(day, 'unknown_date')}.jsonl"
    return os.path.join(HISTORY_DIR, filename)


def append_history(item, day):
    with history_lock:
        with file_lock(HISTORY_LOCK_FILE):
            history_path = daily_history_file(day)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def job_path(job_id):
    return os.path.join(JOBS_DIR, f"{safe_path_segment(job_id, 'job')}.json")


def write_job_status(job_id, payload):
    payload = dict(payload)
    payload["job_id"] = job_id
    payload.setdefault("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = job_path(job_id)
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with file_lock(JOBS_LOCK_FILE):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)


def read_job_status(job_id):
    path = job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_recognition_payload(result, relaxed, thresholds, selected_bands, queue_wait):
    return {
        "faces": [r["name"] for r in result],
        "details": result,
        "mode": "relaxed" if relaxed else "default",
        "thresholds": thresholds,
        "bands": sorted(selected_bands),
        "queue_wait": round(queue_wait, 3),
    }


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


def image_content_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def find_avatar_file(name):
    safe_name = safe_path_segment(name, "")
    if not safe_name:
        return None
    avatar_dir = os.path.join(AVATAR_DIR, safe_name)
    if os.path.isdir(avatar_dir):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            photo = os.path.join(avatar_dir, "1" + ext)
            if os.path.exists(photo):
                return photo

    base_dir = os.path.dirname(os.path.abspath(__file__))
    faces_base = os.path.join(base_dir, "faces")
    if not os.path.isdir(faces_base):
        return None
    for band_dir in sorted(os.scandir(faces_base), key=lambda e: e.name):
        if not band_dir.is_dir():
            continue
        faces_dir = os.path.join(band_dir.path, safe_name)
        if not os.path.isdir(faces_dir):
            continue
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            photo = os.path.join(faces_dir, "1" + ext)
            if os.path.exists(photo):
                return photo
    return None


def find_icon_file(name):
    safe_name = safe_path_segment(name, "")
    if not safe_name:
        return None
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        icon = os.path.join(ICON_DIR, safe_name + ext)
        if os.path.exists(icon):
            return icon
    return None


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
        if (
            not os.path.isdir(band_path)
            or band.startswith(".")
            or band == EASTER_EGG_BAND
        ):
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


def display_score(similarity):
    score = 100 / (1 + np.exp(-8 * (float(similarity) - 0.35)))
    return int(round(max(0, min(99, score))))


def infer_name_bands(names):
    groups, _ = load_face_groups()
    by_name = {}
    for band, people in groups.items():
        for name in people:
            by_name.setdefault(name, set()).add(band)
    return [encode_bands(by_name.get(name, set())) for name in names]


def admin_create_token():
    expiry = str(int(time.time()) + ADMIN_TOKEN_TTL)
    sig = hmac.new(ADMIN_PASSWORD.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def admin_check_token(token):
    if not token or not ADMIN_PASSWORD:
        return False
    parts = token.split(".", 1)
    if len(parts) != 2:
        return False
    expiry_str, sig = parts
    try:
        expiry = int(expiry_str)
    except ValueError:
        return False
    if time.time() > expiry:
        return False
    expected = hmac.new(ADMIN_PASSWORD.encode(), expiry_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def admin_list_photos(day=None):
    photos = []
    base = UPLOAD_PHOTOS_DIR
    if day:
        dirs = [daily_upload_dir(day)]
    else:
        dirs = []
        if os.path.isdir(base):
            for d in sorted(os.listdir(base), reverse=True):
                dp = os.path.join(base, d)
                if os.path.isdir(dp) and not d.startswith("."):
                    dirs.append(dp)
    history_lookup = {}
    for photo_dir in dirs:
        day_name = os.path.basename(photo_dir)
        hf = daily_history_file(day_name)
        if not os.path.isfile(hf):
            # also check old-style history.json
            old_hf = os.path.join(HISTORY_DIR, "history.json")
            if os.path.isfile(old_hf) and not history_lookup:
                try:
                    with open(old_hf, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                    for rec in old_data:
                        photo_rel = rec.get("photo", "")
                        if photo_rel.startswith("uploads/"):
                            photo_rel = photo_rel[len("uploads/"):]
                        history_lookup.setdefault(photo_rel, []).append({
                            "faces": rec.get("faces", []),
                            "mode": rec.get("mode", ""),
                            "bands": rec.get("bands", []),
                            "time": rec.get("time", ""),
                        })
                except Exception:
                    pass
            continue
        with open(hf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    photo_rel = rec.get("photo", "")
                    if photo_rel.startswith("uploads/"):
                        photo_rel = photo_rel[len("uploads/"):]
                    history_lookup.setdefault(photo_rel, []).append({
                        "faces": rec.get("faces", []),
                        "mode": rec.get("mode", ""),
                        "bands": rec.get("bands", []),
                        "time": rec.get("time", ""),
                    })
                except json.JSONDecodeError:
                    pass
    for photo_dir in dirs:
        if not os.path.isdir(photo_dir):
            continue
        day_name = os.path.basename(photo_dir)
        for fn in sorted(os.listdir(photo_dir), reverse=True):
            if fn.startswith(".") or not os.path.splitext(fn)[1].lower() in ALLOWED_IMAGE_EXTENSIONS:
                continue
            fp = os.path.join(photo_dir, fn)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            photo_rel = f"photos/{day_name}/{fn}"
            photos.append({
                "filename": fn,
                "day": day_name,
                "path": photo_rel,
                "size": size,
                "history": history_lookup.get(photo_rel),
            })
    return photos


def admin_list_history(day=None):
    records = []
    if day:
        paths = [daily_history_file(day)]
    else:
        if os.path.isdir(HISTORY_DIR):
            paths = sorted(
                [os.path.join(HISTORY_DIR, f) for f in os.listdir(HISTORY_DIR)
                 if f.endswith(".jsonl") and not f.startswith(".")],
                reverse=True,
            )
        else:
            paths = []
    for hp in paths:
        if not os.path.isfile(hp):
            continue
        with open(hp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def admin_list_feedback():
    records = []
    if not os.path.isfile(FEEDBACK_FILE):
        return records
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def admin_list_faces_upload():
    result = {}
    if not os.path.isdir(FACES_UPLOAD_DIR):
        return result
    for band in sorted(os.listdir(FACES_UPLOAD_DIR)):
        band_path = os.path.join(FACES_UPLOAD_DIR, band)
        if not os.path.isdir(band_path) or band.startswith("."):
            continue
        roles = {}
        for role in sorted(os.listdir(band_path)):
            role_path = os.path.join(band_path, role)
            if not os.path.isdir(role_path) or role.startswith("."):
                continue
            files = []
            for fn in sorted(os.listdir(role_path)):
                if fn.startswith("."):
                    continue
                fp = os.path.join(role_path, fn)
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = 0
                files.append({"filename": fn, "size": size})
            if files:
                roles[role] = files
        if roles:
            result[band] = roles
    return result


def admin_stats():
    photo_count = 0
    photo_size = 0
    if os.path.isdir(UPLOAD_PHOTOS_DIR):
        for root, _, files in os.walk(UPLOAD_PHOTOS_DIR):
            for fn in files:
                if fn.startswith("."):
                    continue
                fp = os.path.join(root, fn)
                try:
                    photo_size += os.path.getsize(fp)
                    photo_count += 1
                except OSError:
                    pass
    history_days = 0
    history_records = 0
    if os.path.isdir(HISTORY_DIR):
        for fn in os.listdir(HISTORY_DIR):
            if fn.endswith(".jsonl") and not fn.startswith("."):
                history_days += 1
                with open(os.path.join(HISTORY_DIR, fn), "r", encoding="utf-8") as f:
                    history_records += sum(1 for l in f if l.strip())
    feedback_count = 0
    if os.path.isfile(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedback_count = sum(1 for l in f if l.strip())
    faces_upload_count = 0
    faces_upload_size = 0
    if os.path.isdir(FACES_UPLOAD_DIR):
        for root, _, files in os.walk(FACES_UPLOAD_DIR):
            for fn in files:
                if fn.startswith("."):
                    continue
                fp = os.path.join(root, fn)
                try:
                    faces_upload_size += os.path.getsize(fp)
                    faces_upload_count += 1
                except OSError:
                    pass
    return {
        "photo_count": photo_count,
        "photo_size": photo_size,
        "history_days": history_days,
        "history_records": history_records,
        "feedback_count": feedback_count,
        "faces_upload_count": faces_upload_count,
        "faces_upload_size": faces_upload_size,
        "job_queue_size": job_queue.qsize(),
    }


ADMIN_HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin.html")


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
    with torch.inference_mode():
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
    easter_egg_indices = [
        idx
        for idx, name in enumerate(names)
        if EASTER_EGG_BAND in person_band_sets[idx]
        and (name.lower() == EASTER_EGG_NAME.lower() or name == BIG_BROTHER_NAME)
    ]
    for idx in easter_egg_indices:
        band_mask[idx] = True
    if not np.any(band_mask):
        return []
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
        easter_egg_filtered_idx = next(
            (
                idx
                for idx, name in enumerate(filtered_names)
                if name.lower() == EASTER_EGG_NAME.lower()
                and EASTER_EGG_BAND in filtered_band_sets[idx]
            ),
            None,
        )
        big_brother_filtered_idx = next(
            (
                idx
                for idx, name in enumerate(filtered_names)
                if name == BIG_BROTHER_NAME
                and EASTER_EGG_BAND in filtered_band_sets[idx]
            ),
            None,
        )
        hidden_indices = {
            idx
            for idx in (easter_egg_filtered_idx, big_brother_filtered_idx)
            if idx is not None
        }
        raw_max_idx = int(np.argmax(cos_results))
        visible_indices = [
            idx for idx in range(len(cos_results)) if idx not in hidden_indices
        ]
        if not visible_indices and raw_max_idx not in hidden_indices:
            visible_indices = [raw_max_idx]
        if not visible_indices:
            continue
        max_idx = max(visible_indices, key=lambda idx: cos_results[idx])
        if (
            big_brother_filtered_idx is not None
            and display_score(cos_results[big_brother_filtered_idx])
            >= BIG_BROTHER_TRIGGER_SCORE
        ):
            max_idx = big_brother_filtered_idx
            easter_egg_triggered = "big_brother"
        elif (
            easter_egg_filtered_idx is not None
            and display_score(cos_results[easter_egg_filtered_idx])
            >= EASTER_EGG_TRIGGER_SCORE
        ):
            max_idx = easter_egg_filtered_idx
            easter_egg_triggered = "liyuu"
        else:
            easter_egg_triggered = ""
        top_indices = [
            idx
            for idx in np.argsort(cos_results)[::-1]
            if idx not in hidden_indices
        ][:5]
        top5 = [
            {
                "name": filtered_names[idx],
                "band": encode_bands(filtered_band_sets[idx]),
                "bands": sorted(filtered_band_sets[idx]),
                "similarity": round(float(cos_results[idx]), 4),
                "display_score": display_score(cos_results[idx]),
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
                "name": (
                    EASTER_EGG_DISPLAY_NAME
                    if easter_egg_triggered == "liyuu"
                    else filtered_names[max_idx]
                ),
                "avatar_name": filtered_names[max_idx],
                "band": encode_bands(filtered_band_sets[max_idx]),
                "bands": sorted(filtered_band_sets[max_idx]),
                "similarity": round(float(cos_results[max_idx]), 4),
                "display_score": display_score(cos_results[max_idx]),
                "top5": [] if easter_egg_triggered else top5,
                "easter_egg": easter_egg_triggered,
                "easter_egg_message": (
                    EASTER_EGG_MESSAGE
                    if easter_egg_triggered == "liyuu"
                    else BIG_BROTHER_MESSAGE
                    if easter_egg_triggered == "big_brother"
                    else ""
                ),
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

    @classmethod
    def process_recognition(cls, body, thresholds, selected_bands, relaxed, queue_wait):
        now = time.localtime()
        day = daily_key(now)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        with recognition_lock:
            result = recognize(
                cls.mtcnn,
                cls.adaface,
                cls.names,
                cls.bands,
                cls.feature_db,
                cls.feature_norms,
                body,
                thresholds,
                selected_bands,
            )
        photo_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.jpg"
        photo_dir = daily_upload_dir(day)
        os.makedirs(photo_dir, exist_ok=True)
        photo_path = os.path.join(photo_dir, photo_name)
        photo_relpath = f"uploads/photos/{day}/{photo_name}"
        nparr = np.frombuffer(body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img = resize_for_recognition(img)
            if cv2.imwrite(photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, 10]):
                append_history(
                    {
                        "photo": photo_relpath,
                        "faces": result,
                        "mode": "relaxed" if relaxed else "default",
                        "bands": sorted(selected_bands),
                        "time": timestamp,
                    },
                    day,
                )
            else:
                print(f"[server] failed to save upload photo: {photo_path}", flush=True)
        return make_recognition_payload(
            result, relaxed, thresholds, selected_bands, queue_wait
        )

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def admin_get_token(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("sm_admin="):
                return part[len("sm_admin="):]
        return ""

    def admin_require(self):
        token = self.admin_get_token()
        if not admin_check_token(token):
            self.send_json(401, {"error": "unauthorized"})
            return None
        return token

    def send_static_file(self, path, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", f"public, max-age={STATIC_CACHE_SECONDS}")
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "index.html"
            )
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
        elif path == "/uploads":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if os.path.isfile(ADMIN_HTML_FILE):
                with open(ADMIN_HTML_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"admin.html not found")
        elif path == "/admin/api/stats":
            if not self.admin_require():
                return
            self.send_json(200, admin_stats())
        elif path == "/admin/api/photos":
            if not self.admin_require():
                return
            params = urllib.parse.parse_qs(parsed.query)
            day = params.get("day", [""])[0] or None
            self.send_json(200, {"photos": admin_list_photos(day)})
        elif path == "/admin/api/history":
            if not self.admin_require():
                return
            params = urllib.parse.parse_qs(parsed.query)
            day = params.get("day", [""])[0] or None
            self.send_json(200, {"records": admin_list_history(day)})
        elif path == "/admin/api/feedback":
            if not self.admin_require():
                return
            self.send_json(200, {"records": admin_list_feedback()})
        elif path == "/admin/api/faces_upload":
            if not self.admin_require():
                return
            self.send_json(200, {"data": admin_list_faces_upload()})
        elif path == "/admin/api/photo_days":
            if not self.admin_require():
                return
            days = []
            if os.path.isdir(UPLOAD_PHOTOS_DIR):
                for d in sorted(os.listdir(UPLOAD_PHOTOS_DIR), reverse=True):
                    dp = os.path.join(UPLOAD_PHOTOS_DIR, d)
                    if not os.path.isdir(dp) or d.startswith("."):
                        continue
                    count = 0
                    size = 0
                    for fn in os.listdir(dp):
                        if fn.startswith("."):
                            continue
                        if os.path.splitext(fn)[1].lower() not in ALLOWED_IMAGE_EXTENSIONS:
                            continue
                        count += 1
                        try:
                            size += os.path.getsize(os.path.join(dp, fn))
                        except OSError:
                            pass
                    if count:
                        days.append({"day": d, "count": count, "size": size})
            self.send_json(200, {"days": days})
        elif path == "/admin/api/history_days":
            if not self.admin_require():
                return
            days = []
            if os.path.isdir(HISTORY_DIR):
                days = sorted(
                    [os.path.splitext(f)[0] for f in os.listdir(HISTORY_DIR)
                     if f.endswith(".jsonl") and not f.startswith(".")],
                    reverse=True,
                )
            self.send_json(200, {"days": days})
        elif path.startswith("/admin/photo/"):
            if not self.admin_require():
                return
            rel = urllib.parse.unquote(path[len("/admin/photo/"):])
            abs_path = os.path.normpath(os.path.join(UPLOADS_DIR, rel))
            if not abs_path.startswith(UPLOADS_DIR) or not os.path.isfile(abs_path):
                self.send_response(404)
                self.end_headers()
                return
            self.send_static_file(abs_path, image_content_type(abs_path))
        elif path.startswith("/admin/faces_upload_photo/"):
            if not self.admin_require():
                return
            rel = urllib.parse.unquote(path[len("/admin/faces_upload_photo/"):])
            abs_path = os.path.normpath(os.path.join(FACES_UPLOAD_DIR, rel))
            if not abs_path.startswith(FACES_UPLOAD_DIR) or not os.path.isfile(abs_path):
                self.send_response(404)
                self.end_headers()
                return
            ext = os.path.splitext(abs_path)[1].lower()
            ct = "image/jpeg"
            if ext == ".png":
                ct = "image/png"
            elif ext == ".webp":
                ct = "image/webp"
            self.send_static_file(abs_path, ct)
        elif path.startswith("/avatar/"):
            name = urllib.parse.unquote(self.path[len("/avatar/"):])
            photo = find_avatar_file(name)
            if photo:
                self.send_static_file(photo, image_content_type(photo))
                return
            self.send_response(404)
            self.end_headers()
        elif self.path.startswith("/icon/"):
            name = urllib.parse.unquote(self.path[len("/icon/"):])
            icon = find_icon_file(name)
            if icon:
                self.send_static_file(icon, image_content_type(icon))
                return
            self.send_response(404)
            self.end_headers()
        elif self.path.startswith("/job/"):
            job_id = urllib.parse.unquote(self.path[len("/job/"):])
            status = read_job_status(job_id)
            if status is None:
                self.send_json(404, {"error": "job not found"})
                return
            self.send_json(200, status)
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
        if parsed_path.path == "/admin/login":
            body = self.rfile.read(max(content_length, 0))
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_json(400, {"error": "invalid json"})
                return
            pwd = data.get("password", "")
            if not ADMIN_PASSWORD or pwd != ADMIN_PASSWORD:
                self.send_json(403, {"error": "wrong password"})
                return
            tok = admin_create_token()
            data = json.dumps({"token": tok}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Set-Cookie", f"sm_admin={tok}; Path=/")
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed_path.path == "/admin/api/delete_photo":
            if not self.admin_require():
                return
            body = self.rfile.read(max(content_length, 0))
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_json(400, {"error": "invalid json"})
                return
            rel = data.get("path", "")
            abs_path = os.path.normpath(os.path.join(UPLOADS_DIR, rel))
            if not abs_path.startswith(UPLOADS_DIR) or not os.path.isfile(abs_path):
                self.send_json(404, {"error": "file not found"})
                return
            os.remove(abs_path)
            self.send_json(200, {"ok": True})
            return
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
        body = self.rfile.read(content_length)
        if params.get("async", [""])[0] == "1":
            job_id = uuid.uuid4().hex
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            write_job_status(
                job_id,
                {
                    "status": "queued",
                    "created_at": now,
                    "updated_at": now,
                    "mode": "relaxed" if relaxed else "default",
                    "bands": sorted(selected_bands),
                    "worker_pid": os.getpid(),
                },
            )
            try:
                job_queue.put_nowait(
                    {
                        "job_id": job_id,
                        "body": body,
                        "thresholds": thresholds,
                        "selected_bands": selected_bands,
                        "relaxed": relaxed,
                        "content_length": content_length,
                        "queued_at": time.time(),
                    }
                )
            except queue.Full:
                write_job_status(
                    job_id,
                    {
                        "status": "failed",
                        "error": "服务器排队人数过多，请稍后再试",
                        "created_at": now,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                self.send_json(503, {"error": "服务器排队人数过多，请稍后再试"})
                return
            self.send_json(202, {"job_id": job_id, "status": "queued"})
            return

        queue_started = time.time()
        if not recognition_slots.acquire(timeout=RECOGNITION_QUEUE_TIMEOUT):
            self.send_json(
                503,
                {
                    "error": "服务器正在识别中，请稍后再试",
                    "queue_timeout": RECOGNITION_QUEUE_TIMEOUT,
                },
            )
            return
        queue_wait = time.time() - queue_started

        try:
            payload = self.process_recognition(
                body, thresholds, selected_bands, relaxed, queue_wait
            )
            self.send_json(200, payload)
            elapsed = time.time() - started
            print(
                f"[server] POST {content_length} bytes mode={'relaxed' if relaxed else 'default'} bands={','.join(sorted(selected_bands))} wait={queue_wait:.2f}s -> {len(payload['faces'])} faces in {elapsed:.2f}s",
                flush=True,
            )
        except Exception as e:
            self.send_json(500, {"error": str(e)})
        finally:
            recognition_slots.release()

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


def recognition_job_worker():
    while True:
        job = job_queue.get()
        job_id = job["job_id"]
        started = time.time()
        queue_wait = started - job["queued_at"]
        try:
            write_job_status(
                job_id,
                {
                    "status": "running",
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "queue_wait": round(queue_wait, 3),
                    "worker_pid": os.getpid(),
                },
            )
            payload = FaceHandler.process_recognition(
                job["body"],
                job["thresholds"],
                job["selected_bands"],
                job["relaxed"],
                queue_wait,
            )
            elapsed = time.time() - started
            write_job_status(
                job_id,
                {
                    "status": "done",
                    "result": payload,
                    "queue_wait": round(queue_wait, 3),
                    "elapsed": round(elapsed, 3),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "worker_pid": os.getpid(),
                },
            )
            print(
                f"[server] JOB {job_id} {job['content_length']} bytes mode={'relaxed' if job['relaxed'] else 'default'} wait={queue_wait:.2f}s -> {len(payload['faces'])} faces in {elapsed:.2f}s",
                flush=True,
            )
        except Exception as e:
            write_job_status(
                job_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "worker_pid": os.getpid(),
                },
            )
        finally:
            job_queue.task_done()


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

    worker = threading.Thread(target=recognition_job_worker, daemon=True)
    worker.start()

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
