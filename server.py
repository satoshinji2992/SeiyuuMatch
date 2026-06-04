import os
import sys
import socket
import argparse
import multiprocessing as mp
import numpy as np
import cv2
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

from face_swap import load_inswapper, swap_faces_insightface

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
JOB_RESULTS_DIR = os.path.join(JOBS_DIR, "results")
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
os.makedirs(JOB_RESULTS_DIR, exist_ok=True)
SWAP_SAVED_DIR = os.path.join(UPLOADS_DIR, "swap")
os.makedirs(SWAP_SAVED_DIR, exist_ok=True)

history_lock = threading.Lock()
RECOGNITION_QUEUE_TIMEOUT = float(os.environ.get("RECOGNITION_QUEUE_TIMEOUT", "30"))
JOB_QUEUE_MAX_SIZE = int(os.environ.get("JOB_QUEUE_MAX_SIZE", "200"))
MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", "1280"))
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_SWAP_UPLOAD_BYTES = int(os.environ.get("MAX_SWAP_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_SWAP_IMAGE_DIM = int(os.environ.get("MAX_SWAP_IMAGE_DIM", "0"))
MAX_SWAP_METADATA_BYTES = 128 * 1024
MAX_FACE_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_FACE_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024
MAX_FEEDBACK_BYTES = 16 * 1024
JOB_STALE_TIMEOUT = int(os.environ.get("JOB_STALE_TIMEOUT", "300"))
SWAP_DEBUG_OUTPUT = os.environ.get("SWAP_DEBUG_OUTPUT", "0") == "1"
SWAP_RESTORE_CMD = os.environ.get("SWAP_RESTORE_CMD", "")
SWAP_RESTORE_BACKEND = os.environ.get("SWAP_RESTORE_BACKEND", "")
CODEFORMER_DIR = os.environ.get("CODEFORMER_DIR", "")
CODEFORMER_WEIGHT = float(os.environ.get("CODEFORMER_WEIGHT", "0.5"))
CODEFORMER_FACE_UPSAMPLE = os.environ.get("CODEFORMER_FACE_UPSAMPLE", "1") == "1"
SWAP_IDENTITY_BLEND = float(os.environ.get("SWAP_IDENTITY_BLEND", "1.0"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
STATIC_CACHE_SECONDS = 7 * 24 * 60 * 60
FACE_GROUPS_CACHE_SECONDS = float(os.environ.get("FACE_GROUPS_CACHE_SECONDS", "30"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
ADMIN_TOKEN_TTL = 24 * 60 * 60
INSWAPPER_MODEL_NAME = os.environ.get("INSWAPPER_MODEL_NAME", "inswapper_128.onnx")
recognition_job_queue = mp.Queue(maxsize=JOB_QUEUE_MAX_SIZE)
swap_job_queue = mp.Queue(maxsize=JOB_QUEUE_MAX_SIZE)
face_groups_cache_lock = threading.Lock()
face_groups_cache_data = None
face_groups_cache_at = 0.0
HIDDEN_PROJECT = "__hidden__"
HIDDEN_GROUP = "???"
HIDDEN_SPECIAL_NAMES = {"立希", "高松灯", "要乐奈"}
HIDDEN_SPECIAL_TRIGGER_SCORE = 70
HIDDEN_SPECIAL_MESSAGE = "老大哥正在看着你"
DEFAULT_SELECTED_GROUPS = {"bangdream:mygo", "bangdream:avemujica", "bangdream:sumimi"}


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


def job_result_path(job_id):
    return os.path.join(JOB_RESULTS_DIR, f"{safe_path_segment(job_id, 'job')}.png")


def sanitize_recognition_details(details):
    if not isinstance(details, list):
        return []
    sanitized = []
    for item in details:
        if not isinstance(item, dict):
            continue
        cleaned = {}
        for key in [
            "name",
            "avatar_name",
            "avatar_project",
            "avatar_group",
            "project",
            "group",
            "easter_egg",
            "easter_egg_message",
        ]:
            value = item.get(key)
            if isinstance(value, str):
                cleaned[key] = value[:120]
        if isinstance(item.get("avatar_feature_index"), int):
            cleaned["avatar_feature_index"] = item["avatar_feature_index"]
        else:
            try:
                cleaned["avatar_feature_index"] = int(item.get("avatar_feature_index"))
            except (TypeError, ValueError):
                pass
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                cleaned["bbox"] = [float(value) for value in bbox]
            except (TypeError, ValueError):
                pass
        if "avatar_feature_index" in cleaned:
            sanitized.append(cleaned)
    return sanitized


def job_debug_dir(job_id):
    return os.path.join(JOB_RESULTS_DIR, "debug", safe_path_segment(job_id, "job"))


def hidden_easter_egg_code(name, project, groups):
    if project == HIDDEN_PROJECT and HIDDEN_GROUP in groups and name in HIDDEN_SPECIAL_NAMES:
        return "big_brother"
    return ""


def apply_hidden_easter_egg(detail):
    detail = dict(detail)
    easter_egg = hidden_easter_egg_code(
        detail.get("name", ""),
        detail.get("avatar_project") or detail.get("project") or "",
        parse_values(detail.get("avatar_group") or detail.get("group") or ""),
    )
    if easter_egg:
        detail["easter_egg"] = easter_egg
        detail["easter_egg_message"] = HIDDEN_SPECIAL_MESSAGE
    return detail


def recognition_has_hidden_entry(details):
    for detail in details or []:
        if isinstance(detail, dict) and detail.get("easter_egg") == "big_brother":
            return True
    return False


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
        status = json.load(f)
    if status.get("status") in {"queued", "running"}:
        ts = status.get("started_at") or status.get("created_at") or status.get("updated_at")
        try:
            started = time.strptime(ts, "%Y-%m-%d %H:%M:%S") if ts else None
        except ValueError:
            started = None
        if started is not None:
            age = time.time() - time.mktime(started)
            if age > JOB_STALE_TIMEOUT:
                status = dict(status)
                status["status"] = "failed"
                job_type = status.get("job_type")
                if job_type == "swap":
                    status["error"] = "换脸任务超时，请重试"
                else:
                    status["error"] = "识别任务超时，请重试"
                status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    write_job_status(job_id, status)
                except Exception:
                    pass
    return status


def make_recognition_payload(result, relaxed, det_score_threshold, selected_groups, queue_wait):
    return {
        "faces": [r["name"] for r in result],
        "details": result,
        "mode": "relaxed" if relaxed else "default",
        "det_score_threshold": det_score_threshold,
        "groups": sorted(selected_groups),
        "bands": sorted(selected_groups),
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


def find_avatar_file(name, project="", group=""):
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
    if project:
        if project == HIDDEN_PROJECT:
            candidate_dirs = [os.path.join(faces_base, HIDDEN_GROUP, safe_name)]
        elif group:
            candidate_dirs = [os.path.join(faces_base, project, group, safe_name)]
        else:
            project_dir = os.path.join(faces_base, project)
            candidate_dirs = [
                os.path.join(group_dir.path, safe_name)
                for group_dir in sorted(os.scandir(project_dir), key=lambda e: e.name)
                if group_dir.is_dir()
            ] if os.path.isdir(project_dir) else []
        for faces_dir in candidate_dirs:
            if not os.path.isdir(faces_dir):
                continue
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                photo = os.path.join(faces_dir, "1" + ext)
                if os.path.exists(photo):
                    return photo

    for project_dir in sorted(os.scandir(faces_base), key=lambda e: e.name):
        if not project_dir.is_dir():
            continue
        if project_dir.name == HIDDEN_GROUP:
            candidate_dirs = [os.path.join(project_dir.path, safe_name)]
        else:
            candidate_dirs = [
                os.path.join(group_dir.path, safe_name)
                for group_dir in sorted(os.scandir(project_dir.path), key=lambda e: e.name)
                if group_dir.is_dir()
            ]
        for faces_dir in candidate_dirs:
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
    projects = {}
    if not os.path.isdir(base_dir):
        return load_face_groups_from_features()
    for project in sorted(os.listdir(base_dir)):
        project_path = os.path.join(base_dir, project)
        if not os.path.isdir(project_path) or project.startswith(".") or project == HIDDEN_GROUP:
            continue

        group_map = {}
        count_map = {}
        for group in sorted(os.listdir(project_path)):
            group_path = os.path.join(project_path, group)
            if not os.path.isdir(group_path) or group.startswith("."):
                continue
            people = []
            count_map[group] = {}
            for person in sorted(os.listdir(group_path)):
                person_path = os.path.join(group_path, person)
                if not os.path.isdir(person_path) or person.startswith("."):
                    continue
                people.append(person)
                count_map[group][person] = sum(
                    1
                    for filename in os.listdir(person_path)
                    if os.path.splitext(filename)[1].lower()
                    in ALLOWED_IMAGE_EXTENSIONS
                )
            if people:
                group_map[group] = people
            else:
                count_map.pop(group, None)
        if group_map:
            projects[project] = {"groups": group_map, "counts": count_map}
    return {"projects": projects}


def load_face_groups_from_features():
    features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
    projects = {}
    if not os.path.isfile(features_path):
        return {"projects": projects}
    try:
        data = np.load(features_path, allow_pickle=True)
    except Exception:
        return {"projects": projects}
    if not {"names", "projects", "groups"}.issubset(set(data.files)):
        return {"projects": projects}
    names = [str(v) for v in data["names"]]
    proj_list = [str(v) for v in data["projects"]]
    group_list = [str(v) for v in data["groups"]]
    for name, proj, group in zip(names, proj_list, group_list):
        if proj.startswith("_"):
            continue
        if proj not in projects:
            projects[proj] = {"groups": {}, "counts": {}}
        if group not in projects[proj]["groups"]:
            projects[proj]["groups"][group] = []
            projects[proj]["counts"][group] = {}
        projects[proj]["groups"][group].append(name)
        projects[proj]["counts"][group][name] = 0
    return {"projects": projects}


def get_face_groups_cached():
    global face_groups_cache_data, face_groups_cache_at
    now = time.time()
    with face_groups_cache_lock:
        if (
            face_groups_cache_data is not None
            and now - face_groups_cache_at <= FACE_GROUPS_CACHE_SECONDS
        ):
            return face_groups_cache_data
        face_groups_cache_data = load_face_groups()
        face_groups_cache_at = now
        return face_groups_cache_data


def invalidate_face_groups_cache():
    global face_groups_cache_data, face_groups_cache_at
    with face_groups_cache_lock:
        face_groups_cache_data = None
        face_groups_cache_at = 0.0


def parse_values(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw = value
    else:
        raw = str(value).split(",")
    return {str(b).strip() for b in raw if str(b).strip()}


def encode_values(values):
    return ",".join(sorted(dict.fromkeys(v for v in values if v)))


def group_key(project, group):
    return f"{project}:{group}"


def group_label_from_keys(group_keys, fallback_project="", fallback_groups=None):
    labels = []
    for key in group_keys:
        project, _, group = key.partition(":")
        if project and group:
            labels.append(f"{project}/{group}")
    if labels:
        return ", ".join(dict.fromkeys(labels))
    if fallback_project and fallback_groups:
        return ", ".join(f"{fallback_project}/{group}" for group in fallback_groups)
    return encode_values(fallback_groups or [])


def parse_selected_groups(value):
    return {str(v).strip() for v in value.split(",") if str(v).strip()}


def is_hidden_entry(project, groups, name):
    return bool(hidden_easter_egg_code(name, project, groups))


def display_score(similarity):
    score = 100 / (1 + np.exp(-8 * (float(similarity) - 0.35)))
    return int(round(max(0, min(99, score))))


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
    for project in sorted(os.listdir(FACES_UPLOAD_DIR)):
        project_path = os.path.join(FACES_UPLOAD_DIR, project)
        if not os.path.isdir(project_path) or project.startswith("."):
            continue
        groups = {}
        for group in sorted(os.listdir(project_path)):
            group_path = os.path.join(project_path, group)
            if not os.path.isdir(group_path) or group.startswith("."):
                continue
            roles = {}
            for role in sorted(os.listdir(group_path)):
                role_path = os.path.join(group_path, role)
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
                groups[group] = roles
        if groups:
            result[project] = groups
    return result


def admin_list_swap():
    result = {}
    if not os.path.isdir(SWAP_SAVED_DIR):
        return result
    for day in sorted(os.listdir(SWAP_SAVED_DIR), reverse=True):
        day_path = os.path.join(SWAP_SAVED_DIR, day)
        if not os.path.isdir(day_path) or day.startswith("."):
            continue
        files = []
        for fn in sorted(os.listdir(day_path), reverse=True):
            if fn.startswith(".") or not os.path.splitext(fn)[1].lower() in ALLOWED_IMAGE_EXTENSIONS:
                continue
            fp = os.path.join(day_path, fn)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            files.append({"filename": fn, "size": size})
        if files:
            result[day] = files
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
        "job_queue_size": queue_size_safe(),
    }


def queue_size_safe():
    total = 0
    for q in (recognition_job_queue, swap_job_queue):
        try:
            total += q.qsize()
        except Exception:
            pass
    return total


ADMIN_HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin.html")


FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
INSIGHTFACE_DET_SIZE = int(os.environ.get("INSIGHTFACE_DET_SIZE", "640"))
INSIGHTFACE_DEFAULT_DET_SCORE = float(os.environ.get("INSIGHTFACE_DEFAULT_DET_SCORE", "0.5"))
INSIGHTFACE_RELAXED_DET_SCORE = float(os.environ.get("INSIGHTFACE_RELAXED_DET_SCORE", "0.3"))


def resize_for_recognition(img):
    return resize_for_longest_edge(img, MAX_IMAGE_DIM)


def resize_for_swap(img):
    return resize_for_longest_edge(img, MAX_SWAP_IMAGE_DIM)


def resize_for_longest_edge(img, max_dim):
    if not max_dim or max_dim <= 0:
        return img
    height, width = img.shape[:2]
    longest = max(width, height)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def load_insightface():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(INSIGHTFACE_DET_SIZE, INSIGHTFACE_DET_SIZE))
    return app


def load_feature_bundle(features_path):
    data = np.load(features_path, allow_pickle=True)
    required_keys = {"names", "projects", "groups", "features"}
    if not required_keys.issubset(set(data.files)):
        raise RuntimeError("features.npz uses the old schema; run python3 register.py first")
    names = [str(n) for n in data["names"]]
    projects = [str(p) for p in data["projects"]]
    groups = [str(g) for g in data["groups"]]
    feature_db = data["features"]
    feature_norms = np.linalg.norm(feature_db, axis=1)
    return {
        "names": names,
        "projects": projects,
        "groups": groups,
        "feature_db": feature_db,
        "feature_norms": feature_norms,
    }


def load_runtime(features_path, include_swapper=False):
    runtime = load_feature_bundle(features_path)
    runtime["insightface_app"] = load_insightface()
    runtime["inswapper"] = load_inswapper(INSWAPPER_MODEL_NAME) if include_swapper else None
    return runtime


def detect_faces_insightface(
    insightface_app,
    image_bytes,
    det_score_threshold,
    resize=True,
):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_raw is None:
        return None, []
    if isinstance(resize, int) and not isinstance(resize, bool):
        img_raw = resize_for_longest_edge(img_raw, resize)
    elif resize:
        img_raw = resize_for_recognition(img_raw)
    detected_faces = insightface_app.get(img_raw)
    detected_faces = [f for f in detected_faces if f.det_score >= det_score_threshold]
    return img_raw, detected_faces


def result_entry(name, project, groups, similarity):
    visible_project = "" if project == HIDDEN_PROJECT else project
    group_list = sorted(groups)
    group_keys = [group_key(project, group) for group in group_list] if visible_project else []
    group_label = group_label_from_keys(group_keys, visible_project, group_list)
    return {
        "name": name,
        "project": visible_project,
        "projects": [visible_project] if visible_project else [],
        "identity_key": f"{visible_project}/{name}" if visible_project else name,
        "group": encode_values(group_list),
        "groups": group_list,
        "group_keys": group_keys,
        "group_label": group_label,
        "band": encode_values(group_list),
        "bands": group_list,
        "similarity": round(float(similarity), 4),
        "display_score": display_score(similarity),
    }


def dedup_results(entries):
    merged = {}
    for entry in entries:
        key = entry["name"]
        if key not in merged or entry["similarity"] > merged[key]["similarity"]:
            base = dict(entry)
            if key in merged:
                prev = merged[key]
                base["groups"] = sorted(set(prev["groups"]) | set(base["groups"]))
                base["bands"] = list(base["groups"])
                base["group"] = encode_values(base["groups"])
                base["band"] = base["group"]
                base["group_keys"] = sorted(set(prev.get("group_keys", [])) | set(base.get("group_keys", [])))
                base["group_label"] = group_label_from_keys(
                    base.get("group_keys", []), base.get("project", ""), base["groups"]
                )
                prev_projects = prev["projects"] if prev["projects"] else []
                cur_projects = base["projects"] if base["projects"] else []
                all_projects = list(dict.fromkeys(cur_projects + prev_projects))
                base["projects"] = all_projects
                base["project"] = cur_projects[0] if cur_projects else (all_projects[0] if all_projects else base["project"])
            merged[key] = base
        else:
            prev = merged[key]
            prev["groups"] = sorted(set(prev["groups"]) | set(entry["groups"]))
            prev["bands"] = list(prev["groups"])
            prev["group"] = encode_values(prev["groups"])
            prev["band"] = prev["group"]
            prev["group_keys"] = sorted(set(prev.get("group_keys", [])) | set(entry.get("group_keys", [])))
            prev["group_label"] = group_label_from_keys(
                prev.get("group_keys", []), prev.get("project", ""), prev["groups"]
            )
            prev_projects = prev["projects"] if prev["projects"] else []
            cur_projects = entry["projects"] if entry["projects"] else []
            all_projects = list(dict.fromkeys(prev_projects + cur_projects))
            prev["projects"] = all_projects
            if not prev["project"]:
                prev["project"] = all_projects[0] if all_projects else ""
    return sorted(merged.values(), key=lambda e: e["similarity"], reverse=True)


def recognize_insightface(
    insightface_app,
    names,
    projects,
    groups,
    feature_db,
    feature_norms,
    image_bytes,
    det_score_threshold,
    selected_groups,
):
    img_raw, detected_faces = detect_faces_insightface(
        insightface_app,
        image_bytes,
        det_score_threshold,
        resize=True,
    )
    if img_raw is None:
        return []

    group_sets = [parse_values(group_value) for group_value in groups]
    entry_group_keys = [
        {group_key(project, group) for group in group_set}
        for project, group_set in zip(projects, group_sets)
    ]
    mask_values = []
    for name, project, group_set, keys in zip(names, projects, group_sets, entry_group_keys):
        keep = bool(keys & selected_groups)
        if is_hidden_entry(project, group_set, name):
            keep = True
        mask_values.append(keep)
    entry_mask = np.array(mask_values, dtype=bool)
    if not np.any(entry_mask):
        return []

    filtered_names = [name for name, keep in zip(names, entry_mask) if keep]
    filtered_projects = [project for project, keep in zip(projects, entry_mask) if keep]
    filtered_group_sets = [group_set for group_set, keep in zip(group_sets, entry_mask) if keep]
    filtered_indices = [idx for idx, keep in enumerate(entry_mask) if keep]
    filtered_features = feature_db[entry_mask]
    filtered_norms = feature_norms[entry_mask]
    img_h, img_w = img_raw.shape[:2]

    results = []
    for face in detected_faces:
        vec = np.asarray(face.normed_embedding, dtype=np.float32)
        if vec.size != feature_db.shape[1] or not np.isfinite(vec).all():
            continue
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0 or not np.isfinite(vec_norm):
            continue
        denom = filtered_norms * vec_norm
        valid = np.isfinite(denom) & (denom > 0)
        if not np.any(valid):
            continue
        cos_results = np.full(len(filtered_names), -np.inf, dtype=np.float32)
        cos_results[valid] = filtered_features[valid] @ vec / denom[valid]
        cos_results = np.nan_to_num(cos_results, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

        hidden_indices = {
            idx
            for idx, name in enumerate(filtered_names)
            if is_hidden_entry(filtered_projects[idx], filtered_group_sets[idx], name)
        }
        raw_max_idx = int(np.argmax(cos_results))
        visible_indices = [idx for idx in range(len(cos_results)) if idx not in hidden_indices]
        if not visible_indices:
            continue

        max_idx = max(visible_indices, key=lambda idx: cos_results[idx])
        easter_egg_triggered = ""
        if (
            raw_max_idx in hidden_indices
            and display_score(cos_results[raw_max_idx]) >= HIDDEN_SPECIAL_TRIGGER_SCORE
        ):
            max_idx = raw_max_idx
            easter_egg_triggered = "big_brother"

        top_indices = [
            idx for idx in np.argsort(cos_results)[::-1] if idx not in hidden_indices
        ][:15]
        top5_raw = [
            result_entry(
                filtered_names[idx],
                filtered_projects[idx],
                filtered_group_sets[idx],
                cos_results[idx],
            )
            for idx in top_indices
        ]
        top5 = dedup_results(top5_raw)[:5]

        bbox = None
        box = face.bbox
        if box is not None:
            bbox = [
                float(box[0]) / img_w,
                float(box[1]) / img_h,
                float(box[2]) / img_w,
                float(box[3]) / img_h,
            ]

        best_name = filtered_names[max_idx]
        best_project = filtered_projects[max_idx]
        best_avatar_groups = set(filtered_group_sets[max_idx])
        best_feature_index = filtered_indices[max_idx]
        if easter_egg_triggered:
            result = result_entry(
                best_name,
                best_project,
                best_avatar_groups,
                cos_results[max_idx],
            )
        else:
            same_name_entries = [
                result_entry(
                    filtered_names[idx],
                    filtered_projects[idx],
                    filtered_group_sets[idx],
                    cos_results[idx],
                )
                for idx in range(len(filtered_names))
                if idx not in hidden_indices and filtered_names[idx] == best_name
            ]
            if not same_name_entries:
                continue
            result = dedup_results(same_name_entries)[0]
        result.update(
            {
                "avatar_name": best_name,
                "avatar_project": best_project,
                "avatar_group": sorted(best_avatar_groups)[0]
                if best_avatar_groups
                else "",
                "avatar_feature_index": int(best_feature_index),
                "top5": [] if easter_egg_triggered else top5,
                "easter_egg": easter_egg_triggered,
                "easter_egg_message": (
                    HIDDEN_SPECIAL_MESSAGE
                    if easter_egg_triggered == "big_brother"
                    else ""
                ),
                "bbox": bbox,
            }
        )
        results.append(result)
    return results


class FaceHandler(BaseHTTPRequestHandler):
    insightface_app = None
    inswapper = None
    runtime_lock = threading.Lock()
    names = None
    projects = None
    groups = None
    feature_db = None
    feature_norms = None

    @classmethod
    def ensure_insightface_app(cls):
        if cls.insightface_app is not None:
            return
        with cls.runtime_lock:
            if cls.insightface_app is None:
                print("[server] lazy loading InsightFace for sync recognition...", flush=True)
                cls.insightface_app = load_insightface()

    @classmethod
    def process_recognition(cls, body, det_score_threshold, selected_groups, relaxed, queue_wait):
        now = time.localtime()
        day = daily_key(now)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        cls.ensure_insightface_app()
        result = recognize_insightface(
            cls.insightface_app,
            cls.names,
            cls.projects,
            cls.groups,
            cls.feature_db,
            cls.feature_norms,
            body,
            det_score_threshold,
            selected_groups,
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
                        "groups": sorted(selected_groups),
                        "bands": sorted(selected_groups),
                        "time": timestamp,
                    },
                    day,
                )
            else:
                print(f"[server] failed to save upload photo: {photo_path}", flush=True)
        return make_recognition_payload(
            result, relaxed, det_score_threshold, selected_groups, queue_wait
        )

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_image(self, image_bgr):
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            self.send_json(500, {"error": "image encode failed"})
            return
        data = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
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
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Cache-Control", f"public, max-age={STATIC_CACHE_SECONDS}")
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

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

        project_value = ""
        group_value = ""
        name_value = ""
        photos = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_param("filename", header="content-disposition")
            payload = part.get_payload(decode=True) or b""
            if name == "project":
                project_value = payload.decode(part.get_content_charset() or "utf-8", "replace")
            elif name == "group":
                group_value = payload.decode(part.get_content_charset() or "utf-8", "replace")
            elif name == "role":
                name_value = payload.decode(part.get_content_charset() or "utf-8", "replace")
            elif name == "photos" and filename:
                photos.append((filename, payload))

        project = safe_path_segment(project_value, "unknown_project")
        group = safe_path_segment(group_value, "unknown_group")
        role = safe_path_segment(name_value, "unknown_role")
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
        target_dir = os.path.join(FACES_UPLOAD_DIR, project, group, role)
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
            f"[server] face upload project={project} group={group} role={role} files={len(saved)} bytes={content_length}",
            flush=True,
        )
        invalidate_face_groups_cache()
        self.send_json(200, {"saved": saved, "project": project, "group": group, "role": role})

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

    def handle_face_swap(self, content_length):
        if content_length <= 0:
            self.send_json(400, {"error": "empty image"})
            return
        if content_length > MAX_SWAP_UPLOAD_BYTES:
            self.send_json(413, {"error": "image too large"})
            return

        content_type = self.headers.get("Content-Type", "")
        request_body = self.rfile.read(content_length)
        body = request_body
        recognition_details = []
        if content_type.startswith("multipart/form-data"):
            message = BytesParser(policy=email_policy).parsebytes(
                (
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {content_length}\r\n"
                    "\r\n"
                ).encode("utf-8")
                + request_body
            )
            body = b""
            details_payload = b""
            for part in message.iter_parts():
                if part.get_content_disposition() != "form-data":
                    continue
                name = part.get_param("name", header="content-disposition")
                payload = part.get_payload(decode=True) or b""
                if name == "image":
                    body = payload
                elif name == "details" and len(payload) <= MAX_SWAP_METADATA_BYTES:
                    details_payload = payload
            if details_payload:
                try:
                    recognition_details = [
                        apply_hidden_easter_egg(detail)
                        for detail in sanitize_recognition_details(
                            json.loads(details_payload.decode("utf-8"))
                        )
                    ]
                except Exception:
                    recognition_details = []
            if recognition_has_hidden_entry(recognition_details):
                self.send_json(400, {"error": "该识别结果不支持换脸"})
                return
        if not body:
            self.send_json(400, {"error": "empty image"})
            return
        parsed_path = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_path.query)
        relaxed = params.get("mode", [""])[0] == "relaxed"
        requested_groups = parse_selected_groups(params.get("groups", [""])[0])
        selected_groups = requested_groups or set(DEFAULT_SELECTED_GROUPS)
        det_score_threshold = (
            INSIGHTFACE_RELAXED_DET_SCORE if relaxed else INSIGHTFACE_DEFAULT_DET_SCORE
        )
        job_id = uuid.uuid4().hex
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        write_job_status(
            job_id,
            {
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "mode": "relaxed" if relaxed else "default",
                "groups": sorted(selected_groups),
                "bands": sorted(selected_groups),
                "worker_pid": os.getpid(),
                "job_type": "swap",
            },
        )
        try:
            swap_job_queue.put_nowait(
                {
                    "job_id": job_id,
                    "body": body,
                    "det_score_threshold": det_score_threshold,
                    "selected_groups": selected_groups,
                    "relaxed": relaxed,
                    "content_length": content_length,
                    "queued_at": time.time(),
                    "recognition_details": recognition_details,
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
                    "job_type": "swap",
                },
            )
            self.send_json(503, {"error": "服务器排队人数过多，请稍后再试"})
            return
        self.send_json(202, {"job_id": job_id, "status": "queued"})

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
        elif path == "/admin/api/swap":
            if not self.admin_require():
                return
            self.send_json(200, {"data": admin_list_swap()})
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
        elif path.startswith("/admin/swap_photo/"):
            if not self.admin_require():
                return
            rel = urllib.parse.unquote(path[len("/admin/swap_photo/"):])
            abs_path = os.path.normpath(os.path.join(SWAP_SAVED_DIR, rel))
            if not abs_path.startswith(SWAP_SAVED_DIR) or not os.path.isfile(abs_path):
                self.send_response(404)
                self.end_headers()
                return
            self.send_static_file(abs_path, image_content_type(abs_path))
        elif path.startswith("/avatar/"):
            parts = [
                urllib.parse.unquote(part)
                for part in path[len("/avatar/"):].split("/")
                if part
            ]
            if len(parts) >= 3:
                project, group, name = parts[0], parts[1], "/".join(parts[2:])
            else:
                project, group, name = "", "", "/".join(parts)
            photo = find_avatar_file(name, project, group)
            if photo:
                self.send_static_file(photo, image_content_type(photo))
                return
            self.send_response(404)
            self.end_headers()
        elif path.startswith("/icon/"):
            name = urllib.parse.unquote(path[len("/icon/"):])
            icon = find_icon_file(name)
            if icon:
                self.send_static_file(icon, image_content_type(icon))
                return
            self.send_response(404)
            self.end_headers()
        elif path.startswith("/job/"):
            job_id = urllib.parse.unquote(path[len("/job/"):])
            status = read_job_status(job_id)
            if status is None:
                self.send_json(404, {"error": "job not found"})
                return
            self.send_json(200, status)
        elif path.startswith("/job_result/"):
            job_id = urllib.parse.unquote(path[len("/job_result/"):])
            img_path = job_result_path(job_id)
            if os.path.isfile(img_path):
                self.send_static_file(img_path, image_content_type(img_path))
                return
            self.send_response(404)
            self.end_headers()
        elif path == "/people":
            self.send_json(200, {"names": self.names})
        elif path == "/face_groups":
            self.send_json(200, get_face_groups_cached())
        elif path == "/health":
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
        if parsed_path.path == "/swap_faces":
            self.handle_face_swap(content_length)
            return

        params = urllib.parse.parse_qs(parsed_path.query)
        relaxed = params.get("mode", [""])[0] == "relaxed"
        requested_groups = parse_selected_groups(params.get("groups", [""])[0])
        selected_groups = requested_groups or set(DEFAULT_SELECTED_GROUPS)
        det_score_threshold = (
            INSIGHTFACE_RELAXED_DET_SCORE if relaxed else INSIGHTFACE_DEFAULT_DET_SCORE
        )
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
                    "groups": sorted(selected_groups),
                    "bands": sorted(selected_groups),
                    "worker_pid": os.getpid(),
                },
            )
            try:
                recognition_job_queue.put_nowait(
                    {
                        "job_id": job_id,
                        "body": body,
                        "det_score_threshold": det_score_threshold,
                        "selected_groups": selected_groups,
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
        queue_wait = 0.0
        try:
            payload = self.process_recognition(
                body, det_score_threshold, selected_groups, relaxed, queue_wait
            )
            self.send_json(200, payload)
            elapsed = time.time() - started
            print(
                f"[server] POST {content_length} bytes mode={'relaxed' if relaxed else 'default'} groups={','.join(sorted(selected_groups))} wait={queue_wait:.2f}s -> {len(payload['faces'])} faces in {elapsed:.2f}s",
                flush=True,
            )
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


def recognition_job_worker(features_path, job_queue):
    runtime = load_runtime(features_path, include_swapper=False)
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
                    "worker_kind": "recognition",
                },
            )
            payload = recognize_insightface(
                runtime["insightface_app"],
                runtime["names"],
                runtime["projects"],
                runtime["groups"],
                runtime["feature_db"],
                runtime["feature_norms"],
                job["body"],
                job["det_score_threshold"],
                job["selected_groups"],
            )
            payload = make_recognition_payload(
                payload, job["relaxed"], job["det_score_threshold"], job["selected_groups"], queue_wait
            )

            now_local = time.localtime()
            day = daily_key(now_local)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now_local)
            photo_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.jpg"
            photo_dir = daily_upload_dir(day)
            os.makedirs(photo_dir, exist_ok=True)
            photo_path = os.path.join(photo_dir, photo_name)
            photo_relpath = f"uploads/photos/{day}/{photo_name}"
            nparr = np.frombuffer(job["body"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = resize_for_recognition(img)
                if cv2.imwrite(photo_path, img, [cv2.IMWRITE_JPEG_QUALITY, 10]):
                    append_history(
                        {
                            "photo": photo_relpath,
                            "faces": payload["faces"],
                            "details": payload.get("details", []),
                            "mode": "relaxed" if job["relaxed"] else "default",
                            "groups": sorted(job["selected_groups"]),
                            "bands": sorted(job["selected_groups"]),
                            "time": timestamp,
                        },
                        day,
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
                    "worker_kind": "recognition",
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
                    "worker_kind": "recognition",
                },
            )
        finally:
            try:
                job_queue.task_done()
            except Exception:
                pass


def swap_job_worker(features_path, job_queue):
    runtime = load_runtime(features_path, include_swapper=True)
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
                    "worker_kind": "swap",
                },
            )
            swapped_img, recognition = swap_faces_insightface(
                runtime["insightface_app"],
                runtime["inswapper"],
                runtime["names"],
                runtime["projects"],
                runtime["groups"],
                runtime["feature_db"],
                runtime["feature_norms"],
                job["body"],
                job["det_score_threshold"],
                job["selected_groups"],
                detect_faces=detect_faces_insightface,
                recognize=recognize_insightface,
                debug_dir=job_debug_dir(job_id) if SWAP_DEBUG_OUTPUT else None,
                restore_cmd=SWAP_RESTORE_CMD,
                restore_backend=SWAP_RESTORE_BACKEND,
                codeformer_dir=CODEFORMER_DIR,
                codeformer_weight=CODEFORMER_WEIGHT,
                codeformer_face_upsample=CODEFORMER_FACE_UPSAMPLE,
                recognition_details=job.get("recognition_details"),
                swap_image_max_dim=MAX_SWAP_IMAGE_DIM,
                identity_blend=SWAP_IDENTITY_BLEND,
            )
            if swapped_img is None or not recognition:
                raise RuntimeError("没有检测到可替换的人脸")
            out_h, out_w = swapped_img.shape[:2]
            result_path = job_result_path(job_id)
            ok, encoded = cv2.imencode(".png", swapped_img)
            if not ok:
                raise RuntimeError("image encode failed")
            with open(result_path, "wb") as f:
                f.write(encoded.tobytes())
            now_local = time.localtime()
            swap_day = daily_key(now_local)
            swap_timestamp = time.strftime("%Y%m%d_%H%M%S", now_local)
            swap_names = "_".join(dict.fromkeys(item["name"] for item in recognition)) if recognition else "unknown"
            swap_save_dir = os.path.join(SWAP_SAVED_DIR, swap_day)
            os.makedirs(swap_save_dir, exist_ok=True)
            swap_save_name = f"{swap_timestamp}_{safe_path_segment(swap_names, 'swap')}_{job_id[:8]}.png"
            cv2.imwrite(os.path.join(swap_save_dir, swap_save_name), swapped_img)
            elapsed = time.time() - started
            write_job_status(
                job_id,
                {
                    "status": "done",
                    "result": {
                        "image_path": f"/job_result/{job_id}",
                        "faces": [item["name"] for item in recognition],
                    },
                    "queue_wait": round(queue_wait, 3),
                    "elapsed": round(elapsed, 3),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "worker_pid": os.getpid(),
                    "worker_kind": "swap",
                },
            )
            print(
                f"[server] SWAP JOB {job_id} {job['content_length']} bytes mode={'relaxed' if job['relaxed'] else 'default'} "
                f"restore={SWAP_RESTORE_BACKEND or 'off'} face_upsample={int(CODEFORMER_FACE_UPSAMPLE)} "
                f"identity_blend={SWAP_IDENTITY_BLEND} out={out_w}x{out_h} wait={queue_wait:.2f}s in {elapsed:.2f}s",
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
                    "worker_kind": "swap",
                },
            )
        finally:
            try:
                job_queue.task_done()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3724)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("-f", "--features", default=FEATURES_FILE)
    args = parser.parse_args()

    print(f"Loading features from {args.features}...")
    bundle = load_feature_bundle(args.features)
    names = bundle["names"]
    projects = bundle["projects"]
    groups = bundle["groups"]
    feature_db = bundle["feature_db"]
    feature_norms = bundle["feature_norms"]
    print(f"Loaded {len(names)} people, feature dim: {feature_db.shape[1]}")
    print(
        "[server] swap config "
        f"restore={SWAP_RESTORE_BACKEND or 'off'} "
        f"codeformer_dir={'set' if CODEFORMER_DIR else 'unset'} "
        f"codeformer_weight={CODEFORMER_WEIGHT} "
        f"face_upsample={int(CODEFORMER_FACE_UPSAMPLE)} "
        f"identity_blend={SWAP_IDENTITY_BLEND} "
        f"max_swap_dim={MAX_SWAP_IMAGE_DIM}",
        flush=True,
    )

    print(
        "[server] InsightFace loads in recognition/swap workers; sync path loads lazily.",
        flush=True,
    )
    FaceHandler.insightface_app = None
    FaceHandler.names = names
    FaceHandler.projects = projects
    FaceHandler.groups = groups
    FaceHandler.feature_db = feature_db
    FaceHandler.feature_norms = feature_norms

    recognition_worker = mp.Process(
        target=recognition_job_worker,
        args=(args.features, recognition_job_queue),
        daemon=True,
    )
    recognition_worker.start()
    swap_worker = mp.Process(
        target=swap_job_worker,
        args=(args.features, swap_job_queue),
        daemon=True,
    )
    swap_worker.start()

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
        recognition_worker.terminate()
        swap_worker.terminate()
        recognition_worker.join(timeout=3)
        swap_worker.join(timeout=3)


if __name__ == "__main__":
    main()
