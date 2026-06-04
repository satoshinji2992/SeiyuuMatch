import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stdout

import cv2
import numpy as np


_CODEFORMER_WORKERS = {}
_CODEFORMER_WORKER_LOCK = threading.Lock()


def load_inswapper(model_name):
    from insightface import model_zoo

    return model_zoo.get_model(
        model_name,
        providers=["CPUExecutionProvider"],
        download=True,
    )


def swap_faces_insightface(
    insightface_app,
    inswapper,
    names,
    projects,
    groups,
    feature_db,
    feature_norms,
    image_bytes,
    det_score_threshold,
    selected_groups,
    detect_faces,
    recognize,
    debug_dir=None,
    restore_cmd="",
    restore_backend="",
    codeformer_dir="",
    codeformer_weight=0.7,
    codeformer_face_upsample=False,
    codeformer_face_upsample_max_faces=1,
    codeformer_face_upsample_max_pixels=0,
    recognition_details=None,
    swap_image_max_dim=0,
    identity_blend=0.8,
):
    img_raw, detected_faces = detect_faces(
        insightface_app,
        image_bytes,
        det_score_threshold,
        resize=swap_image_max_dim if swap_image_max_dim else False,
    )
    if img_raw is None or not detected_faces:
        return None, []

    recognition = recognition_details or recognize(
        insightface_app,
        names,
        projects,
        groups,
        feature_db,
        feature_norms,
        image_bytes,
        det_score_threshold,
        selected_groups,
    )
    if not recognition:
        return None, []
    if any(
        isinstance(detail, dict) and detail.get("easter_egg") == "big_brother"
        for detail in recognition
    ):
        raise RuntimeError("该识别结果不支持换脸")
    face_pairs = pair_detected_faces_with_recognition(detected_faces, recognition, img_raw.shape)
    if not face_pairs:
        return None, []

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    swap_result = img_raw.copy()
    source_cache = {}
    used_recognition = []
    for face_index, (face, detail) in enumerate(face_pairs, 1):
        avatar_name = detail.get("avatar_name", "")
        feature_index = detail.get("avatar_feature_index")
        if feature_index is None:
            continue
        feature_index = int(feature_index)
        if feature_index < 0 or feature_index >= len(feature_db):
            continue

        cached = source_cache.get(feature_index)
        if cached is None:
            cached = FeatureSourceFace(feature_db[feature_index])
            source_cache[feature_index] = cached
        if cached is None:
            continue
        blended_source = blend_source_face_with_target(cached, face, identity_blend)

        if debug_dir:
            save_swap_debug_images(
                debug_dir,
                face_index,
                avatar_name,
                swap_result,
                face,
                blended_source,
                inswapper,
            )

        swap_result = inswapper.get(swap_result, face, blended_source, paste_back=True)
        used_recognition.append(detail)

    restore_face_upsample = bool(codeformer_face_upsample)
    if restore_face_upsample and codeformer_face_upsample_max_faces > 0:
        if len(used_recognition) > int(codeformer_face_upsample_max_faces):
            restore_face_upsample = False
    if restore_face_upsample and codeformer_face_upsample_max_pixels > 0:
        if swap_result.shape[0] * swap_result.shape[1] > int(codeformer_face_upsample_max_pixels):
            restore_face_upsample = False

    restored = restore_image(
        swap_result,
        restore_cmd=restore_cmd,
        restore_backend=restore_backend,
        codeformer_dir=codeformer_dir,
        codeformer_weight=codeformer_weight,
        codeformer_face_upsample=restore_face_upsample,
    )
    if restored is not None:
        swap_result = restored
    return swap_result, used_recognition


def pair_detected_faces_with_recognition(detected_faces, recognition, image_shape):
    if not detected_faces or not recognition:
        return []

    height, width = image_shape[:2]
    unused = list(detected_faces)
    pairs = []
    for detail in recognition:
        bbox = detail.get("bbox") if isinstance(detail, dict) else None
        if not bbox or len(bbox) != 4:
            break
        try:
            target = (
                ((float(bbox[0]) + float(bbox[2])) * 0.5) * width,
                ((float(bbox[1]) + float(bbox[3])) * 0.5) * height,
            )
        except (TypeError, ValueError):
            break
        best_index = None
        best_distance = None
        for index, face in enumerate(unused):
            face_bbox = getattr(face, "bbox", None)
            if face_bbox is None:
                continue
            center = (
                (float(face_bbox[0]) + float(face_bbox[2])) * 0.5,
                (float(face_bbox[1]) + float(face_bbox[3])) * 0.5,
            )
            distance = (center[0] - target[0]) ** 2 + (center[1] - target[1]) ** 2
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        pairs.append((unused.pop(best_index), detail))

    if pairs:
        return pairs
    return list(zip(detected_faces, recognition))


class FeatureSourceFace:
    def __init__(self, normed_embedding):
        embedding = np.asarray(normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.normed_embedding = embedding


def blend_source_face_with_target(source_face, target_face, source_weight=0.8):
    source_weight = float(source_weight)
    if source_weight <= 0:
        return target_face
    if source_weight >= 1:
        return source_face
    source = np.asarray(source_face.normed_embedding, dtype=np.float32)
    target_embedding = getattr(target_face, "normed_embedding", None)
    if target_embedding is None:
        target_embedding = getattr(target_face, "embedding", None)
    if target_embedding is None:
        return source_face
    target = np.asarray(target_embedding, dtype=np.float32)
    if target.size != source.size:
        return source_face
    mixed = source_weight * source + (1.0 - source_weight) * target
    norm = np.linalg.norm(mixed)
    if norm > 0:
        mixed = mixed / norm
    return FeatureSourceFace(mixed)


def save_swap_debug_images(debug_dir, face_index, avatar_name, target_img, target_face, source_face, inswapper):
    name = safe_path_segment(avatar_name, "face")
    prefix = f"{face_index:02d}-{name}"
    before_crop = crop_face_region(target_img, target_face)
    if before_crop is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}-target-before.png"), before_crop)
    try:
        raw_model_face = unwrap_raw_model_face(
            inswapper.get(target_img.copy(), target_face, source_face, paste_back=False)
        )
        if raw_model_face is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{prefix}-model-raw-128.png"), raw_model_face)
    except Exception as exc:
        with open(os.path.join(debug_dir, f"{prefix}-model-raw-error.txt"), "w", encoding="utf-8") as f:
            f.write(str(exc))


def crop_face_region(image_bgr, face, padding=24):
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    h, w = image_bgr.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2].copy()


def unwrap_raw_model_face(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def safe_path_segment(value, fallback):
    value = (value or "").strip()
    value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    value = value.replace("*", "_").replace("?", "_").replace('"', "_")
    value = value.replace("<", "_").replace(">", "_").replace("|", "_")
    value = " ".join(value.split()).strip(" .")
    return value[:80] or fallback


def restore_image_with_command(image_bgr, command_template):
    with tempfile.TemporaryDirectory(prefix="seiyuumatch-restore-") as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        if not cv2.imwrite(input_path, image_bgr):
            raise RuntimeError("failed to write restore input")

        command = command_template.format(
            input=shlex.quote(input_path),
            output=shlex.quote(output_path),
        )
        subprocess.run(command, shell=True, check=True)

        restored = cv2.imread(output_path)
        if restored is None:
            raise RuntimeError(f"restore command did not write output: {output_path}")
        return restored


def restore_image(
    image_bgr,
    restore_cmd="",
    restore_backend="",
    codeformer_dir="",
    codeformer_weight=0.7,
    codeformer_face_upsample=False,
):
    backend = (restore_backend or "").strip().lower()
    if backend == "codeformer":
        return restore_image_with_codeformer(
            image_bgr,
            codeformer_dir=codeformer_dir,
            fidelity_weight=codeformer_weight,
            face_upsample=codeformer_face_upsample,
        )
    if restore_cmd:
        return restore_image_with_command(image_bgr, restore_cmd)
    return None


def restore_image_with_codeformer(
    image_bgr,
    codeformer_dir="",
    fidelity_weight=0.7,
    face_upsample=False,
):
    with tempfile.TemporaryDirectory(prefix="seiyuumatch-codeformer-") as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.png")
        if not cv2.imwrite(input_path, image_bgr):
            raise RuntimeError("failed to write CodeFormer input")

        worker = get_codeformer_worker(
            codeformer_dir=codeformer_dir,
            fidelity_weight=fidelity_weight,
            face_upsample=face_upsample,
        )
        worker.restore(input_path, output_path)

        restored = cv2.imread(output_path)
        if restored is None:
            raise RuntimeError(f"CodeFormer did not write output: {output_path}")
        return restored


def get_codeformer_worker(codeformer_dir="", fidelity_weight=0.7, face_upsample=False):
    codeformer_dir = codeformer_dir or os.environ.get("CODEFORMER_DIR", "")
    if not codeformer_dir:
        raise RuntimeError("CODEFORMER_DIR is required when using CodeFormer restore")
    key = (os.path.abspath(codeformer_dir), float(fidelity_weight), bool(face_upsample))
    with _CODEFORMER_WORKER_LOCK:
        worker = _CODEFORMER_WORKERS.get(key)
        if worker is None or worker.is_dead():
            worker = CodeFormerWorker(
                codeformer_dir=codeformer_dir,
                fidelity_weight=fidelity_weight,
                face_upsample=face_upsample,
            )
            _CODEFORMER_WORKERS[key] = worker
        return worker


class CodeFormerWorker:
    def __init__(self, codeformer_dir, fidelity_weight=0.7, face_upsample=False):
        self.codeformer_dir = os.path.abspath(codeformer_dir)
        self.fidelity_weight = float(fidelity_weight)
        self.face_upsample = bool(face_upsample)
        self._io_lock = threading.Lock()
        self._proc = None
        self._start()

    def _start(self):
        script_path = os.path.abspath(__file__)
        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--codeformer-worker",
            "--codeformer-dir",
            self.codeformer_dir,
            "--codeformer-weight",
            str(self.fidelity_weight),
        ]
        if self.face_upsample:
            cmd.append("--codeformer-face-upsample")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            self.codeformer_dir
            if not env.get("PYTHONPATH")
            else self.codeformer_dir + os.pathsep + env["PYTHONPATH"]
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            cwd=self.codeformer_dir,
            env=env,
        )
        ready = self._read_response_line()
        if ready.get("status") != "ready":
            raise RuntimeError(f"CodeFormer worker failed to start: {ready}")

    def is_dead(self):
        return self._proc is None or self._proc.poll() is not None

    def restore(self, input_path, output_path):
        with self._io_lock:
            if self.is_dead():
                self._start()
            request = {
                "cmd": "restore",
                "input_path": input_path,
                "output_path": output_path,
            }
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            response = self._read_response_line()
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "CodeFormer worker failed")

    def _read_response_line(self):
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("CodeFormer worker is not running")
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("CodeFormer worker exited unexpectedly")
        try:
            return json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"invalid CodeFormer worker response: {line!r}") from exc


def run_codeformer_worker(args):
    import torch
    from torchvision.transforms.functional import normalize

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils import img2tensor, tensor2img
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.utils.misc import get_device, gpu_is_available
    from basicsr.utils.realesrgan_utils import RealESRGANer
    from basicsr.utils.registry import ARCH_REGISTRY
    from facelib.utils.face_restoration_helper import FaceRestoreHelper

    def emit(payload):
        sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()

    pretrain_model_url = {
        "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
    }

    device = get_device()

    def build_realesrgan():
        use_half = False
        if torch.cuda.is_available():
            no_half_gpu_list = ["1650", "1660"]
            if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
                use_half = True

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path=pretrain_model_url["realesrgan"],
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=use_half,
        )

        if not gpu_is_available():
            print("[codeformer-worker] Running on CPU now!", file=sys.stderr, flush=True)
        return upsampler

    with redirect_stdout(sys.stderr):
        print("[codeformer-worker] loading model...", file=sys.stderr, flush=True)
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        ckpt_path = load_file_from_url(
            url=pretrain_model_url["restoration"],
            model_dir="weights/CodeFormer",
            progress=True,
            file_name=None,
        )
        checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        face_upsampler = None
        if args.codeformer_face_upsample:
            face_upsampler = build_realesrgan()

        face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
    emit({"status": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "stop":
                emit({"ok": True, "stopped": True})
                break

            input_path = request.get("input_path", "")
            output_path = request.get("output_path", "")
            if not input_path or not output_path:
                raise RuntimeError("input_path and output_path are required")

            with redirect_stdout(sys.stderr):
                image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"failed to read CodeFormer input: {input_path}")

                restored = restore_image_with_codeformer_core(
                    image_bgr,
                    net=net,
                    face_helper=face_helper,
                    bg_upsampler=None,
                    face_upsampler=face_upsampler,
                    fidelity_weight=args.codeformer_weight,
                    face_upsample=args.codeformer_face_upsample,
                )
                if restored is None:
                    raise RuntimeError("CodeFormer returned empty result")
                if not cv2.imwrite(output_path, restored):
                    raise RuntimeError(f"failed to write CodeFormer output: {output_path}")
            emit({"ok": True})
        except Exception as exc:
            emit({"ok": False, "error": str(exc)})


def restore_image_with_codeformer_core(
    image_bgr,
    net,
    face_helper,
    bg_upsampler=None,
    face_upsampler=None,
    fidelity_weight=0.7,
    face_upsample=False,
):
    import torch
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize

    image_bgr = image_bgr.copy()
    face_helper.clean_all()
    face_helper.read_image(image_bgr)
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    if num_det_faces == 0:
        return None
    face_helper.align_warp_face()

    for cropped_face in face_helper.cropped_faces:
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(face_helper.device)
        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            print(f"[codeformer-worker] failed inference: {error}", file=sys.stderr, flush=True)
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)

    bg_img = None
    if bg_upsampler is not None:
        bg_img = bg_upsampler.enhance(image_bgr, outscale=2)[0]
    face_helper.get_inverse_affine(None)
    if face_upsample and face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
            face_upsampler=face_upsampler,
        )
    else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
        )
    return restored_img


def load_face_analysis(det_size):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def largest_face(faces):
    return max(
        faces,
        key=lambda f: (
            float(getattr(f, "det_score", 0.0)),
            float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        ),
    )


def run_cli(args):
    os.makedirs(args.output_dir, exist_ok=True)

    target_img = cv2.imread(args.target)
    if target_img is None:
        raise RuntimeError(f"target image not found or unreadable: {args.target}")

    app = load_face_analysis(args.det_size)
    inswapper = load_inswapper(args.model)
    source_face, source_label = load_feature_source_face(
        args.features,
        args.source,
        project=args.source_project,
        group=args.source_group,
    )

    target_faces = app.get(target_img)
    if not target_faces:
        raise RuntimeError("no face detected in target image")

    target_faces = sorted(
        target_faces,
        key=lambda f: (float(f.bbox[1]), float(f.bbox[0])),
    )
    if args.target_index:
        if args.target_index < 1 or args.target_index > len(target_faces):
            raise RuntimeError(f"target-index out of range: 1..{len(target_faces)}")
        selected_faces = [target_faces[args.target_index - 1]]
    else:
        selected_faces = target_faces

    result = target_img.copy()
    for index, target_face in enumerate(selected_faces, 1):
        prefix = f"{index:02d}"
        before_crop = crop_face_region(result, target_face, padding=args.crop_padding)
        if before_crop is not None:
            cv2.imwrite(os.path.join(args.output_dir, f"{prefix}-target-before.png"), before_crop)

        blended_source_face = blend_source_face_with_target(source_face, target_face, args.identity_blend)
        raw_model_face = unwrap_raw_model_face(
            inswapper.get(result.copy(), target_face, blended_source_face, paste_back=False)
        )
        if raw_model_face is not None:
            cv2.imwrite(os.path.join(args.output_dir, f"{prefix}-model-raw-128.png"), raw_model_face)

        result = inswapper.get(result, target_face, blended_source_face, paste_back=True)
        after_crop = crop_face_region(result, target_face, padding=args.crop_padding)
        if after_crop is not None:
            cv2.imwrite(os.path.join(args.output_dir, f"{prefix}-target-after.png"), after_crop)

    restored = restore_image(
        result,
        restore_cmd=args.restore_cmd,
        restore_backend=args.restore_backend,
        codeformer_dir=args.codeformer_dir,
        codeformer_weight=args.codeformer_weight,
        codeformer_face_upsample=args.codeformer_face_upsample,
    )
    if restored is not None:
        result = restored
        restored_path = os.path.join(args.output_dir, "swap-result-restored.png")
        cv2.imwrite(restored_path, result)

    result_path = os.path.join(args.output_dir, "swap-result.png")
    ok = cv2.imwrite(result_path, result)
    if not ok:
        raise RuntimeError("failed to write swap result")

    print(f"target faces: {len(target_faces)}")
    print(f"source feature: {source_label}")
    print(f"identity blend: {args.identity_blend}")
    print(f"output: {result_path}")
    print(f"raw model face(s): {args.output_dir}/*-model-raw-128.png")


def load_feature_source_face(features_path, source_name, project="", group=""):
    data = np.load(features_path, allow_pickle=True)
    required = {"names", "projects", "groups", "features"}
    if not required.issubset(set(data.files)):
        raise RuntimeError("features file uses the old schema; run register.py first")

    names = [str(v) for v in data["names"]]
    projects = [str(v) for v in data["projects"]]
    groups = [str(v) for v in data["groups"]]
    features = data["features"]

    matches = []
    for idx, (name, proj, group_value) in enumerate(zip(names, projects, groups)):
        group_set = {g.strip() for g in group_value.split(",") if g.strip()}
        if name != source_name:
            continue
        if project and proj != project:
            continue
        if group and group not in group_set:
            continue
        matches.append(idx)

    if not matches:
        raise RuntimeError(f"source feature not found: {source_name}")
    if len(matches) > 1 and not (project and group):
        choices = [
            f"{names[idx]} [{projects[idx]}/{groups[idx]}]"
            for idx in matches
        ]
        raise RuntimeError(
            "source name is ambiguous; pass --source-project and --source-group. "
            + "matches: "
            + "; ".join(choices)
        )

    idx = matches[0]
    label = f"{names[idx]} [{projects[idx]}/{groups[idx]}]"
    return FeatureSourceFace(features[idx]), label


def parse_args():
    parser = argparse.ArgumentParser(description="Run a standalone InsightFace inswapper test.")
    parser.add_argument("target", nargs="?", help="Target image path; faces in this image will be replaced.")
    parser.add_argument("source", nargs="?", help="Source person name in features.npz.")
    parser.add_argument("-o", "--output-dir", default="swap_test_output")
    parser.add_argument("--features", default="features.npz")
    parser.add_argument("--source-project", default="")
    parser.add_argument("--source-group", default="")
    parser.add_argument("--model", default=os.environ.get("INSWAPPER_MODEL_NAME", "inswapper_128.onnx"))
    parser.add_argument("--det-size", type=int, default=int(os.environ.get("INSIGHTFACE_DET_SIZE", "640")))
    parser.add_argument("--target-index", type=int, default=0, help="1-based target face index; 0 swaps all.")
    parser.add_argument("--crop-padding", type=int, default=32)
    parser.add_argument(
        "--restore-cmd",
        default=os.environ.get("SWAP_RESTORE_CMD", ""),
        help="Optional face restoration command template. Use {input} and {output}.",
    )
    parser.add_argument(
        "--restore-backend",
        default=os.environ.get("SWAP_RESTORE_BACKEND", ""),
        choices=["", "codeformer"],
    )
    parser.add_argument("--codeformer-dir", default=os.environ.get("CODEFORMER_DIR", ""))
    parser.add_argument(
        "--codeformer-weight",
        type=float,
        default=float(os.environ.get("CODEFORMER_WEIGHT", "0.7")),
    )
    parser.add_argument(
        "--codeformer-face-upsample",
        action="store_true",
        default=os.environ.get("CODEFORMER_FACE_UPSAMPLE", "1") == "1",
    )
    parser.add_argument(
        "--identity-blend",
        type=float,
        default=float(os.environ.get("SWAP_IDENTITY_BLEND", "0.8")),
        help="Blend weight for target identity. 0.8 keeps 80% target identity and 20% target face embedding.",
    )
    parser.add_argument("--codeformer-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.codeformer_worker:
        run_codeformer_worker(args)
        return
    if not args.target or not args.source:
        raise SystemExit("target and source are required")
    run_cli(args)


if __name__ == "__main__":
    main()
