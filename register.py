import os
import sys
import argparse
import gc
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdaFace"))
import net as adaface_net
from face_alignment.mtcnn import MTCNN

FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
FEATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
)
MAX_REGISTER_IMAGE_DIM = int(os.environ.get("MAX_REGISTER_IMAGE_DIM", "1280"))
EXTRA_PERSON_BANDS = {
    "佐々木李子": ["sumimi"],
}


def person_bands(name, primary_band):
    bands = [primary_band]
    bands.extend(EXTRA_PERSON_BANDS.get(name, []))
    return sorted(dict.fromkeys(b for b in bands if b))


def encode_bands(bands):
    return ",".join(bands)


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
    m.thresholds = [0.5, 0.7, 0.7]
    return m


def resize_for_registration(img):
    longest = max(img.size)
    if longest <= MAX_REGISTER_IMAGE_DIM:
        return img

    scale = MAX_REGISTER_IMAGE_DIM / longest
    size = (
        max(1, int(round(img.width * scale))),
        max(1, int(round(img.height * scale))),
    )
    return img.resize(size, Image.Resampling.LANCZOS)


def adaface_infer(model, face_aligned):
    bgr = ((face_aligned[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(np.array([bgr.transpose(2, 0, 1)])).float()
    with torch.inference_mode():
        feature, _ = model(tensor)
    return feature[0].numpy()


def collect_person_vectors(mtcnn, adaface, person_dir):
    vecs = []
    for entry in sorted(os.scandir(person_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue
        pp = entry.path
        if not pp.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        try:
            with Image.open(pp) as source:
                img = resize_for_registration(source.convert("RGB"))
            _, faces = mtcnn.align_multi(img)
            if len(faces) == 0:
                print(f"  Warning: no face in {pp}, skipping")
                continue
            if len(faces) > 1:
                print(f"  Warning: {len(faces)} faces in {pp}, skipping")
                continue
            vecs.append(adaface_infer(adaface, np.array(faces[0])))
        except Exception as exc:
            print(f"  Warning: failed to process {pp}: {exc}, skipping")
        finally:
            gc.collect()
    return vecs


def register(mtcnn, adaface, faces_dir):
    name_vecs = {}
    name_bands = {}
    for band_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
        if not band_entry.is_dir():
            continue
        band = band_entry.name
        for person_entry in sorted(os.scandir(band_entry.path), key=lambda e: e.name):
            if not person_entry.is_dir():
                continue
            name = person_entry.name
            name_bands[name] = person_bands(name, band)
            name_vecs.setdefault(name, []).extend(
                collect_person_vectors(mtcnn, adaface, person_entry.path)
            )

    names = []
    bands = []
    feature_vectors = []
    for name in sorted(name_vecs):
        vecs = name_vecs[name]
        if not vecs:
            print(f"  Warning: no usable faces for {name}, skipping")
            continue
        mean_vec = np.mean(vecs, axis=0)
        names.append(name)
        bands.append(encode_bands(name_bands.get(name, [])))
        feature_vectors.append(mean_vec)
        print(f"  Registered: {name} [{bands[-1]}] ({len(vecs)} photos)")
    return names, bands, np.array(feature_vectors) if feature_vectors else np.array([])


def register_one(mtcnn, adaface, faces_dir, band, name):
    person_dir = os.path.join(faces_dir, band, name)
    if not os.path.isdir(person_dir):
        for band_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
            if not band_entry.is_dir():
                continue
            candidate = os.path.join(band_entry.path, name)
            if os.path.isdir(candidate):
                person_dir = candidate
                break
        else:
            print(f"Error: missing directory {person_dir}")
            sys.exit(1)
    source_band = os.path.basename(os.path.dirname(person_dir))
    vecs = collect_person_vectors(mtcnn, adaface, person_dir)
    if not vecs:
        print(f"Error: no usable faces for {band}/{name}")
        sys.exit(1)
    mean_vec = np.mean(vecs, axis=0)
    bands = person_bands(name, source_band)
    bands.append(band)
    encoded_bands = encode_bands(sorted(dict.fromkeys(bands)))
    print(f"  Registered one: {name} [{encoded_bands}] ({len(vecs)} photos)")
    return name, encoded_bands, mean_vec


def register_band(mtcnn, adaface, faces_dir, band):
    band_dir = os.path.join(faces_dir, band)
    if not os.path.isdir(band_dir):
        print(f"Error: missing directory {band_dir}")
        sys.exit(1)

    registered = []
    for person_entry in sorted(os.scandir(band_dir), key=lambda e: e.name):
        if not person_entry.is_dir():
            continue
        name = person_entry.name
        name, encoded_bands, vector = register_one(mtcnn, adaface, faces_dir, band, name)
        registered.append((name, encoded_bands, vector))

    if not registered:
        print(f"Error: no people found in {band_dir}")
        sys.exit(1)
    return registered


def upsert_feature(output, name, band, vector):
    if os.path.exists(output):
        data = np.load(output, allow_pickle=True)
        names = [str(n) for n in data["names"]]
        features = data["features"]
        if "bands" in data:
            bands = [str(b) for b in data["bands"]]
        else:
            bands = [""] * len(names)
    else:
        names = []
        bands = []
        features = np.empty((0, vector.shape[0]), dtype=vector.dtype)

    if name in names:
        idx = names.index(name)
        features[idx] = vector
        bands[idx] = band
        action = "Updated"
    else:
        names.append(name)
        bands.append(band)
        features = np.vstack([features, vector.reshape(1, -1)])
        action = "Inserted"

    np.savez(output, names=names, bands=bands, features=features)
    print(f"{action}: {name} [{band}] -> {output}")
    print(f"Saved {len(names)} feature vectors to {output}")


def main():
    parser = argparse.ArgumentParser(description="Register faces and save features")
    parser.add_argument("-o", "--output", default=FEATURES_FILE)
    parser.add_argument("--band", help="Only register one band, or one band/person pair")
    parser.add_argument("--name", help="Only register one person with --band")
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    if args.name and not args.band:
        print("Error: --name must be used with --band")
        sys.exit(1)

    if args.band and args.name:
        print(f"Registering one person from {FACES_DIR}/{args.band}/{args.name}...")
        name, band, vector = register_one(mtcnn, adaface, FACES_DIR, args.band, args.name)
        upsert_feature(args.output, name, band, vector)
        return

    if args.band:
        print(f"Registering band from {FACES_DIR}/{args.band}...")
        registered = register_band(mtcnn, adaface, FACES_DIR, args.band)
        for name, band, vector in registered:
            upsert_feature(args.output, name, band, vector)
        print(f"Registered band {args.band}: {[name for name, _, _ in registered]}")
        return

    print(f"Registering faces from {FACES_DIR}...")
    names, bands, feature_db = register(mtcnn, adaface, FACES_DIR)
    if len(names) == 0:
        print("Error: no faces registered")
        sys.exit(1)

    np.savez(args.output, names=names, bands=bands, features=feature_db)
    print(f"Saved {len(names)} feature vectors to {args.output}")
    print(f"Registered: {names}")


if __name__ == "__main__":
    main()
