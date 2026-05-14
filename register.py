import os
import sys
import glob
import argparse
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
    m.thresholds = [0.4, 0.5, 0.7]
    return m


def adaface_infer(model, face_aligned):
    bgr = ((face_aligned[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(np.array([bgr.transpose(2, 0, 1)])).float()
    with torch.no_grad():
        feature, _ = model(tensor)
    return feature[0].numpy()


def collect_person_vectors(mtcnn, adaface, person_dir):
    vecs = []
    for pp in sorted(glob.glob(os.path.join(person_dir, "*"))):
        if not pp.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(pp).convert("RGB")
        _, faces = mtcnn.align_multi(img)
        if len(faces) == 0:
            print(f"  Warning: no face in {pp}, skipping")
            continue
        if len(faces) > 1:
            print(f"  Warning: {len(faces)} faces in {pp}, skipping")
            continue
        vecs.append(adaface_infer(adaface, np.array(faces[0])))
    return vecs


def register(mtcnn, adaface, faces_dir):
    name_vecs = {}
    name_bands = {}
    for band_dir in sorted(glob.glob(os.path.join(faces_dir, "*"))):
        if not os.path.isdir(band_dir):
            continue
        band = os.path.basename(band_dir)
        for person_dir in sorted(glob.glob(os.path.join(band_dir, "*"))):
            if not os.path.isdir(person_dir):
                continue
            name = os.path.basename(person_dir)
            name_bands[name] = person_bands(name, band)
            name_vecs.setdefault(name, []).extend(
                collect_person_vectors(mtcnn, adaface, person_dir)
            )

    names = []
    bands = []
    feature_vectors = []
    for name in sorted(name_vecs):
        vecs = name_vecs[name]
        mean_vec = np.mean(vecs, axis=0)
        names.append(name)
        bands.append(encode_bands(name_bands.get(name, [])))
        feature_vectors.append(mean_vec)
        print(f"  Registered: {name} [{bands[-1]}] ({len(vecs)} photos)")
    return names, bands, np.array(feature_vectors) if feature_vectors else np.array([])


def register_one(mtcnn, adaface, faces_dir, band, name):
    person_dir = os.path.join(faces_dir, band, name)
    if not os.path.isdir(person_dir):
        for candidate in sorted(glob.glob(os.path.join(faces_dir, "*", name))):
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
    parser.add_argument("--band", help="Only register one band/person pair")
    parser.add_argument("--name", help="Only register one band/person pair")
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

    if args.band or args.name:
        if not args.band or not args.name:
            print("Error: --band and --name must be used together")
            sys.exit(1)
        print(f"Registering one person from {FACES_DIR}/{args.band}/{args.name}...")
        name, band, vector = register_one(mtcnn, adaface, FACES_DIR, args.band, args.name)
        upsert_feature(args.output, name, band, vector)
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
