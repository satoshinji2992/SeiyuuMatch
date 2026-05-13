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
            name_bands[name] = band
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
                name_vecs.setdefault(name, []).append(adaface_infer(adaface, np.array(faces[0])))

    names = []
    bands = []
    feature_vectors = []
    for name in sorted(name_vecs):
        vecs = name_vecs[name]
        mean_vec = np.mean(vecs, axis=0)
        names.append(name)
        bands.append(name_bands.get(name, ""))
        feature_vectors.append(mean_vec)
        print(f"  Registered: {name} [{name_bands.get(name, '')}] ({len(vecs)} photos)")
    return names, bands, np.array(feature_vectors) if feature_vectors else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Register faces and save features")
    parser.add_argument("-o", "--output", default=FEATURES_FILE)
    args = parser.parse_args()

    print("Loading MTCNN...")
    mtcnn = load_mtcnn()

    print("Loading AdaFace...")
    adaface = load_adaface()

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
