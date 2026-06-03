import os
import sys
import argparse
import gc
import numpy as np
import cv2
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
FEATURES_FILE = os.path.join(BASE_DIR, "features.npz")
MAX_REGISTER_IMAGE_DIM = int(os.environ.get("MAX_REGISTER_IMAGE_DIM", "800"))
DET_SIZE = int(os.environ.get("REGISTER_DET_SIZE", "960"))
HIDDEN_PROJECT = "__hidden__"
HIDDEN_GROUP = "???"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def feature_id(project, group, name):
    return f"{project}/{group}/{name}"


def encode_values(values):
    return ",".join(sorted(dict.fromkeys(v for v in values if v)))


def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
    return app


def resize_for_registration(img):
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest <= MAX_REGISTER_IMAGE_DIM:
        return img
    scale = MAX_REGISTER_IMAGE_DIM / longest
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def collect_person_vectors(app, person_dir):
    vecs = []
    for entry in sorted(os.scandir(person_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue
        path = entry.path
        if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"  Warning: unreadable image {path}, skipping")
                continue
            img = resize_for_registration(img)
            faces = app.get(img)
            if len(faces) == 0:
                print(f"  Warning: no face in {path}, skipping")
                continue
            if len(faces) > 1:
                print(f"  Warning: {len(faces)} faces in {path}, skipping")
                continue
            vecs.append(faces[0].normed_embedding.copy())
            del img, faces
        except Exception as exc:
            print(f"  Warning: failed to process {path}: {exc}, skipping")
        finally:
            gc.collect()
    return vecs


def mean_feature(vecs):
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec


def iter_people(faces_dir, project_filter=None, group_filter=None, name_filter=None):
    for project_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
        if not project_entry.is_dir() or project_entry.name.startswith("."):
            continue

        project_name = project_entry.name
        if project_name == HIDDEN_GROUP:
            if project_filter and project_filter not in {HIDDEN_GROUP, HIDDEN_PROJECT}:
                continue
            for person_entry in sorted(os.scandir(project_entry.path), key=lambda e: e.name):
                if not person_entry.is_dir() or person_entry.name.startswith("."):
                    continue
                if name_filter and person_entry.name != name_filter:
                    continue
                yield HIDDEN_PROJECT, HIDDEN_GROUP, person_entry.name, person_entry.path
            continue

        if project_filter and project_name != project_filter:
            continue
        for group_entry in sorted(os.scandir(project_entry.path), key=lambda e: e.name):
            if not group_entry.is_dir() or group_entry.name.startswith("."):
                continue
            if group_filter and group_entry.name != group_filter:
                continue
            for person_entry in sorted(os.scandir(group_entry.path), key=lambda e: e.name):
                if not person_entry.is_dir() or person_entry.name.startswith("."):
                    continue
                if name_filter and person_entry.name != name_filter:
                    continue
                yield project_name, group_entry.name, person_entry.name, person_entry.path


def register(app, faces_dir, project=None, group=None, name=None):
    entries = []
    for project_name, group_name, person_name, person_dir in iter_people(
        faces_dir, project, group, name
    ):
        vecs = collect_person_vectors(app, person_dir)
        if not vecs:
            print(f"  Warning: no usable faces for {project_name}/{group_name}/{person_name}, skipping")
            continue
        vector = mean_feature(vecs)
        entries.append(
            {
                "id": feature_id(project_name, group_name, person_name),
                "name": person_name,
                "project": project_name,
                "groups": group_name,
                "feature": vector,
                "photos": len(vecs),
            }
        )
        print(
            f"  Registered: {person_name} [{project_name}/{group_name}] "
            f"({len(vecs)} photos)"
        )
    return entries


def load_existing(output, vector_dim):
    if not os.path.exists(output):
        return [], [], [], [], np.empty((0, vector_dim), dtype=np.float32)

    data = np.load(output, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    features = data["features"]
    if "ids" in data and "projects" in data and "groups" in data:
        ids = [str(v) for v in data["ids"]]
        projects = [str(v) for v in data["projects"]]
        groups = [str(v) for v in data["groups"]]
    else:
        print("Error: existing features file uses the old schema; rebuild without --project/--group/--name")
        sys.exit(1)
    return ids, names, projects, groups, features


def save_rows(output, rows):
    if not rows:
        print("Error: no faces registered")
        sys.exit(1)
    np.savez(
        output,
        ids=[row["id"] for row in rows],
        names=[row["name"] for row in rows],
        projects=[row["project"] for row in rows],
        groups=[row["groups"] for row in rows],
        features=np.array([row["feature"] for row in rows]),
    )
    print(f"Saved {len(rows)} feature vectors to {output}")


def upsert_rows(output, rows):
    if not rows:
        print("Error: no faces registered")
        sys.exit(1)

    ids, names, projects, groups, features = load_existing(output, rows[0]["feature"].shape[0])
    for row in rows:
        if row["id"] in ids:
            idx = ids.index(row["id"])
            names[idx] = row["name"]
            projects[idx] = row["project"]
            groups[idx] = row["groups"]
            features[idx] = row["feature"]
            action = "Updated"
        else:
            ids.append(row["id"])
            names.append(row["name"])
            projects.append(row["project"])
            groups.append(row["groups"])
            features = np.vstack([features, row["feature"].reshape(1, -1)])
            action = "Inserted"
        print(f"{action}: {row['id']} [{row['groups']}] -> {output}")

    np.savez(output, ids=ids, names=names, projects=projects, groups=groups, features=features)
    print(f"Saved {len(ids)} feature vectors to {output}")


def iter_groups(faces_dir):
    for project_entry in sorted(os.scandir(faces_dir), key=lambda e: e.name):
        if not project_entry.is_dir() or project_entry.name.startswith("."):
            continue
        if project_entry.name == HIDDEN_GROUP:
            yield HIDDEN_PROJECT, HIDDEN_GROUP
            continue
        for group_entry in sorted(os.scandir(project_entry.path), key=lambda e: e.name):
            if not group_entry.is_dir() or group_entry.name.startswith("."):
                continue
            yield project_entry.name, group_entry.name


def main():
    parser = argparse.ArgumentParser(description="Register faces and save features (InsightFace)")
    parser.add_argument("-o", "--output", default=FEATURES_FILE)
    parser.add_argument("--project", help="Only register one project, e.g. bangdream or lovelive")
    parser.add_argument("--group", help="Only register one group under --project")
    parser.add_argument("--name", help="Only register one person")
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="Register hidden candidates from faces/???",
    )
    parser.add_argument(
        "--by-group",
        action="store_true",
        help="Register group by group to reduce memory usage",
    )
    args = parser.parse_args()

    if args.group and not args.project:
        print("Error: --group must be used with --project")
        sys.exit(1)
    if args.hidden and (args.project or args.group):
        print("Error: --hidden cannot be combined with --project or --group")
        sys.exit(1)
    if args.by_group and (args.group or args.name):
        print("Error: --by-group cannot be combined with --group or --name")
        sys.exit(1)

    print("Loading InsightFace buffalo_l...")
    app = load_insightface()

    if args.by_group:
        first = True
        for project_name, group_name in iter_groups(FACES_DIR):
            if args.project and project_name != args.project:
                continue
            if project_name == HIDDEN_PROJECT:
                continue
            print(f"\nRegistering [{project_name}/{group_name}]...")
            rows = register(app, FACES_DIR, project=project_name, group=group_name)
            if not rows:
                print(f"  No faces in {project_name}/{group_name}, skipping")
                continue
            if first:
                save_rows(args.output, rows)
                first = False
            else:
                upsert_rows(args.output, rows)
            gc.collect()
        if first:
            print("Error: no faces registered")
            sys.exit(1)
        print("\nAll groups registered.")
        return

    project = HIDDEN_PROJECT if args.hidden else args.project
    print("Registering faces...")
    rows = register(app, FACES_DIR, project=project, group=args.group, name=args.name)

    if args.project or args.group or args.name or args.hidden:
        upsert_rows(args.output, rows)
    else:
        save_rows(args.output, rows)
    print(f"Registered: {[row['id'] for row in rows]}")


if __name__ == "__main__":
    main()
