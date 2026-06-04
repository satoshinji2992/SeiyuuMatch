import argparse
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_sample_image():
    for root in ["tests", "faces"]:
        base = Path(root)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
    return None


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def build_url(base_url, path="", params=None):
    url = base_url.rstrip("/")
    if path:
        url += "/" + path.lstrip("/")
    if params:
        separator = "&" if "?" in url else "?"
        url += separator + urllib.parse.urlencode(params)
    return url


def read_json_response(resp):
    body = resp.read().decode("utf-8", "replace")
    return json.loads(body) if body else {}


def post_bytes(url, body, timeout, content_type="application/octet-stream"):
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, read_json_response(resp)


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, read_json_response(resp)


def get_bytes(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read()


def make_multipart(fields, files):
    boundary = "----seiyuumatch-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, filename, data, content_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def poll_job(base_url, job_id, timeout, poll_interval, max_wait):
    started = time.perf_counter()
    status_url = build_url(base_url, f"/job/{urllib.parse.quote(job_id)}")
    while True:
        _, data = get_json(status_url, timeout)
        status = data.get("status")
        if status == "done":
            return data
        if status == "failed":
            raise RuntimeError(data.get("error") or "job failed")
        if time.perf_counter() - started > max_wait:
            raise TimeoutError(f"job timeout: {job_id}")
        time.sleep(poll_interval)


def sanitize_details_for_swap(details):
    sanitized = []
    for item in details or []:
        if item.get("easter_egg"):
            continue
        sanitized.append(
            {
                "name": item.get("name", ""),
                "avatar_name": item.get("avatar_name") or item.get("name", ""),
                "avatar_project": item.get("avatar_project", ""),
                "avatar_group": item.get("avatar_group", ""),
                "avatar_feature_index": item.get("avatar_feature_index"),
                "bbox": item.get("bbox"),
            }
        )
    return sanitized


def recognize_once(base_url, image_bytes, args):
    params = {"async": "1", "groups": args.groups}
    if args.relaxed:
        params["mode"] = "relaxed"
    url = build_url(base_url, "/", params)
    started = time.perf_counter()
    _, data = post_bytes(url, image_bytes, args.timeout, "image/jpeg")
    if not data.get("job_id"):
        result = data
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "status": 200,
            "elapsed": elapsed,
            "queue_wait": float(result.get("queue_wait") or 0),
            "faces": len(result.get("faces", [])),
            "details": result.get("details", []),
            "error": "",
        }
    job = poll_job(base_url, data["job_id"], args.timeout, args.poll_interval, args.recognition_max_wait)
    elapsed = time.perf_counter() - started
    result = job.get("result") or {}
    return {
        "ok": True,
        "status": 202,
        "elapsed": elapsed,
        "queue_wait": float(job.get("queue_wait") or result.get("queue_wait") or 0),
        "faces": len(result.get("faces", [])),
        "details": result.get("details", []),
        "error": "",
    }


def swap_once(base_url, image_bytes, image_path, details, args):
    params = {"groups": args.groups}
    if args.relaxed:
        params["mode"] = "relaxed"
    fields = {"details": json.dumps(sanitize_details_for_swap(details), ensure_ascii=False)}
    content_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    multipart_body, multipart_type = make_multipart(
        fields,
        [("image", image_path.name, image_bytes, content_type)],
    )
    started = time.perf_counter()
    _, data = post_bytes(
        build_url(base_url, "/swap_faces", params),
        multipart_body,
        args.timeout,
        multipart_type,
    )
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError("swap request did not return job_id")
    job = poll_job(base_url, job_id, args.timeout, args.poll_interval, args.swap_max_wait)
    result = job.get("result") or {}
    image_size = 0
    if args.fetch_swap_result and result.get("image_path"):
        _, image_data = get_bytes(build_url(base_url, result["image_path"]), args.timeout)
        image_size = len(image_data)
    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "status": 202,
        "elapsed": elapsed,
        "queue_wait": float(job.get("queue_wait") or 0),
        "faces": len(result.get("faces", [])),
        "image_size": image_size,
        "error": "",
    }


def run_workflow(index, base_url, image_bytes, image_path, args):
    workflow_started = time.perf_counter()
    result = {
        "index": index,
        "ok": True,
        "recognize": None,
        "swap": None,
        "elapsed": 0.0,
        "error": "",
    }
    try:
        if args.mode in {"recognize", "both"}:
            result["recognize"] = recognize_once(base_url, image_bytes, args)
        if args.mode in {"swap", "both"}:
            details = result["recognize"]["details"] if result["recognize"] else []
            if args.mode == "swap" and args.details_json:
                details = json.loads(Path(args.details_json).read_text(encoding="utf-8"))
            result["swap"] = swap_once(base_url, image_bytes, image_path, details, args)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            error = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            error = body
        result["ok"] = False
        result["error"] = f"HTTP {exc.code}: {error}"
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
    result["elapsed"] = time.perf_counter() - workflow_started
    return result


def summarize_stage(results, stage):
    items = [item[stage] for item in results if item.get(stage)]
    ok = [item for item in items if item["ok"]]
    if not items:
        return
    elapsed = [item["elapsed"] for item in ok]
    waits = [float(item["queue_wait"] or 0) for item in ok]
    print(f"\n== {stage.capitalize()} ==")
    print(f"Success: {len(ok)}/{len(items)}")
    if elapsed:
        print(
            "Latency: "
            f"avg={statistics.mean(elapsed):.3f}s "
            f"median={statistics.median(elapsed):.3f}s "
            f"p95={percentile(elapsed, 95):.3f}s "
            f"max={max(elapsed):.3f}s"
        )
    if waits:
        print(
            "Queue wait: "
            f"avg={statistics.mean(waits):.3f}s "
            f"p95={percentile(waits, 95):.3f}s "
            f"max={max(waits):.3f}s"
        )
    if stage == "swap":
        sizes = [item["image_size"] for item in ok if item.get("image_size")]
        if sizes:
            print(f"Result image: avg={statistics.mean(sizes):.0f} bytes max={max(sizes)} bytes")


def summarize(results):
    ok = [item for item in results if item["ok"]]
    failed = [item for item in results if not item["ok"]]
    elapsed = [item["elapsed"] for item in results]

    print("\n== Workflow ==")
    print(f"Total: {len(results)}")
    print(f"Success: {len(ok)}")
    print(f"Failed: {len(failed)}")
    if elapsed:
        print(
            "Latency: "
            f"avg={statistics.mean(elapsed):.3f}s "
            f"median={statistics.median(elapsed):.3f}s "
            f"p95={percentile(elapsed, 95):.3f}s "
            f"max={max(elapsed):.3f}s"
        )
    summarize_stage(results, "recognize")
    summarize_stage(results, "swap")
    if failed:
        errors = {}
        for item in failed:
            errors[item["error"]] = errors.get(item["error"], 0) + 1
        print("\nFailures:")
        for error, count in sorted(errors.items(), key=lambda kv: kv[1], reverse=True):
            print(f"- {count}x {error}")


def main():
    parser = argparse.ArgumentParser(description="Async recognition/swap load test for SeiyuuMatch.")
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--image", help="Image file to upload.")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--recognition-max-wait", type=float, default=120)
    parser.add_argument("--swap-max-wait", type=float, default=240)
    parser.add_argument("--poll-interval", type=float, default=0.8)
    parser.add_argument("--groups", default="bangdream:mygo,bangdream:avemujica")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--mode", choices=["recognize", "swap", "both"], default="both")
    parser.add_argument(
        "--details-json",
        help="Recognition details JSON for --mode swap. Without it, swap runs without cached details.",
    )
    parser.add_argument(
        "--fetch-swap-result",
        action="store_true",
        help="Download /job_result image after swap to verify result serving.",
    )
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else find_sample_image()
    if not image_path or not image_path.exists():
        raise SystemExit("No sample image found. Pass --image path/to/photo.jpg")

    image_bytes = image_path.read_bytes()
    base_url = args.url.rstrip("/")
    print("== Load Test ==")
    print(f"Base URL: {base_url}")
    print(f"Mode: {args.mode}")
    print(f"Groups: {args.groups}")
    print(f"Image: {image_path} ({len(image_bytes)} bytes)")
    print(f"Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print()

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_workflow, index, base_url, image_bytes, image_path, args)
            for index in range(1, args.requests + 1)
        ]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            results.append(item)
            status = "OK" if item["ok"] else "ERR"
            rec = item.get("recognize") or {}
            swap = item.get("swap") or {}
            print(
                f"{done:03d}/{args.requests} {status} "
                f"workflow={item['elapsed']:.3f}s "
                f"rec={rec.get('elapsed', 0):.3f}s/{rec.get('faces', 0)}f "
                f"swap={swap.get('elapsed', 0):.3f}s/{swap.get('faces', 0)}f "
                f"{item['error']}"
            )
    total_elapsed = time.perf_counter() - started
    summarize(results)
    print(f"\nWall time: {total_elapsed:.3f}s")
    print(f"Throughput: {len(results) / total_elapsed:.2f} workflows/s")


if __name__ == "__main__":
    main()
