import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
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


def request_once(url, image_bytes, timeout):
    started = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=image_bytes,
        method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            elapsed = time.perf_counter() - started
            data = json.loads(body) if body else {}
            return {
                "ok": True,
                "status": resp.status,
                "elapsed": elapsed,
                "faces": len(data.get("faces", [])),
                "queue_wait": data.get("queue_wait", 0),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        elapsed = time.perf_counter() - started
        error = body
        try:
            error = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            pass
        return {
            "ok": False,
            "status": exc.code,
            "elapsed": elapsed,
            "faces": 0,
            "queue_wait": 0,
            "error": error,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "status": 0,
            "elapsed": elapsed,
            "faces": 0,
            "queue_wait": 0,
            "error": str(exc),
        }


def summarize(results):
    elapsed = [item["elapsed"] for item in results]
    ok = [item for item in results if item["ok"]]
    failed = [item for item in results if not item["ok"]]
    waits = [float(item["queue_wait"] or 0) for item in ok]

    print()
    print("== Summary ==")
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
    if waits:
        print(
            "Queue wait: "
            f"avg={statistics.mean(waits):.3f}s "
            f"p95={percentile(waits, 95):.3f}s "
            f"max={max(waits):.3f}s"
        )
    if failed:
        errors = {}
        for item in failed:
            key = f"{item['status']} {item['error']}"
            errors[key] = errors.get(key, 0) + 1
        print("Failures:")
        for error, count in sorted(errors.items(), key=lambda kv: kv[1], reverse=True):
            print(f"- {count}x {error}")


def main():
    parser = argparse.ArgumentParser(description="HTTP load test for SeiyuuMatch.")
    parser.add_argument("--url", default="http://127.0.0.1:8080/")
    parser.add_argument("--image", help="Image file to upload.")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--groups", default="bangdream:mygo,bangdream:avemujica")
    parser.add_argument("--relaxed", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else find_sample_image()
    if not image_path or not image_path.exists():
        raise SystemExit("No sample image found. Pass --image path/to/photo.jpg")

    params = {"groups": args.groups}
    if args.relaxed:
        params["mode"] = "relaxed"
    url = args.url
    separator = "&" if "?" in url else "?"
    url = url + separator + urllib.parse.urlencode(params)

    image_bytes = image_path.read_bytes()
    print("== Load Test ==")
    print(f"URL: {url}")
    print(f"Image: {image_path} ({len(image_bytes)} bytes)")
    print(f"Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print()

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, url, image_bytes, args.timeout)
            for _ in range(args.requests)
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else "ERR"
            print(
                f"{index:03d}/{args.requests} {status} "
                f"status={result['status']} "
                f"time={result['elapsed']:.3f}s "
                f"wait={float(result['queue_wait'] or 0):.3f}s "
                f"faces={result['faces']} "
                f"{result['error']}"
            )
    total_elapsed = time.perf_counter() - started
    summarize(results)
    print(f"Wall time: {total_elapsed:.3f}s")
    print(f"Throughput: {len(results) / total_elapsed:.2f} req/s")


if __name__ == "__main__":
    main()
