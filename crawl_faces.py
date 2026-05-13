import os
import glob
import requests
from PIL import Image
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor, as_completed

BANDS = {
    "ppp": ["愛美", "西本りみ", "大橋彩香", "伊藤彩沙", "大塚紗英"],
    "roselia": ["相羽あいな", "工藤晴香", "遠藤ゆりか", "中島由貴", "桜川めぐ", "志崎樹音", "明坂聡美"],
    "hhw": ["伊藤美来", "田所あずさ", "豊田萌絵", "吉田有里", "黒沢ともよ"],
    "afterglow": ["佐倉綾音", "三澤紗千香", "加藤英美里", "日笠陽子", "金元寿子"],
    "pastel": ["前島亜美", "小澤亜李", "上坂すみれ", "中上育実", "秦佐和子"],
    "ras": ["夏芽", "仓知玲凤", "紡木吏佐", "Raychell", "小原莉子"],
    "morfonica": ["進藤あまね", "西尾夕香", "Ayasa", "直田姫奈", "Mika"],
    "mygo": ["羊宮妃那", "立石凛", "青木陽菜", "小日向美香", "林鼓子"],
    "avemujica": ["佐々木李子", "渡瀬結月", "岡田夢以", "米澤茜", "高尾奏音"],
}
PEOPLE = []
for band, members in BANDS.items():
    for name in members:
        PEOPLE.append((band, name))
NUM_IMAGES = 20
FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
BAIDU_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
BAIDU_KEY = os.environ.get("BAIDU_SEARCH_AUTH", "")


def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or len(r.content) < 20000:
            return False
        from io import BytesIO
        img = Image.open(BytesIO(r.content))
        w, h = img.size
        if w < 100 or h < 100 or w * h < 40000:
            return False
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        pass
    return False


def tavily_search(name):
    if not TAVILY_KEY:
        print("  Tavily skipped: TAVILY_API_KEY is not set")
        return []
    client = TavilyClient(TAVILY_KEY)
    urls = []
    for query in [f"{name} 声優 写真", f"{name} 声優 画像"]:
        try:
            r = client.search(query=query, search_depth="advanced", include_images=True)
            urls.extend(r.get("images", []))
        except Exception as e:
            print(f"  Tavily error: {e}")
    return list(dict.fromkeys(urls))


def baidu_search(name):
    urls = []
    if not BAIDU_KEY:
        print("  Baidu skipped: BAIDU_SEARCH_AUTH is not set")
        return urls
    try:
        r = requests.post(
            BAIDU_URL,
            headers={"Content-Type": "application/json", "Authorization": BAIDU_KEY},
            json={"messages": [{"role": "user", "content": f"{name} 声優 图片 写真"}]},
            timeout=15,
        )
        data = r.json()
        for ref in data.get("references", []):
            u = ref.get("image")
            if u and u.startswith("http"):
                urls.append(u)
    except Exception as e:
        print(f"  Baidu error: {e}")
    return urls


def google_crawl(name, person_dir):
    from icrawler.builtin import GoogleImageCrawler
    downloaded = 0
    for query in [f"{name} 声優 写真", f"{name} seiyuu"]:
        try:
            crawler = GoogleImageCrawler(
                feeder_threads=1, parser_threads=1, downloader_threads=3,
                storage={"root_dir": person_dir},
            )
            before = len(glob.glob(os.path.join(person_dir, "*")))
            crawler.crawl(keyword=query, max_num=15, min_size=(200, 200))
            after = len(glob.glob(os.path.join(person_dir, "*")))
            downloaded += max(0, after - before)
        except Exception as e:
            print(f"  Google(icrawler) error: {e}")
    return downloaded


def rename_sequential(person_dir):
    count = 1
    for f in sorted(glob.glob(os.path.join(person_dir, "*"))):
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        new_ext = ".jpg" if ext == ".webp" else ext
        new_name = os.path.join(person_dir, f"{count}{new_ext}")
        if f != new_name:
            os.rename(f, new_name)
        count += 1


def crawl_person(band, name):
    person_dir = os.path.join(FACES_DIR, band, name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(person_dir, "*")))
    print(f"[{band}/{name}] existing: {existing}, target: +{NUM_IMAGES}")

    urls = []
    urls.extend(tavily_search(name))
    urls.extend(baidu_search(name))
    urls = list(dict.fromkeys(urls))
    google_dl = google_crawl(name, person_dir)
    print(f"  Found {len(urls)} image URLs")

    downloaded = 0
    idx = existing + 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for url in urls:
            ext = ".jpg"
            if ".png" in url:
                ext = ".png"
            save_path = os.path.join(person_dir, f"{idx}{ext}")
            futures[executor.submit(download_image, url, save_path)] = save_path
            idx += 1

        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                path = futures[future]
                if os.path.exists(path):
                    os.remove(path)

    rename_sequential(person_dir)
    total = len(glob.glob(os.path.join(person_dir, "*")))
    print(f"[{band}/{name}] downloaded: {downloaded} (tavily+baidu) + {google_dl} (google), total: {total}")


def main():
    for band, name in PEOPLE:
        crawl_person(band, name)


if __name__ == "__main__":
    main()
