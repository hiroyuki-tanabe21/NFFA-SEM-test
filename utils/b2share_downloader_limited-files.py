import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from urllib3.exceptions import ProtocolError
from http.client import IncompleteRead
from tqdm import tqdm
import yaml
from pathlib import Path

# ───────────── Load config ─────────────
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    sys.exit("config.yaml が見つかりません")
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

BASE_URL = config.get("base_url", "https://b2share.eudat.eu")
RECORD_ID = config["record_id"]
OUT_DIR = Path(config.get("out_dir", "downloads"))
OUT_DIR.mkdir(exist_ok=True)
NUM_THREADS = config.get("num_threads", 2)
CHUNK = config.get("chunk", 512 * 1024)
MAX_RETRIES = config.get("max_retries", 10)
BACKOFF_START = config.get("backoff_start", 10)
ONLY_FILES = set(config.get("only", []))
MIN_EXPECTED_SIZE = 1 * 1024 * 1024
LOG_YAML_PATH = OUT_DIR / "download_log.yaml"
# ──────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("b2share_dl")

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=7, connect=7, read=7, backoff_factor=1,
    status_forcelist=[500, 502, 503, 504], allowed_methods=["HEAD", "GET"]
)))

def build_true_urls(record_id: str) -> Tuple[List[dict], Dict[str, dict]]:
    api_url = f"{BASE_URL}/api/records/{record_id}"
    logger.info("メタデータ取得: %s", api_url)
    r = session.get(api_url, timeout=30)
    r.raise_for_status()
    record = r.json()
    files = record.get("files", record.get("entries", []))
    tar_infos, md5_lookup = [], {}
    for f in files:
        name = f.get("key")
        bucket = f.get("bucket")
        checksum = f.get("checksum")
        if not name or not bucket:
            continue

        if ONLY_FILES and name not in ONLY_FILES:
            continue

        url = f"{BASE_URL}/api/files/{bucket}/{name}"

        if name.endswith(".tar"):
            tar_infos.append({"name": name, "url": url, "size": f.get("size")})

            # checksumが "md5:xxxxx" 形式で含まれている場合にのみ処理
            if checksum and checksum.startswith("md5:"):
                md5_lookup[name] = {
                    "name": name + ".virtual_md5",
                    "url": None,  # 実ファイルなし
                    "md5": checksum.replace("md5:", "")
                }

    return tar_infos, md5_lookup
    
def request_stream(url: str, headers: dict | None = None):
    return session.get(url, headers=headers or {}, stream=True, timeout=60, allow_redirects=True)

def download_file(url: str, local_path: str) -> bool:
    resume_from = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    attempt = 0
    backoff = BACKOFF_START
    while attempt < MAX_RETRIES:
        attempt += 1
        hdr = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            with request_stream(url, hdr) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "").lower()
                clen = int(r.headers.get("Content-Length", "0"))
                if ("tar" not in ctype and "octet" not in ctype) or clen < MIN_EXPECTED_SIZE:
                    raise ValueError(f"Unexpected response: {ctype}, {clen} bytes")
                total = clen + resume_from
                mode = "ab" if resume_from else "wb"
                written = 0

                with open(local_path, mode) as fp, tqdm(
                    total=total,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(local_path),
                    disable=False,
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            fp.write(chunk)
                            bar.update(len(chunk))
                            written += len(chunk)

                actual_size = os.path.getsize(local_path)
                expected_size = resume_from + written
                if actual_size != expected_size:
                    logger.warning("Size mismatch: expected=%d, actual=%d → 削除して再試行", expected_size, actual_size)
                    os.remove(local_path)
                    resume_from = 0
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if written == 0:
                    logger.warning("進捗なし: %s → 削除して再試行", local_path)
                    os.remove(local_path)
                    resume_from = 0
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                logger.info("%s 完了 (attempt %d)", local_path, attempt)
                return True

        except (IncompleteRead, ProtocolError, requests.exceptions.ChunkedEncodingError) as e:
            logger.warning("Read error %s – retry", e)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning("Connection error %s – retry", e)
        except Exception as e:
            logger.warning("Download error (%s – attempt %d/%d): %s", local_path, attempt, MAX_RETRIES, e)

        time.sleep(backoff)
        backoff *= 2
        resume_from = os.path.getsize(local_path) if os.path.exists(local_path) else 0

    logger.error("Failed to download %s after %d attempts", local_path, MAX_RETRIES)
    return False

def verify_md5(path_: str, expected: str) -> Tuple[bool, str]:
    md5 = hashlib.md5()
    with open(path_, "rb") as fr:
        for chunk in iter(lambda: fr.read(CHUNK), b""):
            md5.update(chunk)
    actual = md5.hexdigest().lower()
    return actual == expected.lower(), actual

def task(tar: dict, md5: dict | None):
    name = tar["name"]
    result = {
        "file": name,
        "status": "pending",
        "expected_md5": None,
        "actual_md5": None,
        "verified": None,
    }
    local_path = OUT_DIR / name
    if not download_file(tar["url"], local_path):
        result["status"] = "download_failed"
        return name, result

    if md5:
        try:
            if "md5" in md5:  # ← APIから取得した checksum を使う場合
                expected = md5["md5"]
            else:  # 従来型: .md5ファイルが実体として存在する場合のみURLアクセス
                txt = session.get(md5["url"], timeout=20).text.strip()
                expected = txt.split()[0]
    
            result["expected_md5"] = expected
            ok, actual = verify_md5(local_path, expected)
            result["actual_md5"] = actual
            result["verified"] = ok
            result["status"] = "ok" if ok else "md5_mismatch"
    
        except Exception as e:
            logger.error("MD5取得失敗 %s: %s", name, e)
            result["status"] = "md5_error"

    else:
        logger.warning("MD5 無し: %s", name)
        result["status"] = "no_md5"

    return name, result

def main(rec_id: str = RECORD_ID):
    tar_files, md5_map = build_true_urls(rec_id)

    if not tar_files:
        logger.error("対象 .tar が見つかりません。")
        sys.exit(1)
    logger.info("%d 個の .tar をダウンロード開始 (threads=%d)", len(tar_files), NUM_THREADS)
    results = {}
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exe:
        fut2name = {exe.submit(task, t, md5_map.get(t["name"])): t["name"] for t in tar_files}
        for fut in as_completed(fut2name):
            name = fut2name[fut]
            try:
                _, info = fut.result()
                results[name] = info
            except Exception as exc:
                logger.error("Unhandled exception in %s: %s", name, exc)
                results[name] = {"file": name, "status": "exception", "error": str(exc)}

    yaml.safe_dump(results, LOG_YAML_PATH.open("w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    logger.info("ログ保存完了: %s", LOG_YAML_PATH)

if __name__ == "__main__":
    main()
