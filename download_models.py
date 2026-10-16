# -*- coding: utf-8 -*-
"""Iki (veya daha fazla) Hugging Face modelini paralel ve kalici indirir.

Tekrar calistirinca klasoru silmez. Tamamlanan dosyaya dokunmaz.
Yarim kalan *.incomplete dosyasinin disk'e yazilmis kismina Range ile devam eder.

Kullanim:
  python download_models.py
  python download_models.py --dry-run
  python download_models.py --repo org/model-adi
  python download_models.py --config models.json --output-dir D:\\modeller

Yeni model: models.json icine repo_id ekle, ya da --repo ile tek seferlik ver.
Token (gerekirse): HF_TOKEN ortam degiskeni veya --token. Token loglanmaz.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
from collections import deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
try:
    import httpx2
    from filelock import FileLock, Timeout
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.errors import (
        DisabledRepoError,
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_url
except ImportError:
    sys.stderr.write(
        "Eksik paket. Su komutu calistir, sonra scripti yeniden baslat:\n"
        "  python -m pip install huggingface_hub\n"
    )
    raise SystemExit(1)


USER_AGENT = "hf-resume-downloader/1.0"
PART_SUFFIX = ".incomplete"
SIDECAR_SUFFIX = ".incomplete.json"
SYNC_EVERY = 64 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
READ_TIMEOUT = 30.0
STATE_NAME = "_download_state.json"
LOCK_NAME = "_download.lock"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ("Qwen/Qwen3-8B", "meta-models/Muse-Glimmer-30B")

SHUTDOWN = threading.Event()
PRINT_LOCK = threading.Lock()
_SIGNAL_HITS = 0


class DownloadInterrupted(Exception):
    pass


class FatalDownloadError(Exception):
    pass


class RetryableDownloadError(Exception):
    def __init__(self, message: str, retry_after: float | None = None, kind: str = "retry"):
        super().__init__(message)
        self.retry_after = retry_after
        self.kind = kind


@dataclass
class RemoteFile:
    path: str
    size: int | None
    sha256: str | None
    etag: str | None


@dataclass
class ModelJob:
    repo_id: str
    revision: str | None = None
    local_dir: Path | None = None
    allow: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)


@dataclass
class FileResult:
    path: str
    status: str
    bytes_have: int = 0
    size: int | None = None
    detail: str = ""


def repo_label(repo_id: str) -> str:
    if "Qwen" in repo_id:
        return "Qwen 27B"
    if "Muse" in repo_id:
        return "Muse 30B"
    return repo_id.split("/")[-1][:16]


def progress_bar(percent: float, width: int = 22) -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(round(width * percent / 100.0))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def human_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}s {minutes:02d}dk"
    if minutes:
        return f"{minutes}dk {secs:02d}sn"
    return f"{secs}sn"


class LiveBoard:
    def __init__(self) -> None:
        self._data = threading.Lock()
        self._files: dict[tuple[str, str], tuple[int, int]] = {}
        self._order: list[str] = []
        self._samples: dict[str, deque[tuple[float, int]]] = {}
        self._total_samples: deque[tuple[float, int]] = deque(maxlen=60)
        self.tty = False
        self._painted = 0
        self._last = 0.0
        self._cache: list[str] = []
        self._notes: deque[str] = deque()
        self._flags: dict[str, str] = {}
        self._last_gain: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="hf-status", daemon=True)
        self._thread.start()

    def update(self, repo_id: str, path: str, have: int, total: int, force: bool = False) -> None:
        del force
        with self._data:
            if repo_id not in self._order:
                self._order.append(repo_id)
            key = (repo_id, path)
            previous = self._files.get(key)
            have = max(0, have)
            total = max(0, total)
            gained = previous is not None and have > previous[0]
            self._files[key] = (have, total)
            if gained:
                now = time.monotonic()
                self._last_gain[repo_id] = now
                self._flags.pop(repo_id, None)
                self._samples.setdefault(repo_id, deque(maxlen=60)).append((now, self._sum(repo_id)[0]))
                self._total_samples.append((now, self._sum(None)[0]))

    def flag(self, repo_id: str, text: str) -> None:
        with self._data:
            if text:
                self._flags[repo_id] = text
            else:
                self._flags.pop(repo_id, None)

    def note(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with self._data:
            self._notes.append(line)

    def _sum(self, repo_id: str | None) -> tuple[int, int]:
        have = 0
        total = 0
        for (repo, _path), (got, size) in self._files.items():
            if repo_id is not None and repo != repo_id:
                continue
            have += got
            total += size
        return have, total

    def _speed(self, samples: deque[tuple[float, int]], now: float) -> float:
        fresh = [(stamp, amount) for stamp, amount in samples if now - stamp <= 20]
        if len(fresh) < 2:
            return 0.0
        elapsed = fresh[-1][0] - fresh[0][0]
        delta = fresh[-1][1] - fresh[0][1]
        if elapsed < 0.5 or delta <= 0:
            return 0.0
        return delta / elapsed

    def _row(self, name: str, have: int, total: int, speed: float, note: str) -> str:
        percent = (100.0 * have / total) if total else 0.0
        eta = ((total - have) / speed) if speed > 1 else None
        tail = note if note else f"kalan {human_eta(eta)}"
        return (
            f"  {name:<12} {progress_bar(percent)} {percent:5.1f}%  "
            f"{human_bytes(have):>10} / {human_bytes(total):<10}  "
            f"{human_bytes(int(speed)) + '/s':>12}  {tail}"
        )

    def _render(self, now: float) -> list[str]:
        lines = [f"  guncelleme {time.strftime('%H:%M:%S')}"]
        for repo_id in self._order:
            have, total = self._sum(repo_id)
            speed = self._speed(self._samples.get(repo_id, deque()), now)
            note = self._flags.get(repo_id, "")
            if not note:
                last = self._last_gain.get(repo_id)
                if last is not None and now - last >= 20 and have < total:
                    note = f"sessiz {int(now - last)}sn"
            lines.append(self._row(repo_label(repo_id), have, total, speed, note))
        have, total = self._sum(None)
        speed = self._speed(self._total_samples, now)
        stalled = "yeniden baglaniyor" if self._flags else ""
        lines.append(self._row("TOPLAM", have, total, speed, stalled))
        return lines

    def _loop(self) -> None:
        while not self._stop.wait(2):
            with self._data:
                notes = list(self._notes)
                self._notes.clear()
                lines = self._render(time.monotonic()) if self._files else []
            for line in notes:
                self._write(line + "\n")
            self._paint(lines, True)
            self._write_status(lines)

    def _write_status(self, lines: list[str]) -> None:
        if not lines:
            return
        path = SCRIPT_DIR / "download_status.txt"
        tmp = path.with_name("download_status.txt.tmp")
        try:
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            return

    def _erase(self) -> None:
        if not self.tty or not self._painted:
            return
        sys.stdout.write(f"\033[{self._painted}F")
        for _ in range(self._painted):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{self._painted}F")
        self._painted = 0

    def _write(self, text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
            sys.stdout.buffer.flush()
        except OSError:
            return

    def _paint(self, lines: list[str], force: bool, locked: bool = False) -> None:
        if not lines:
            return

        def paint() -> None:
            now = time.monotonic()
            if not force and now - self._last < (0.4 if self.tty else 8.0):
                return
            self._last = now
            if self.tty and self._painted:
                sys.stdout.write(f"\033[{self._painted}F")
            text = ""
            for line in lines:
                text += ("\033[2K" + line + "\n") if self.tty else (line + "\n")
            if not self.tty:
                text += "\n"
            self._write(text)
            self._painted = len(lines) if self.tty else 0

        if locked:
            paint()
        else:
            with PRINT_LOCK:
                paint()


BOARD = LiveBoard()


def log(message: str) -> None:
    BOARD.note(message)


def human_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{value} B"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    BOARD.tty = enable_vt()


def enable_vt() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, AttributeError):
        return False


def on_signal(signum: int, _frame: Any) -> None:
    global _SIGNAL_HITS
    _SIGNAL_HITS += 1
    SHUTDOWN.set()
    raise KeyboardInterrupt


def install_signals() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, on_signal)
        except (OSError, ValueError):
            pass


def check_shutdown() -> None:
    if SHUTDOWN.is_set():
        raise DownloadInterrupted()


def sleep_interruptible(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        check_shutdown()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.4, remaining))


def backoff_seconds(attempt: int, retry_after: float | None) -> float:
    growth = min(120.0, float(2 ** min(attempt, 7)))
    wait = growth + random.uniform(0.0, 1.5)
    if retry_after is not None:
        wait = max(wait, min(retry_after, 300.0))
    return wait


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log(f"Okunamadi, agirlik dosyalarina dokunulmadi: {path} ({exc})")
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        kept = quarantine(path, "bozuk json")
        log(f"Bozuk durum dosyasi silinmedi, yana alindi: {kept}")
        return None
    return value if isinstance(value, dict) else None


def quarantine(path: Path, reason: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.kept-{stamp}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.kept-{stamp}-{counter}")
    os.replace(path, target)
    log(f"Dosya silinmedi, yana alindi ({reason}): {target.name}")
    return target


def safe_dest(local_dir: Path, relative: str) -> Path:
    pure = Path(relative.replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise FatalDownloadError(f"Guvenli olmayan dosya yolu: {relative}")
    root = local_dir.resolve()
    dest = (local_dir / pure).resolve()
    if dest != root and root not in dest.parents:
        raise FatalDownloadError(f"Hedef klasorun disina cikiyor: {relative}")
    return dest


def pattern_ok(relative: str, allow: list[str], ignore: list[str]) -> bool:
    name = Path(relative).name
    if ignore and any(fnmatch(relative, pat) or fnmatch(name, pat) for pat in ignore):
        return False
    if not allow:
        return True
    return any(fnmatch(relative, pat) or fnmatch(name, pat) for pat in allow)


def validate_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    parts = repo_id.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(part in (".", "..") for part in parts)
        or "\\" in repo_id
        or ".." in repo_id
    ):
        raise FatalDownloadError(
            f"repo_id 'org/model' biciminde olmali: {repo_id!r}"
        )
    return repo_id


def load_models(config: dict[str, Any], config_dir: Path) -> list[ModelJob]:
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise FatalDownloadError("Config icinde en az bir model olmali.")
    output_dir = Path(str(config.get("output_dir") or "models"))
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    jobs: list[ModelJob] = []
    seen: set[str] = set()
    for item in raw_models:
        if isinstance(item, str):
            item = {"repo_id": item}
        if not isinstance(item, dict) or "repo_id" not in item:
            raise FatalDownloadError(f"Gecersiz model satiri: {item!r}")
        repo_id = validate_repo_id(str(item["repo_id"]))
        if repo_id in seen:
            continue
        seen.add(repo_id)
        local_dir = item.get("local_dir")
        if local_dir:
            dest = Path(str(local_dir))
            if not dest.is_absolute():
                dest = (config_dir / dest).resolve()
        else:
            dest = output_dir / repo_id
        revision = item.get("revision") or None
        allow = [str(x) for x in item.get("allow") or []]
        ignore = [str(x) for x in item.get("ignore") or []]
        jobs.append(
            ModelJob(
                repo_id=repo_id,
                revision=str(revision) if revision else None,
                local_dir=dest,
                allow=allow,
                ignore=ignore,
            )
        )
    return jobs


def default_config() -> dict[str, Any]:
    return {
        "output_dir": "models",
        "model_workers": 2,
        "file_workers": 2,
        "models": [{"repo_id": repo_id} for repo_id in DEFAULT_MODELS],
    }


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    if path is None:
        candidate = SCRIPT_DIR / "models.json"
        path = candidate if candidate.is_file() else None
    if path is None:
        return default_config(), SCRIPT_DIR
    payload = read_json(path)
    if payload is None:
        raise FatalDownloadError(f"Config okunamadi: {path}")
    return payload, path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hugging Face modellerini paralel, silmeden ve kaldigi yerden indirir."
    )
    parser.add_argument("--config", type=Path, default=None, help="models.json yolu")
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Bu calistirmada yalnizca bu repolar. Tekrarlanabilir. Dosyalari silmez.",
    )
    parser.add_argument("--revision", default=None, help="Tum repolar icin revision")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-workers", type=int, default=None)
    parser.add_argument("--file-workers", type=int, default=None)
    parser.add_argument("--allow", action="append", default=[], help="Glob. Tekrarlanabilir.")
    parser.add_argument("--ignore", action="append", default=[], help="Glob. Tekrarlanabilir.")
    parser.add_argument("--token", default=None, help="HF token. Loglanmaz.")
    parser.add_argument("--dry-run", action="store_true", help="Indirmez, sadece listeler.")
    return parser.parse_args()


def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def status_code_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def retry_after_of(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def classify(exc: BaseException) -> str:
    if isinstance(exc, (DownloadInterrupted, KeyboardInterrupt, SystemExit)):
        return "interrupt"
    if isinstance(exc, FatalDownloadError):
        return "fatal"
    if isinstance(exc, RetryableDownloadError):
        return "retry"
    if isinstance(
        exc,
        (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
            EntryNotFoundError,
            DisabledRepoError,
            OfflineModeIsEnabled,
        ),
    ):
        return "fatal"
    if isinstance(exc, LocalEntryNotFoundError):
        return "retry"
    code = status_code_of(exc)
    if code in (401, 403, 404, 410, 451):
        return "fatal"
    if code in (408, 409, 425, 429) or (code is not None and 500 <= code <= 599):
        return "retry"
    if isinstance(exc, HfHubHTTPError):
        return "retry" if code is None or code >= 500 else "fatal"
    winerror = getattr(exc, "winerror", None)
    errno = getattr(exc, "errno", None)
    if winerror in (32, 33, 112) or errno in (28, 11, 122):
        return "retry"
    if isinstance(
        exc,
        (
            httpx2.TimeoutException,
            httpx2.NetworkError,
            httpx2.RemoteProtocolError,
            httpx2.ProtocolError,
            httpx2.ProxyError,
            ConnectionError,
            TimeoutError,
            InterruptedError,
        ),
    ):
        return "retry"
    if isinstance(exc, OSError) and errno in (4, 13, 16, 32, 104, 110, 10053, 10054, 10060):
        return "retry"
    if isinstance(exc, OSError) and winerror in (5, 32, 33, 64, 121, 1231, 1232):
        return "retry"
    return "fatal"


def http_timeout() -> httpx2.Timeout:
    return httpx2.Timeout(connect=30.0, read=READ_TIMEOUT, write=30.0, pool=30.0)


def parse_content_range(value: str | None) -> tuple[int | None, int | None, int | None]:
    if not value:
        return None, None, None
    text = value.strip()
    if not text.lower().startswith("bytes"):
        return None, None, None
    _, _, rest = text.partition(" ")
    range_part, _, total_part = rest.partition("/")
    total = int(total_part) if total_part.isdigit() else None
    if range_part.strip() == "*":
        return None, None, total
    start_text, sep, end_text = range_part.partition("-")
    if not sep or not start_text.isdigit():
        return None, None, total
    end = int(end_text) if end_text.isdigit() else None
    return int(start_text), end, total


def fetch_metadata(repo_id: str, relative: str, revision: str | None, token: str | None) -> Any:
    url = hf_hub_url(repo_id, relative, revision=revision)
    return get_hf_file_metadata(url, token=token, timeout=60, retry_on_errors=True)


def list_remote_files(job: ModelJob, token: str | None) -> list[RemoteFile]:
    api = HfApi(token=token)
    entries = api.list_repo_tree(
        job.repo_id,
        recursive=True,
        expand=True,
        revision=job.revision,
    )
    files: list[RemoteFile] = []
    for entry in entries:
        check_shutdown()
        if not isinstance(entry, RepoFile) or not entry.path:
            continue
        lfs = getattr(entry, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        lfs_size = getattr(lfs, "size", None) if lfs is not None else None
        chosen = lfs_size if lfs_size is not None else entry.size
        files.append(
            RemoteFile(
                path=str(entry.path),
                size=int(chosen) if chosen is not None else None,
                sha256=str(sha) if sha else None,
                etag=str(sha) if sha else getattr(entry, "blob_id", None),
            )
        )
    files.sort(key=lambda item: (item.size or 0, item.path))
    return files


def sidecar_path(dest: Path) -> Path:
    return dest.with_name(dest.name + SIDECAR_SUFFIX)


def part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + PART_SUFFIX)


def load_sidecar(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload or {}


def save_sidecar(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def durable_bytes(part: Path, _sidecar: dict[str, Any]) -> int:
    # Dosya boyu esas. Terminal kill sonrasi sidecar geride kalsa bile
    # inmis baytlar kesilmez; Range o boyuttan devam eder.
    return part.stat().st_size if part.exists() else 0


def prepare_part(
    dest: Path,
    expected: int | None,
    etag: str | None,
    sha256: str | None,
    repo_id: str,
    relative: str,
) -> tuple[Path, Path, int]:
    part = part_path(dest)
    side = sidecar_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = load_sidecar(side)
    if part.exists() and meta.get("etag") and etag and meta.get("etag") != etag:
        quarantine(part, "uzak dosyanin etiketi degismis")
        if side.exists():
            quarantine(side, "eski etiket")
        meta = {}
    if dest.exists() and expected is not None and dest.stat().st_size != expected:
        quarantine(
            dest,
            "boyut uzak dosyayla uyusmuyor; yarim kalan *.incomplete ayrica duruyor",
        )
    if part.exists() and expected is not None and part.stat().st_size > expected:
        quarantine(part, "parca uzak boyuttan buyuk")
        meta = {}
    if not part.exists():
        part.touch()
    payload = {
        "repo_id": repo_id,
        "path": relative,
        "etag": etag,
        "sha256": sha256,
        "size": expected,
        "durable_bytes": meta.get("durable_bytes", 0) if meta.get("etag") in (None, etag) else 0,
    }
    if payload["durable_bytes"] is None:
        payload["durable_bytes"] = 0
    save_sidecar(side, payload)
    have = durable_bytes(part, payload)
    payload["durable_bytes"] = have
    save_sidecar(side, payload)
    return part, side, have


def mark_durable(side: Path, payload: dict[str, Any], have: int) -> None:
    payload["durable_bytes"] = have
    save_sidecar(side, payload)


def promote(part: Path, dest: Path, side: Path, expected: int | None) -> None:
    current = part.stat().st_size
    if expected is not None and current != expected:
        raise RetryableDownloadError(
            f"Boyut henuz tutmadi: {human_bytes(current)} / {human_bytes(expected)}"
        )
    os.replace(part, dest)
    try:
        side.unlink(missing_ok=True)
    except OSError:
        pass


def stream_range(
    url: str,
    part: Path,
    side: Path,
    payload: dict[str, Any],
    have: int,
    expected: int | None,
    label: str,
    repo_id: str,
    relative: str,
) -> str:
    headers = {"Accept-Encoding": "identity", "User-Agent": USER_AGENT}
    if have > 0:
        headers["Range"] = f"bytes={have}-"
    timeout = http_timeout()
    with httpx2.Client(timeout=timeout, follow_redirects=True, trust_env=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            code = response.status_code
            if code == 416:
                raise RetryableDownloadError("Sunucu Range istegini reddetti", kind="range-rejected")
            if code == 429 or code in (408, 425) or code >= 500:
                raise RetryableDownloadError(
                    f"HTTP {code}",
                    retry_after_header(response.headers.get("Retry-After")),
                    kind="http",
                )
            if code in (401, 403):
                raise RetryableDownloadError(
                    f"CDN HTTP {code}; adres yenilenecek", kind="cdn"
                )
            if code not in (200, 206):
                raise FatalDownloadError(f"HTTP {code} beklenmiyordu: {label}")
            if have > 0 and code == 200:
                raise RetryableDownloadError(
                    "Sunucu Range basligini yok saydi", kind="range-ignored"
                )
            if have > 0 and code == 206:
                start, _end, total = parse_content_range(response.headers.get("Content-Range"))
                if start != have:
                    raise RetryableDownloadError(
                        f"Content-Range baslangici {start}, yerel {have}"
                    )
                if expected is not None and total is not None and total != expected:
                    raise RetryableDownloadError("Uzak boyut indirme sirasinda degisti")
            clock = [time.monotonic()]
            stop_watch = threading.Event()

            def watch() -> None:
                while not stop_watch.wait(5):
                    if time.monotonic() - clock[0] < 45:
                        continue
                    BOARD.flag(repo_id, "takildi, yeniden baglaniyor")
                    try:
                        response.close()
                    except OSError:
                        pass
                    try:
                        client.close()
                    except OSError:
                        pass
                    return

            threading.Thread(target=watch, name="hf-stall", daemon=True).start()
            try:
                with open(part, "r+b", buffering=0) as handle:
                    if have == 0:
                        handle.truncate(0)
                    handle.seek(have)
                    since_sync = 0
                    try:
                        for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                            if SHUTDOWN.is_set():
                                handle.flush()
                                os.fsync(handle.fileno())
                                mark_durable(side, payload, have)
                                return "interrupted"
                            if not chunk:
                                continue
                            clock[0] = time.monotonic()
                            handle.write(chunk)
                            have += len(chunk)
                            since_sync += len(chunk)
                            if since_sync >= SYNC_EVERY:
                                handle.flush()
                                os.fsync(handle.fileno())
                                mark_durable(side, payload, have)
                                since_sync = 0
                            BOARD.update(repo_id, relative, have, expected or 0)
                    finally:
                        try:
                            handle.flush()
                            os.fsync(handle.fileno())
                            mark_durable(side, payload, have)
                        except OSError as exc:
                            log(f"Parca diske yazilirken hata, dosya silinmedi: {exc}")
            finally:
                stop_watch.set()
                BOARD.flag(repo_id, "")
    if expected is None or have == expected:
        return "done"
    if have > (expected or have):
        raise RetryableDownloadError("Indirilen bayt beklenen boyutu asti")
    return "partial"


def retry_after_header(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def location_is_fresh(meta: Any) -> str:
    location = getattr(meta, "location", None)
    if not location:
        raise RetryableDownloadError("Indirme adresi bos dondu")
    return str(location)


def download_one(
    job: ModelJob,
    remote: RemoteFile,
    token: str | None,
    state: dict[str, Any],
    state_lock: threading.Lock,
) -> FileResult:
    assert job.local_dir is not None
    label = f"{job.repo_id} :: {remote.path}"
    try:
        dest = safe_dest(job.local_dir, remote.path)
    except FatalDownloadError as exc:
        log(f"ATLANDI {label}: {exc}")
        return FileResult(remote.path, "failed", detail=str(exc))

    if dest.exists() and dest.is_symlink():
        log(f"Sembolik link atlandi, silinmedi: {label}")
        return FileResult(remote.path, "failed", detail="symlink")

    expected = remote.size
    stored = state.get("files", {}).get(remote.path, {})
    if not isinstance(stored, dict):
        stored = {}
    stored_sha = stored.get("sha256")
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        sha_changed = bool(stored_sha and remote.sha256 and stored_sha != remote.sha256)
        if not sha_changed:
            part = part_path(dest)
            side = sidecar_path(dest)
            side_meta = load_sidecar(side)
            same_part = side_meta.get("etag") in (None, remote.etag) or not remote.etag
            if same_part and part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
                try:
                    side.unlink(missing_ok=True)
                except OSError:
                    pass
            elif part.exists():
                log(f"Tam dosya duruyor, ayri parcaya dokunulmadi: {part.name}")
            remember(state, state_lock, job, remote, "complete", expected)
            log(f"Zaten tamam, dokunulmadi: {label} ({human_bytes(expected)})")
            return FileResult(remote.path, "complete", expected, expected)
        log(f"Boyut ayni, icerik degismis. Eski dosya yenisi bitene kadar duruyor: {label}")

    attempt = 0
    range_ignored = 0
    range_rejected = 0
    cdn_denied = 0
    while not SHUTDOWN.is_set():
        attempt += 1
        try:
            check_shutdown()
            meta = fetch_metadata(job.repo_id, remote.path, job.revision, token)
            expected = getattr(meta, "size", None) or remote.size
            etag = getattr(meta, "etag", None) or remote.etag
            part, side, have = prepare_part(
                dest, expected, etag, remote.sha256, job.repo_id, remote.path
            )
            if expected is not None and have == expected:
                promote(part, dest, side, expected)
                remember(state, state_lock, job, remote, "complete", expected, etag)
                log(f"Parca zaten tamdi, yerine kondu: {label}")
                return FileResult(remote.path, "complete", expected, expected)
            BOARD.update(job.repo_id, remote.path, have, expected or 0, force=True)
            free = shutil.disk_usage(dest.parent).free
            need = max(0, (expected or 0) - have)
            if need and free < need:
                raise RetryableDownloadError(
                    f"Disk dolu. Gereken {human_bytes(need)}, bos {human_bytes(free)}. "
                    "Dosyalar silinmedi.",
                    kind="disk",
                )
            payload = load_sidecar(side)
            outcome = stream_range(
                location_is_fresh(meta),
                part,
                side,
                payload,
                have,
                expected,
                label,
                job.repo_id,
                remote.path,
            )
            if outcome == "interrupted":
                remember(state, state_lock, job, remote, "partial", part.stat().st_size, etag)
                log(f"Durdu, parca duruyor: {label}")
                return FileResult(remote.path, "interrupted", part.stat().st_size, expected)
            if outcome == "done":
                promote(part, dest, side, expected)
                final_size = dest.stat().st_size
                remember(state, state_lock, job, remote, "complete", final_size, etag)
                log(f"Tamam: {label} ({human_bytes(final_size)})")
                return FileResult(remote.path, "complete", final_size, expected)
            log(f"Baglanti erken kapandi, parcadan devam edilecek: {label}")
            range_ignored = 0
            range_rejected = 0
            cdn_denied = 0
            continue
        except DownloadInterrupted:
            log(f"Durdu, parca duruyor: {label}")
            return FileResult(remote.path, "interrupted", detail="interrupted")
        except RetryableDownloadError as exc:
            if exc.kind == "range-ignored":
                range_ignored += 1
                part = part_path(dest)
                if part.exists():
                    quarantine(part, "sunucu devam istegini kabul etmedi")
                if range_ignored >= 3:
                    log(f"HATA {label}: sunucu devam ettirmeyi uc kez reddetti")
                    return FileResult(remote.path, "failed", detail=str(exc))
            elif exc.kind == "range-rejected":
                range_rejected += 1
                if range_rejected >= 5:
                    log(f"HATA {label}: Range istegi reddedildi, parca dosyasi yerinde birakildi")
                    return FileResult(remote.path, "failed", detail=str(exc))
            elif exc.kind == "cdn":
                cdn_denied += 1
                if cdn_denied >= 8:
                    log(f"HATA {label}: indirme adresi tekrar tekrar reddedildi")
                    return FileResult(remote.path, "failed", detail=str(exc))
            else:
                cdn_denied = 0
            wait = backoff_seconds(attempt, exc.retry_after)
            if exc.kind == "disk":
                wait = max(wait, 30.0)
            log(f"Yeniden denenecek ({attempt}) {label}: {exc}  bekleme {wait:.0f}s")
            try:
                sleep_interruptible(wait)
            except DownloadInterrupted:
                return FileResult(remote.path, "interrupted", detail="interrupted")
        except Exception as exc:
            kind = classify(exc)
            if kind == "interrupt":
                log(f"Durdu, parca duruyor: {label}")
                return FileResult(remote.path, "interrupted", detail="interrupted")
            if kind == "fatal":
                log(f"HATA {label}: {exc}")
                return FileResult(remote.path, "failed", detail=str(exc))
            wait = backoff_seconds(attempt, retry_after_of(exc))
            log(f"Yeniden denenecek ({attempt}) {label}: {exc}  bekleme {wait:.0f}s")
            try:
                sleep_interruptible(wait)
            except DownloadInterrupted:
                return FileResult(remote.path, "interrupted", detail="interrupted")
    return FileResult(remote.path, "interrupted", detail="interrupted")


def remember(
    state: dict[str, Any],
    state_lock: threading.Lock,
    job: ModelJob,
    remote: RemoteFile,
    status: str,
    have: int,
    etag: str | None = None,
) -> None:
    assert job.local_dir is not None
    with state_lock:
        files = state.setdefault("files", {})
        files[remote.path] = {
            "status": status,
            "bytes": have,
            "size": remote.size,
            "etag": etag or remote.etag,
            "sha256": remote.sha256,
        }
        state["repo_id"] = job.repo_id
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            atomic_write_json(job.local_dir / STATE_NAME, state)
        except OSError as exc:
            log(f"Durum dosyasi yazilamadi, indirilenler duruyor: {exc}")


def acquire_lock(lock_path: Path, repo_id: str) -> FileLock | None:
    lock = FileLock(str(lock_path))
    announced = False
    while not SHUTDOWN.is_set():
        try:
            lock.acquire(timeout=5)
            return lock
        except Timeout:
            if not announced:
                log(
                    f"{repo_id} baska bir surecte iniyor. "
                    "Kilit acilinca devam edecek; mevcut dosyalar silinmez."
                )
                announced = True
        except OSError as exc:
            log(f"Kilit alinamadi, dosyalar silinmedi: {repo_id} ({exc})")
            sleep_interruptible(5)
    return None


def download_repo(job: ModelJob, token: str | None, file_workers: int, dry_run: bool) -> list[FileResult]:
    assert job.local_dir is not None
    log(f"Model: {job.repo_id} -> {job.local_dir}")
    attempt = 0
    while not SHUTDOWN.is_set():
        attempt += 1
        try:
            files = list_remote_files(job, token)
            break
        except DownloadInterrupted:
            return [FileResult("*", "interrupted")]
        except Exception as exc:
            if classify(exc) == "fatal":
                log(f"HATA {job.repo_id}: {exc}")
                return [FileResult("*", "failed", detail=str(exc))]
            wait = backoff_seconds(attempt, retry_after_of(exc))
            log(f"Dosya listesi alinamadi, tekrar ({attempt}) {job.repo_id}: {exc}")
            try:
                sleep_interruptible(wait)
            except DownloadInterrupted:
                return [FileResult("*", "interrupted")]
    else:
        return [FileResult("*", "interrupted")]

    if not dry_run:
        job.local_dir.mkdir(parents=True, exist_ok=True)
    selected = [item for item in files if pattern_ok(item.path, job.allow, job.ignore)]
    if not selected:
        log(f"Indirilecek dosya yok: {job.repo_id}")
        return []

    remaining = 0
    for item in selected:
        try:
            dest = safe_dest(job.local_dir, item.path)
        except FatalDownloadError as exc:
            log(f"ATLANDI {item.path}: {exc}")
            continue
        dest_size = dest.stat().st_size if dest.is_file() else 0
        part_size = part_path(dest).stat().st_size if part_path(dest).is_file() else 0
        if item.size is not None and dest_size == item.size:
            have = item.size
            left = 0
        elif item.size is None:
            have = max(dest_size, part_size)
            left = 0
        else:
            have = 0
            if 0 < dest_size < item.size:
                have = dest_size
            if 0 < part_size <= item.size:
                have = max(have, part_size)
            left = max(0, item.size - have)
        remaining += left
        if not dry_run and item.size is not None:
            BOARD.update(job.repo_id, item.path, have, item.size)
        if dry_run:
            mark = "tamam" if left == 0 and item.size is not None else "kalan"
            log(
                f"  {mark:5} {human_bytes(item.size):>10}  "
                f"yerelde {human_bytes(have):>10}  {item.path}"
            )
    if dry_run:
        log(f"{job.repo_id} kalan yaklaşık {human_bytes(remaining)}")
        return [FileResult(item.path, "dry-run", size=item.size) for item in selected]

    try:
        free = shutil.disk_usage(job.local_dir).free
    except OSError:
        free = 0
    if remaining and free < remaining:
        log(
            f"UYARI {job.repo_id}: bos alan {human_bytes(free)}, "
            f"kalan indirme {human_bytes(remaining)}. Dosya silinmez; yer acilinca devam eder."
        )

    state_path = job.local_dir / STATE_NAME
    state = read_json(state_path) or {"repo_id": job.repo_id, "files": {}}
    if "files" not in state or not isinstance(state["files"], dict):
        state["files"] = {}
    state_lock = threading.Lock()
    lock = acquire_lock(job.local_dir / LOCK_NAME, job.repo_id)
    if lock is None:
        return [FileResult("*", "interrupted")]
    results: list[FileResult] = []
    futures: list[Any] = []
    try:
        workers = max(1, min(file_workers, len(selected)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hf-file") as pool:
            futures = [
                pool.submit(download_one, job, item, token, state, state_lock)
                for item in selected
            ]
            try:
                for future in as_completed(futures):
                    results.append(future.result())
                    if SHUTDOWN.is_set():
                        break
            except KeyboardInterrupt:
                SHUTDOWN.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        try:
            lock.release()
        except OSError as exc:
            log(f"Kilit birakilirken hata, dosyalar duruyor: {exc}")
    if SHUTDOWN.is_set():
        for future in futures:
            if future.done():
                continue
            results.append(FileResult("?", "interrupted"))
    return results


def apply_cli(jobs: list[ModelJob], args: argparse.Namespace, config_dir: Path) -> list[ModelJob]:
    if args.repo:
        output_dir = args.output_dir or (config_dir / "models")
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
        jobs = []
        seen: set[str] = set()
        for repo_id in args.repo:
            repo_id = validate_repo_id(repo_id)
            if repo_id in seen:
                continue
            seen.add(repo_id)
            jobs.append(ModelJob(repo_id=repo_id, local_dir=output_dir / repo_id))
    if args.output_dir and not args.repo:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
        for job in jobs:
            if job.local_dir is None:
                continue
            job.local_dir = output_dir / job.repo_id
    if args.revision:
        for job in jobs:
            job.revision = args.revision
    if args.allow:
        for job in jobs:
            job.allow = list(args.allow)
    if args.ignore:
        for job in jobs:
            job.ignore = list(args.ignore)
    return jobs


def summarize(repo_id: str, results: list[FileResult]) -> str:
    if not results:
        return "bos"
    counts: dict[str, int] = {}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
    parts = [f"{name}={count}" for name, count in sorted(counts.items())]
    return f"{repo_id}  " + ", ".join(parts)


def main() -> int:
    configure_stdio()
    BOARD.start()
    install_signals()
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        log("HF_HUB_OFFLINE acik. Indirme yapilmaz, yerel dosyalar silinmedi.")
        return 1
    args = parse_args()
    try:
        config, config_dir = load_config(args.config)
        jobs = load_models(config, config_dir)
        jobs = apply_cli(jobs, args, config_dir)
    except FatalDownloadError as exc:
        log(str(exc))
        return 1
    if not jobs:
        log("Model listesi bos.")
        return 1

    model_workers = args.model_workers or int(config.get("model_workers") or 2)
    file_workers = args.file_workers or int(config.get("file_workers") or 2)
    model_workers = max(1, min(int(model_workers), len(jobs), 8))
    file_workers = max(1, min(int(file_workers), 8))
    token = resolve_token(args.token)

    log(
        f"{len(jobs)} model, model isleri={model_workers}, dosya isleri={file_workers}. "
        "Kesilirse ayni komut kaldigi yerden surer."
    )
    if token is None:
        log("HF_TOKEN yok. Acik modeller iner; limit olursa token koy.")

    outcomes: dict[str, list[FileResult]] = {}
    try:
        with ThreadPoolExecutor(max_workers=model_workers, thread_name_prefix="hf-model") as pool:
            future_map = {
                pool.submit(download_repo, job, token, file_workers, args.dry_run): job
                for job in jobs
            }
            try:
                for future in as_completed(future_map):
                    job = future_map[future]
                    try:
                        outcomes[job.repo_id] = future.result()
                    except Exception as exc:
                        if classify(exc) == "interrupt":
                            SHUTDOWN.set()
                            outcomes[job.repo_id] = [FileResult("*", "interrupted", detail=str(exc))]
                        else:
                            log(f"HATA {job.repo_id}: {exc}")
                            outcomes[job.repo_id] = [FileResult("*", "failed", detail=str(exc))]
            except KeyboardInterrupt:
                SHUTDOWN.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        log("Cikis. Yarim dosyalar *.incomplete olarak duruyor. Komutu yeniden calistir.")
        for job in jobs:
            outcomes.setdefault(job.repo_id, [FileResult("*", "interrupted")])
        for repo_id, results in outcomes.items():
            log(summarize(repo_id, results))
        return 130

    interrupted = False
    failed = False
    for repo_id, results in outcomes.items():
        log(summarize(repo_id, results))
        statuses = {item.status for item in results}
        if "interrupted" in statuses:
            interrupted = True
        if "failed" in statuses:
            failed = True
    if interrupted:
        log("Yarim kaldi. Ayni komut, ayni klasor: dosyalar silinmeden devam eder.")
        return 130
    if failed:
        log("Bazi dosyalar inmedi. Duzeltip ayni komutu calistir; inenler tekrar inmez.")
        return 2
    if args.dry_run:
        log("Dry-run bitti. Dosya yazilmadi.")
        return 0
    log("Tum secilen dosyalar tamam.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        SHUTDOWN.set()
        log("Cikis. Parca dosyalar silinmedi.")
        raise SystemExit(130)
