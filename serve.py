"""One RTX 4090, two GGUF models, one OpenAI endpoint.

OpenCode talks to http://127.0.0.1:8080/v1. The model field picks
qwen3.8-27b or muse-glimmer. The other model is unloaded first.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"
LLAMA_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b11669/llama-b11669-bin-win-cuda-12.4-x64.zip"
)
CUDART_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b11669/cudart-llama-bin-win-cuda-12.4-x64.zip"
)
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8080
UPSTREAM_PORT = 8081
TARGET_CTX = 65536
TARGET_OUTPUT = 8192
LOAD_TIMEOUT_S = 600
SWITCH_MAX_TOKENS = 256
ERROR_TTL_S = 90
WATCH_S = 15
HEALTH_MISSES_BEFORE_RELOAD = 3

MODELS = {
    "qwen3.8-27b": {
        "path": ROOT
        / "models"
        / "ggml-org"
        / "Qwen3.8-27B-GGUF"
        / "Qwen3.8-27B-Q4_K_M.gguf",
        "name": "Qwen3.8 27B Q4",
        "extra": [],
    },
    "muse-glimmer": {
        "path": ROOT
        / "models"
        / "meta-models"
        / "Muse-Glimmer-30B-GGUF"
        / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "name": "Muse Glimmer 30B Q4",
        "extra": [
            "--chat-template-kwargs",
            json.dumps({"reasoning_strength": "low"}),
        ],
    },
}

ATTEMPTS = (
    (TARGET_CTX, "q8_0", 4096, 2048),
    (TARGET_CTX, "q8_0", 2048, 512),
    (49152, "q8_0", 4096, 1024),
    (32768, "q8_0", 4096, 1024),
)


STOP = threading.Event()
DASH = None

PHASE_COLOR = {
    "bosta": "32",
    "istek": "34",
    "yukleniyor": "35",
    "prompt": "33",
    "yaziyor": "36",
    "sesyok": "31",
}
PROMPT_DONE_RE = re.compile(
    r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*?([0-9.]+) tokens per second\)"
)
EVAL_DONE_RE = re.compile(
    r"(?<!prompt )eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \([^)]*?([0-9.]+) tokens per second\)"
)


def enable_vt() -> bool:
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
    except OSError:
        return False


def log(message: str) -> None:
    if DASH is not None:
        DASH.event(message)
        return
    print(message, flush=True)


def client_closed(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
        return True
    winerror = getattr(exc, "winerror", None)
    return isinstance(exc, OSError) and winerror in (10053, 10054, 10060)


_close_lock = threading.Lock()
_last_close = 0.0


def note_client_closed() -> None:
    global _last_close
    now = time.monotonic()
    with _close_lock:
        if now - _last_close < 5:
            return
        _last_close = now
    log("KAPANDI     OpenCode baglantiyi kesti")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((PROXY_HOST, port)) == 0


def wait_port_closed(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_open(port):
            return
        time.sleep(0.3)


def find_server_exe() -> Path | None:
    if not RUNTIME.exists():
        return None
    hits = sorted(RUNTIME.rglob("llama-server.exe"))
    return hits[0] if hits else None


def ensure_binary() -> Path:
    found = find_server_exe()
    if found:
        return found
    RUNTIME.mkdir(parents=True, exist_ok=True)
    zip_path = RUNTIME / "llama-b11669-bin-win-cuda-12.4-x64.zip"
    dest = RUNTIME / "llama-b11669"
    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        log(f"DOWNLOAD {LLAMA_URL}")
        subprocess.run(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--retry",
                "3",
                "-o",
                str(zip_path),
                LLAMA_URL,
            ],
            check=True,
        )
    if dest.exists():
        shutil.rmtree(dest)
    log(f"UNPACK {zip_path.name}")
    shutil.unpack_archive(zip_path, dest)
    found = find_server_exe()
    if not found:
        raise SystemExit("llama-server.exe not found after unpack")
    ensure_cudart(found.parent)
    return found


def ensure_cudart(bin_dir: Path) -> None:
    if (bin_dir / "cudart64_12.dll").exists():
        return
    zip_path = RUNTIME / "cudart-llama-bin-win-cuda-12.4-x64.zip"
    dest = RUNTIME / "cudart"
    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        log(f"DOWNLOAD {CUDART_URL}")
        subprocess.run(
            ["curl.exe", "-L", "--fail", "--retry", "3", "-o", str(zip_path), CUDART_URL],
            check=True,
        )
    if dest.exists():
        shutil.rmtree(dest)
    log(f"UNPACK {zip_path.name}")
    shutil.unpack_archive(zip_path, dest)
    copied = 0
    for dll in dest.rglob("*.dll"):
        shutil.copy2(dll, bin_dir / dll.name)
        copied += 1
    if copied == 0:
        raise SystemExit("CUDA runtime DLLs were not in the cudart package")


def server_help(exe: Path) -> str:
    env = os.environ.copy()
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [str(exe), "--help"],
        cwd=exe.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return (proc.stdout or "") + (proc.stderr or "")


def looks_oom(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "out of memory",
        "cudamalloc",
        "failed to allocate",
        "insufficient memory",
        "alloc failed",
    )
    return any(needle in lowered for needle in needles)


def publish_limits(ctx: int) -> None:
    output = min(TARGET_OUTPUT, max(2048, ctx // 5))
    if not OPENCODE_CONFIG.exists():
        return
    text = OPENCODE_CONFIG.read_text(encoding="utf-8")
    updated = re.sub(r'("context":\s*)\d+', rf"\g<1>{ctx}", text)
    updated = re.sub(r'("output":\s*)\d+', rf"\g<1>{output}", updated)
    if updated != text:
        OPENCODE_CONFIG.write_text(updated, encoding="utf-8")
        log(f"OPENCODE limits now context={ctx} output={output}")


class Dashboard:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vt = False
        self.tty = False
        self.phase = "bosta"
        self.model = "-"
        self.ctx = 0
        self.kv = ""
        self.request_open = False
        self.was_processing = False
        self.prompt_done = 0
        self.prompt_total = 0
        self.gen_done = 0
        self.gen_left = -1
        self.speed = 0.0
        self.prompt_started = 0.0
        self.gen_started = 0.0
        self.last_move = time.monotonic()
        self.load_started = 0.0
        self.last_hist = 0.0
        self.last_prompt_printed = -1
        self.last_gen_printed = -1
        self.stall_sent = False
        self.summary = "henuz istek yok"
        self.started = time.monotonic()
        self.exact_prompt: tuple[float, int, float] | None = None
        self.exact_gen: tuple[float, int, float] | None = None
        self.last_idle = time.monotonic()
        self.last_beat = 0.0

    def start(self) -> None:
        self.vt = enable_vt()
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        threading.Thread(target=self._poll, name="dashboard", daemon=True).start()
        threading.Thread(target=self._keys, name="keys", daemon=True).start()

    def event(self, text: str) -> None:
        with self.lock:
            self._line(text)
            self._paint()

    def mark_loading(self, model_id: str) -> None:
        with self.lock:
            self.phase = "yukleniyor"
            self.model = model_id
            self.load_started = time.monotonic()
            self.last_move = self.load_started
            self.stall_sent = False
            self._paint()

    def mark_ready(self, model_id: str, ctx: int, kv: str) -> None:
        with self.lock:
            self.model = model_id
            self.ctx = ctx
            self.kv = kv
            self.phase = "istek" if self.request_open else "bosta"
            self.last_move = time.monotonic()
            self._paint()

    def request(self, model_id: str, path: str) -> None:
        short = "chat" if "/chat/" in path else path.rsplit("/", 1)[-1]
        with self.lock:
            self.request_open = True
            self.model = model_id
            self.was_processing = False
            self.phase = "istek"
            self.prompt_done = 0
            self.prompt_total = 0
            self.gen_done = 0
            self.gen_left = -1
            self.speed = 0.0
            self.prompt_started = 0.0
            self.gen_started = 0.0
            self.stall_sent = False
            self.last_prompt_printed = -1
            self.last_gen_printed = -1
            self.exact_prompt = None
            self.exact_gen = None
            self.last_move = time.monotonic()
            self._line(f"ISTEK       {model_id}  {short}")
            self._paint()

    def end_request(self) -> None:
        with self.lock:
            self.request_open = False

    def ingest(self, raw: str) -> None:
        prompt = PROMPT_DONE_RE.search(raw)
        gen = EVAL_DONE_RE.search(raw)
        if not prompt and not gen:
            return
        with self.lock:
            if prompt:
                self.exact_prompt = (float(prompt.group(1)), int(prompt.group(2)), float(prompt.group(3)))
            if gen:
                self.exact_gen = (float(gen.group(1)), int(gen.group(2)), float(gen.group(3)))

    def _clear(self) -> str:
        if self.vt:
            return "\r\x1b[2K"
        width = shutil.get_terminal_size((100, 20)).columns
        return "\r" + (" " * max(0, width - 1)) + "\r"

    def _line(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        prefix = self._clear() if self.tty else ""
        sys.stdout.write(prefix + f"{stamp}  {text}\n")
        sys.stdout.flush()

    def _paint(self) -> None:
        now = time.monotonic()
        if self.tty:
            sys.stdout.write(self._clear() + self._status())
            sys.stdout.flush()
            return
        if self.phase == "yukleniyor" and now - self.last_hist >= 2:
            self.last_hist = now
            sec = int(now - self.load_started)
            self._line(f"YUKLENIYOR  {self.model}  {sec}s  model aciliyor, donmedi")
        elif self.phase == "bosta" and now - self.last_idle >= 15:
            self.last_idle = now
            self._line(f"bosta       {self.model}  ctx {self.ctx}  OpenCode bekleniyor")

    def _color(self, phase: str, text: str) -> str:
        if not self.vt:
            return text
        return f"\x1b[{PHASE_COLOR.get(phase, '0')}m{text}\x1b[0m"

    def _bar(self, frac: float) -> str:
        cells = 16
        filled = int(round(max(0.0, min(1.0, frac)) * cells))
        return "[" + ("#" * filled) + ("." * (cells - filled)) + "]"

    def _status(self) -> str:
        stamp = time.strftime("%H:%M:%S")
        spin = "|/-\\"[int(time.monotonic() * 8) % 4]
        phase = self.phase
        stalled = self.stall_sent and phase in ("prompt", "yaziyor", "istek")
        if stalled:
            quiet = int(time.monotonic() - self.last_move)
            body = f"{spin} SES YOK     {quiet}s sayac ilerlemedi"
            phase = "sesyok"
        elif phase == "bosta":
            body = f"bosta       {self.model}  ctx {self.ctx}  bekleniyor   [s] durum  [q] cikis"
        elif phase == "yukleniyor":
            sec = int(time.monotonic() - self.load_started)
            body = f"{spin} YUKLENIYOR  {self.model}  {sec}s  model aciliyor"
        elif phase == "istek":
            sec = int(time.monotonic() - self.last_move)
            body = f"{spin} ISTEK       {self.model}  {sec}s  basliyor"
        elif phase == "prompt":
            total = max(self.prompt_total, self.prompt_done, 1)
            frac = self.prompt_done / total
            body = (
                f"{spin} PROMPT      {self._bar(frac)}  "
                f"{self.prompt_done}/{total}  {int(frac * 100)}%  {self.speed:.0f} t/s"
            )
        elif phase == "yaziyor":
            whole = self.gen_done + self.gen_left if self.gen_left >= 0 else 0
            bar = self._bar(self.gen_done / whole) if whole else "[................]"
            body = f"{spin} YAZIYOR     {bar}  {self.gen_done} tok  {self.speed:.0f} t/s"
        else:
            body = phase
        return self._color(phase, f"{stamp}  {body}")

    def _poll(self) -> None:
        url = f"http://{PROXY_HOST}:{UPSTREAM_PORT}/slots"
        with httpx.Client(timeout=0.8) as client:
            while not STOP.is_set():
                self._sample(client, url)
                if STOP.wait(0.4):
                    break

    def _sample(self, client: httpx.Client, url: str) -> None:
        engine = ENGINE
        if engine is None:
            return
        with self.lock:
            loading = self.phase == "yukleniyor" or engine.model_id is None
        if loading:
            with self.lock:
                self._paint()
            return
        try:
            response = client.get(url)
            if response.status_code != 200:
                raise httpx.HTTPError(str(response.status_code))
            slot = response.json()[0]
        except (httpx.HTTPError, IndexError, ValueError, TypeError):
            with self.lock:
                self._paint()
            return
        self._apply(slot)

    def _apply(self, slot: dict) -> None:
        processing = bool(slot.get("is_processing"))
        prompt_total = int(slot.get("n_prompt_tokens") or 0)
        prompt_done = int(slot.get("n_prompt_tokens_processed") or 0)
        nxt = (slot.get("next_token") or [{}])[0]
        decoded = int(nxt.get("n_decoded") or 0)
        remain = nxt.get("n_remain")
        remain = int(remain) if remain is not None else -1
        now = time.monotonic()
        with self.lock:
            if self.phase == "yukleniyor":
                self._paint()
                return
            if processing and not self.was_processing:
                self.was_processing = True
                self.prompt_started = now
                self.gen_started = 0.0
                self.stall_sent = False
                self.last_move = now
                self.last_prompt_printed = -1
                self.last_gen_printed = -1
            if not processing and self.was_processing:
                self._finish(now)
                self.was_processing = False
                self.phase = "bosta"
                self._paint()
                return
            if not processing:
                self.phase = "istek" if self.request_open else "bosta"
                self._watch_stall(now)
                self._paint()
                return
            moved = (
                prompt_done != self.prompt_done
                or decoded != self.gen_done
                or prompt_total != self.prompt_total
            )
            self.prompt_done = prompt_done
            self.prompt_total = prompt_total
            self.gen_done = decoded
            self.gen_left = remain
            if decoded > 0 and self.gen_started <= 0:
                self.gen_started = now
            prompt_open = decoded == 0 and prompt_done + 8 < max(prompt_total, 1)
            self.phase = "prompt" if prompt_open else "yaziyor"
            if prompt_open:
                elapsed = now - self.prompt_started
                if elapsed >= 0.2 and prompt_done > 0:
                    self.speed = prompt_done / elapsed
            else:
                elapsed = now - self.gen_started if self.gen_started > 0 else 0
                if elapsed >= 0.2 and decoded > 0:
                    self.speed = decoded / elapsed
                else:
                    self.speed = 0
            if moved:
                self.last_move = now
                self.stall_sent = False
                self._history(now)
            else:
                self._watch_stall(now)
                if not self.tty and now - self.last_beat >= 1:
                    if self.phase == "prompt" and self.prompt_done > 0:
                        self.last_beat = now
                        total = max(self.prompt_total, self.prompt_done, 1)
                        self._line(
                            f"PROMPT      processing  {self.prompt_done}/{total}  "
                            f"{int(self.prompt_done * 100 / total)}%  {self.speed:.0f} t/s  calisiyor"
                        )
                    elif self.phase == "yaziyor" and self.gen_done > 0:
                        self.last_beat = now
                        rate = f"  {self.speed:.0f} t/s" if self.speed > 0 else ""
                        self._line(f"YAZIYOR     {self.gen_done} tok{rate}  calisiyor")
            self._paint()

    def _history(self, now: float) -> None:
        if self.phase == "prompt" and self.prompt_done > 0 and self.prompt_done != self.last_prompt_printed:
            total = max(self.prompt_total, self.prompt_done, 1)
            frac = self.prompt_done / total
            self.last_prompt_printed = self.prompt_done
            self._line(
                "PROMPT      processing  "
                f"{self._bar(frac)}  {self.prompt_done}/{total}  "
                f"{int(frac * 100)}%  {self.speed:.0f} t/s"
            )
            return
        if (
            self.phase == "yaziyor"
            and self.gen_done > 0
            and self.gen_done != self.last_gen_printed
            and now - self.last_hist >= 1
        ):
            self.last_hist = now
            self.last_gen_printed = self.gen_done
            rate = f"  {self.speed:.0f} t/s" if self.speed > 0 else ""
            self._line(f"YAZIYOR     {self.model}  {self.gen_done} tok{rate}")

    def _finish(self, now: float) -> None:
        prompt_n = max(self.prompt_total, self.prompt_done)
        gen_n = self.gen_done
        prompt_s = 0.0
        gen_s = 0.0
        prompt_tps = self.speed
        gen_tps = 0.0
        if self.exact_prompt:
            prompt_s = self.exact_prompt[0] / 1000
            prompt_n = self.exact_prompt[1]
            prompt_tps = self.exact_prompt[2]
        elif self.prompt_started > 0:
            end = self.gen_started if self.gen_started > 0 else now
            prompt_s = max(0.0, end - self.prompt_started)
            if prompt_s > 0.05 and prompt_n:
                prompt_tps = prompt_n / prompt_s
        if self.exact_gen:
            gen_s = self.exact_gen[0] / 1000
            gen_n = self.exact_gen[1]
            gen_tps = self.exact_gen[2]
        elif self.gen_started > 0:
            gen_s = max(0.0, now - self.gen_started)
            if gen_s > 0.05 and gen_n:
                gen_tps = gen_n / gen_s
        text = f"BITTI       prompt {prompt_n} tok / {prompt_s:.1f}s ({prompt_tps:.0f} t/s)"
        if gen_n and gen_tps:
            text += f"   cevap {gen_n} tok / {gen_s:.1f}s ({gen_tps:.0f} t/s)"
        elif gen_n:
            text += f"   cevap {gen_n} tok"
        if self.ctx:
            text += f"   ctx {prompt_n}/{self.ctx}"
        self.summary = text
        self.phase = "bosta"
        self.stall_sent = False
        self.last_idle = now
        self._line(text)

    def _watch_stall(self, now: float) -> None:
        if self.phase not in ("prompt", "yaziyor", "istek"):
            return
        quiet = now - self.last_move
        limit = 15 if self.phase == "istek" else 10
        if quiet < limit or self.stall_sent:
            return
        self.stall_sent = True
        self._line(f"SES YOK     {int(quiet)}s sayac ilerlemedi  istek suruyor olabilir")

    def snapshot(self) -> None:
        with self.lock:
            up = int(time.monotonic() - self.started)
            lines = (
                "---- durum ----",
                f"model    {self.model}",
                f"ctx      {self.ctx}   kv {self.kv}",
                f"durum    {self.phase}",
                f"adres    http://{PROXY_HOST}:{PROXY_PORT}/v1",
                f"sure     {up}s",
                f"son      {self.summary}",
                "komut    [s] durum   [h] yardim   [q] cikis",
                "---------------",
            )
            for line in lines:
                self._line(line)
            self._paint()

    def _keys(self) -> None:
        if not self.tty or os.name != "nt":
            return
        try:
            import msvcrt
        except ImportError:
            return
        while not STOP.is_set():
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                ch = msvcrt.getwch()
            except OSError:
                return
            if ch in ("\x00", "\xe0"):
                try:
                    msvcrt.getwch()
                except OSError:
                    return
                continue
            ch = ch.lower()
            if ch == "q":
                self.event("DURDURULUYOR")
                STOP.set()
            elif ch == "s":
                self.snapshot()
            elif ch == "h":
                self.event("komutlar   [s] durum   [h] yardim   [q] cikis")


class Engine:
    def __init__(self, exe: Path, help_text: str) -> None:
        self.exe = exe
        self.help_text = help_text
        self.cv = threading.Condition()
        self.proc: subprocess.Popen[bytes] | None = None
        self.pump: threading.Thread | None = None
        self.log_handle = None
        self.model_id: str | None = None
        self.ctx = 0
        self.kv = ""
        self.busy = 0
        self.fault: str | None = None
        self.fault_at = 0.0
        self.health_misses = 0

    def stop(self) -> None:
        proc = self.proc
        pump = self.pump
        self.proc = None
        self.pump = None
        self.model_id = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
                proc.wait(timeout=15)
        if pump is not None:
            pump.join(timeout=3)
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        wait_port_closed(UPSTREAM_PORT, 30)

    def _command(self, model_id: str, ctx: int, kv: str, batch: int, ubatch: int) -> list[str]:
        spec = MODELS[model_id]
        cmd = [
            str(self.exe),
            "-m",
            str(spec["path"]),
            "-a",
            model_id,
            "--host",
            PROXY_HOST,
            "--port",
            str(UPSTREAM_PORT),
            "-ngl",
            "99",
            "-c",
            str(ctx),
            "-np",
            "1",
            "--jinja",
            "-b",
            str(batch),
            "-ub",
            str(ubatch),
        ]
        if "--cache-type-k" in self.help_text:
            cmd += ["--cache-type-k", kv, "--cache-type-v", kv]
        if "--flash-attn" in self.help_text:
            cmd += ["--flash-attn", "on"]
        cmd += list(spec["extra"])
        return cmd

    def _start_once(self, model_id: str, ctx: int, kv: str, batch: int, ubatch: int) -> bool:
        logs = RUNTIME / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{model_id}-{ctx}-{kv}-b{batch}.log"
        self.log_handle = open(log_path, "wb")
        env = os.environ.copy()
        env["PATH"] = str(self.exe.parent) + os.pathsep + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = self._command(model_id, ctx, kv, batch, ubatch)
        if DASH is not None:
            DASH.mark_loading(model_id)
        log(f"LOAD {model_id} ctx={ctx} kv={kv} batch={batch} ubatch={ubatch}")
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.exe.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.pump = threading.Thread(target=self._pump, name="llama-log", daemon=True)
        self.pump.start()
        if self._wait_ready():
            self.model_id = model_id
            self.ctx = ctx
            self.kv = kv
            log(f"LOADED {model_id} ctx={ctx} kv={kv}")
            if DASH is not None:
                DASH.mark_ready(model_id, ctx, kv)
            return True
        if self.log_handle:
            self.log_handle.flush()
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        oom = looks_oom(tail)
        log(f"LOAD_FAIL {model_id} ctx={ctx} kv={kv} oom={oom}")
        log(tail[-1200:])
        self.stop()
        if not oom:
            raise RuntimeError(tail[-1200:] or "llama-server exited")
        return False

    def _pump(self) -> None:
        proc = self.proc
        handle = self.log_handle
        if proc is None or proc.stdout is None or handle is None:
            return
        try:
            for line in iter(proc.stdout.readline, b""):
                try:
                    handle.write(line)
                    handle.flush()
                except ValueError:
                    break
                if DASH is not None:
                    DASH.ingest(line.decode("utf-8", "replace"))
        finally:
            try:
                handle.flush()
            except ValueError:
                pass

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        url = f"http://{PROXY_HOST}:{UPSTREAM_PORT}/health"
        while time.monotonic() < deadline:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            try:
                response = httpx.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1)
        return False

    def ensure(self, model_id: str) -> None:
        if model_id not in MODELS:
            raise KeyError(model_id)
        path = MODELS[model_id]["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        with self.cv:
            while self.busy and self.model_id not in (None, model_id):
                self.cv.wait()
            if (
                self.model_id == model_id
                and self.proc is not None
                and self.proc.poll() is None
            ):
                self.busy += 1
                return
            if DASH is not None:
                DASH.mark_loading(model_id)
            self.stop()
            for ctx, kv, batch, ubatch in ATTEMPTS:
                if self._start_once(model_id, ctx, kv, batch, ubatch):
                    try:
                        publish_limits(ctx)
                    except OSError as exc:
                        log(f"OPENCODE_WRITE_FAIL {exc}")
                    self.busy += 1
                    return
            raise RuntimeError(f"could not load {model_id} into VRAM")

    def release(self) -> None:
        with self.cv:
            if self.busy:
                self.busy -= 1
            self.cv.notify_all()

    def note_fault(self, message: str) -> None:
        with self.cv:
            self.fault = message[:400]
            self.fault_at = time.monotonic()

    def clear_fault(self) -> None:
        with self.cv:
            self.fault = None
            self.health_misses = 0

    def fresh_fault(self) -> str | None:
        with self.cv:
            if self.fault and time.monotonic() - self.fault_at < ERROR_TTL_S:
                return self.fault
            if self.fault and self.busy == 0:
                self.fault = None
                self.health_misses = 0
            return None

    def _alive_locked(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _expire_fault_locked(self) -> None:
        if (
            self.fault
            and self.busy == 0
            and time.monotonic() - self.fault_at >= ERROR_TTL_S
        ):
            self.fault = None
            self.health_misses = 0
            log("FAULT_CLEARED")

    def _upstream_ok(self) -> bool:
        try:
            response = httpx.get(
                f"http://{PROXY_HOST}:{UPSTREAM_PORT}/health",
                timeout=2,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _reload(self, model_id: str, reason: str) -> None:
        with self.cv:
            if self.busy or self.model_id not in (None, model_id):
                return
            log(f"HEAL {reason} model={model_id}")
            if DASH is not None:
                DASH.mark_loading(model_id)
            self.stop()
            try:
                for ctx, kv, batch, ubatch in ATTEMPTS:
                    if self._start_once(model_id, ctx, kv, batch, ubatch):
                        self.fault = None
                        self.health_misses = 0
                        log(f"HEALED {model_id}")
                        return
                self.fault = f"could not reload {model_id}"
                self.fault_at = time.monotonic()
            except Exception as exc:
                self.fault = str(exc)[:400]
                self.fault_at = time.monotonic()
                log(f"HEAL_FAIL {self.fault}")

    def heal(self) -> None:
        with self.cv:
            self._expire_fault_locked()
            if self.busy or not self.model_id:
                return
            model_id = self.model_id
            alive = self._alive_locked()
            cooling = bool(self.fault) and time.monotonic() - self.fault_at < ERROR_TTL_S
        if not alive:
            if not cooling:
                self._reload(model_id, "process exited")
            return
        if self._upstream_ok():
            self.clear_fault()
            return
        with self.cv:
            if self.busy:
                return
            self.health_misses += 1
            misses = self.health_misses
            cooling = bool(self.fault) and time.monotonic() - self.fault_at < ERROR_TTL_S
        if misses >= HEALTH_MISSES_BEFORE_RELOAD and not cooling:
            self._reload(model_id, "health failed")


ENGINE: Engine | None = None


def resolve_model(payload: dict) -> str:
    assert ENGINE is not None
    requested = payload.get("model") or ENGINE.model_id or "qwen3.8-27b"
    if requested not in MODELS:
        raise KeyError(requested)
    loaded = ENGINE.model_id
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if (
        loaded
        and requested != loaded
        and isinstance(max_tokens, int)
        and max_tokens <= SWITCH_MAX_TOKENS
    ):
        payload["model"] = loaded
        return loaded
    return requested


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if exc is not None and client_closed(exc):
            note_client_closed()
            return
        log(f"REQUEST_FAIL {exc}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120

    def log_message(self, fmt: str, *args) -> None:
        return

    def handle(self) -> None:
        try:
            super().handle()
        except OSError as exc:
            if client_closed(exc):
                note_client_closed()
                return
            raise

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 64 * 1024 * 1024:
            raise ValueError("body too large")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _models(self) -> None:
        data = [
            {"id": model_id, "object": "model", "owned_by": "local"}
            for model_id in MODELS
        ]
        self._send_json(200, {"object": "list", "data": data})

    def _health(self) -> None:
        assert ENGINE is not None
        fault = ENGINE.fresh_fault()
        if ENGINE.model_id and not fault:
            status = "ok"
        elif ENGINE.model_id:
            status = "degraded"
        else:
            status = "loading"
        self._send_json(
            200,
            {
                "status": status,
                "model": ENGINE.model_id,
                "ctx": ENGINE.ctx,
                "kv": ENGINE.kv,
                "fault": fault,
            },
        )

    def _proxy(self, body: bytes) -> None:
        assert ENGINE is not None
        url = f"http://{PROXY_HOST}:{UPSTREAM_PORT}{self.path}"
        headers = {}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        timeout = httpx.Timeout(connect=10, read=3600, write=120, pool=10)
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream(
                    self.command,
                    url,
                    content=body if body else None,
                    headers=headers,
                ) as response:
                    self._started = True
                    self.send_response(response.status_code)
                    upstream_type = response.headers.get(
                        "content-type", "application/json"
                    )
                    self.send_header("Content-Type", upstream_type)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            ENGINE.clear_fault()
        except OSError as exc:
            if client_closed(exc):
                note_client_closed()
                self.close_connection = True
                return
            ENGINE.note_fault(str(exc))
            raise
        except httpx.HTTPError as exc:
            ENGINE.note_fault(f"upstream: {exc.__class__.__name__}")
            raise

    def _dispatch(self) -> None:
        assert ENGINE is not None
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/v1/health"):
            self._health()
            return
        if path in ("/v1/models", "/models"):
            self._models()
            return
        body = self._read_body()
        payload = None
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict) and (
            "model" in payload or path.endswith("/chat/completions")
        ):
            model_id = resolve_model(payload)
            body = json.dumps(payload).encode("utf-8")
        else:
            model_id = ENGINE.model_id or "qwen3.8-27b"
        watched = path.endswith("/chat/completions") or path.endswith("/completions")
        if watched and DASH is not None:
            DASH.request(model_id, path)
        ENGINE.ensure(model_id)
        try:
            self._proxy(body)
        finally:
            ENGINE.release()
            if watched and DASH is not None:
                DASH.end_request()

    def _fail(self, status: int, message: str) -> None:
        if getattr(self, "_started", False):
            self.close_connection = True
            return
        self._send_json(status, {"error": {"message": message}})

    def do_GET(self) -> None:
        try:
            self._dispatch()
        except OSError as exc:
            if client_closed(exc):
                note_client_closed()
                self.close_connection = True
                return
            self._fail(503, str(exc))
        except Exception as exc:
            self._fail(503, str(exc))

    def do_POST(self) -> None:
        try:
            self._dispatch()
        except OSError as exc:
            if client_closed(exc):
                note_client_closed()
                self.close_connection = True
                return
            self._fail(503, str(exc))
        except KeyError as exc:
            self._fail(404, f"unknown model: {exc}")
        except Exception as exc:
            self._fail(503, str(exc))


def main() -> None:
    global ENGINE, DASH
    for spec in MODELS.values():
        if not spec["path"].exists():
            raise SystemExit(f"missing model: {spec['path']}")
    if port_open(PROXY_PORT):
        raise SystemExit(f"{PROXY_HOST}:{PROXY_PORT} is already in use")
    exe = ensure_binary()
    log(f"LLAMA {exe}")
    ENGINE = Engine(exe, server_help(exe))
    DASH = Dashboard()
    DASH.start()
    server = ProxyServer((PROXY_HOST, PROXY_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f"PROXY http://{PROXY_HOST}:{PROXY_PORT}/v1")
    try:
        ENGINE.ensure("qwen3.8-27b")
        ENGINE.release()
        log(
            "HAZIR      "
            f"{ENGINE.model_id}  ctx {ENGINE.ctx}  kv {ENGINE.kv}  "
            f"http://{PROXY_HOST}:{PROXY_PORT}/v1"
        )
        log("komutlar   [s] durum   [h] yardim   [q] cikis")
        last_heal = time.monotonic()
        while not STOP.wait(1):
            if time.monotonic() - last_heal >= WATCH_S:
                ENGINE.heal()
                last_heal = time.monotonic()
    except KeyboardInterrupt:
        log("STOP")
    finally:
        STOP.set()
        sys.stdout.write("\n")
        sys.stdout.flush()
        ENGINE.stop()
        server.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL {exc}")
        sys.exit(1)
