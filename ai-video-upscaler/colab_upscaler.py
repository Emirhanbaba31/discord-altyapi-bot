#@title 🎬 AI Video Upscaler + FPS İnterpolasyon (Real-ESRGAN + RIFE 4.26) — Tek Hücre { display-mode: "form" }
#@markdown **Kullanım:** Çalışma zamanı → *Çalışma zamanı türünü değiştir* → **L4 GPU** seçin, aşağıyı doldurun ve ▶ ile çalıştırın.
#@markdown Hücre çalıştığı sürece site açık kalır. Durdurmak için ■ (durdur) butonuna basın.

#@markdown ---
#@markdown ### 🔑 Sabit link (ngrok – ücretsiz)
#@markdown Ücretsiz hesap açın: https://dashboard.ngrok.com/signup → **Your Authtoken** sayfasındaki token'ı yapıştırın.
NGROK_TOKEN = ""  #@param {type:"string"}
#@markdown Sabit adresiniz: https://dashboard.ngrok.com/domains (ör. `benim-adim.ngrok-free.app`). Boş bırakırsanız hesabınızın varsayılan sabit adresi kullanılır.
NGROK_DOMAIN = ""  #@param {type:"string"}

#@markdown ---
#@markdown ### ⚙️ Diğer ayarlar
#@markdown **SIFRE:** Siteye girişte şifre sorulsun (boş = şifresiz). Kullanıcı adı alanına herhangi bir şey yazabilirsiniz.
SIFRE = ""  #@param {type:"string"}
#@markdown **DRIVE_KAYDET:** Biten videolar Google Drive'da `AI_Upscaler` klasörüne de kopyalansın (Colab kapansa da kaybolmaz).
DRIVE_KAYDET = False  #@param {type:"boolean"}
#@markdown **TUNEL:** `ngrok` = link hiç değişmez · `cloudflare` = hesap gerekmez ama link her çalıştırmada değişir.
TUNEL = "ngrok"  #@param ["ngrok", "cloudflare"]

# =============================================================================
#  Buradan aşağısını değiştirmenize gerek yok.
#  Büyütme : Real-ESRGAN (RealESRGAN_x4plus / x2plus / x4plus_anime_6B / realesr-general-x4v3)
#  Ara kare: RIFE 4.26 (Practical-RIFE) + yinelenen kare ayıklama + sahne geçişi koruması
#  Kodlama : NVENC 10-bit HEVC / H.264 (p7 + multipass + AQ) → libx265/libx264 yedekli
#  L4 / A100 gibi büyük GPU'larda: fp16 + channels_last + TF32, toplu (batch) RIFE
# =============================================================================

import base64
import builtins
import hashlib
import hmac
import importlib.util
import json
import math
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction
from pathlib import Path

VERSION = "2.0.1"

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
if IN_COLAB:
    print(f"⏳ AI Video Upscaler v{VERSION} yükleniyor… (modeller ilk seferde indirilir, 1-2 dk sürebilir)", flush=True)

# Ortam değişkenleri (UPSCALER_TUNEL=yok gibi) form değerlerini ezer; başlatıcı hücre ve yerel test içindir.
for _key in ("NGROK_TOKEN", "NGROK_DOMAIN", "SIFRE", "TUNEL"):
    if os.environ.get("UPSCALER_" + _key):
        globals()[_key] = os.environ["UPSCALER_" + _key]
if os.environ.get("UPSCALER_DRIVE_KAYDET"):
    DRIVE_KAYDET = os.environ["UPSCALER_DRIVE_KAYDET"].lower() in ("1", "true", "evet")

PORT = int(os.environ.get("UPSCALER_PORT", "7860"))
BASE = Path(os.environ.get("UPSCALER_DIR") or ("/content/ai_upscaler" if IN_COLAB else "ai_upscaler_data")).resolve()
UP_DIR, OUT_DIR, WORK_DIR, MODEL_DIR, BIN_DIR = (BASE / d for d in ("uploads", "outputs", "work", "models", "bin"))
DRIVE_DIR = None
CHUNK = 8 * 1024 * 1024
DL_SECRET = secrets.token_bytes(16)
DL_STATE = {"base": None}

_RE = "https://github.com/xinntao/Real-ESRGAN/releases/download/"
SR_MODELS = {
    "RealESRGAN_x4plus": _RE + "v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus": _RE + "v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B": _RE + "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-general-x4v3": _RE + "v0.2.5.0/realesr-general-x4v3.pth",
}
RIFE_URL = "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife426.pth"

RES_TARGETS = {"orig": None, "1080": 1080, "1440": 1440, "2160": 2160}
RES_LABELS = {"orig": "orijinal", "1080": "1080p", "1440": "2K", "2160": "4K"}
FPS_CHOICES = ("orig", "60", "90", "120")
MODEL_CHOICES = ("kalite", "hizli", "anime")
CODEC_CHOICES = ("h265", "h264")
QUALITY_CHOICES = ("maksimum", "yuksek", "dengeli")

_MAIN_THREAD = threading.main_thread()
_LOGQ = queue.Queue()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if threading.current_thread() is _MAIN_THREAD:
        print(line, flush=True)
    else:
        _LOGQ.put(line)


def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def pip_install(*pkgs):
    missing = [p for p in pkgs if importlib.util.find_spec(p.replace("-", "_")) is None]
    if missing:
        log(f"Paketler kuruluyor: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def download(url, dst, label):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    last = None
    for attempt in range(4):
        try:
            log(f"İndiriliyor: {label}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ai-upscaler"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if tmp.stat().st_size < 1_000_000:
                raise IOError("dosya eksik indi")
            os.replace(tmp, dst)
            return dst
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"İndirme hatası ({label}): {e} — tekrar deneniyor")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{label} indirilemedi: {last}")


# =============================================================================
#  GPU ayarları ve model mimarileri (Real-ESRGAN, RIFE 4.26) — ağırlıklarla birebir uyumlu
# =============================================================================
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HALF = DEV.type == "cuda"
GPU_NAME = torch.cuda.get_device_name(0) if HALF else None
VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1) if HALF else 0
BIG_GPU = VRAM_GB >= 20  # L4 (24 GB), A100, H100 …
if HALF:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
MEM_FMT = torch.channels_last if HALF else torch.contiguous_format


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(nf, gc) for _ in range(3))

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale=4, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale
        cin = 3 * {4: 1, 2: 4, 1: 16}[scale]
        self.conv_first = nn.Conv2d(cin, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


_GRID_CACHE = {}


def _warp(x, flow):
    _, _, h, w = flow.shape
    key = (flow.device, flow.dtype, h, w)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        gx = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(1, 1, h, w)
        gy = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(1, 1, h, w)
        grid = _GRID_CACHE[key] = torch.cat((gx, gy), 1)
    flow = torch.cat((flow[:, 0:1] / ((x.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((x.shape[2] - 1.0) / 2.0)), 1)
    return F.grid_sample(x, (grid + flow).permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


def _rconv(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=True), nn.LeakyReLU(0.2, True))


class _ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class _IFBlock(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv0 = nn.Sequential(_rconv(cin, c // 2, 3, 2, 1), _rconv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[_ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        return self.cnn3(self.relu(self.cnn2(x)))


class IFNet(nn.Module):
    """RIFE 4.26 (Practical-RIFE IFNet_HDv3, 5 kademe)."""

    def __init__(self):
        super().__init__()
        self.block0 = _IFBlock(7 + 8, 192)
        self.block1 = _IFBlock(8 + 4 + 8 + 8, 128)
        self.block2 = _IFBlock(8 + 4 + 8 + 8, 96)
        self.block3 = _IFBlock(8 + 4 + 8 + 8, 64)
        self.block4 = _IFBlock(8 + 4 + 8 + 8, 32)
        self.encode = _Head()

    def forward(self, img0, img1, timestep, scale_list):
        """img0, img1: (1,3,H,W) · timestep: (N,1,1,1) → N ara kare (N,3,H,W)."""
        n, (h, w) = timestep.shape[0], img0.shape[2:]
        f0, f1 = (self.encode(i).expand(n, -1, -1, -1) for i in (img0, img1))
        img0, img1 = img0.expand(n, -1, -1, -1), img1.expand(n, -1, -1, -1)
        timestep = timestep.expand(n, 1, h, w)
        flow = mask = feat = None
        wimg0, wimg1 = img0, img1
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        for blk, scale in zip(blocks, scale_list):
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale)
            else:
                wf0, wf1 = _warp(f0, flow[:, :2]), _warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat((wimg0, wimg1, wf0, wf1, timestep, mask, feat), 1), flow, scale)
                flow = flow + fd
            wimg0, wimg1 = _warp(img0, flow[:, :2]), _warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return wimg0 * mask + wimg1 * (1 - mask)


def _load_state(path):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    for k in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and isinstance(sd.get(k), dict):
            sd = sd[k]
            break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _build_sr(sd):
    if "conv_first.weight" in sd:
        nb = len({k.split(".")[1] for k in sd if k.startswith("body.") and ".rdb1.conv1.weight" in k})
        scale = {3: 4, 12: 2, 48: 1}[sd["conv_first.weight"].shape[1]]
        return RRDBNet(scale=scale, nf=sd["conv_first.weight"].shape[0], nb=nb), scale
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
    scale = int(round(math.sqrt(sd[f"body.{last}.weight"].shape[0] / 3)))
    return SRVGGNetCompact(sd["body.0.weight"].shape[0], (last - 2) // 2, scale), scale


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


class SRRunner:
    """Real-ESRGAN çalıştırıcı: kenar dolgusu, gerekirse otomatik karo (tile) bölme."""

    def __init__(self, net, scale):
        self.net, self.scale, self.tile = net, scale, 0

    def _tiled(self, x):
        if not self.tile:
            return self.net(x)
        s, t, pad = self.scale, self.tile, 32
        _, c, h, w = x.shape
        out = x.new_zeros((1, c, h * s, w * s))
        for ty in range(0, h, t):
            for tx in range(0, w, t):
                th, tw = min(t, h - ty), min(t, w - tx)
                y0, x0 = max(ty - pad, 0), max(tx - pad, 0)
                y1, x1 = min(ty + th + pad, h), min(tx + tw + pad, w)
                o = self.net(x[:, :, y0:y1, x0:x1])
                oy, ox = (ty - y0) * s, (tx - x0) * s
                out[:, :, ty * s:(ty + th) * s, tx * s:(tx + tw) * s] = o[:, :, oy:oy + th * s, ox:ox + tw * s]
        return out

    def __call__(self, x):
        h, w = x.shape[2:]
        p = 10 if min(h, w) > 10 else 0
        if p:
            x = F.pad(x, (p, p, p, p), mode="reflect")
        eh, ew = x.shape[2] % 2, x.shape[3] % 2  # x2 modeli çift boyut ister
        if self.scale == 2 and (eh or ew):
            x = F.pad(x, (0, ew, 0, eh), mode="replicate")
        x = x.contiguous(memory_format=MEM_FMT)
        while True:
            try:
                y = self._tiled(x)
                break
            except (RuntimeError, MemoryError) as e:
                if not _is_oom(e) or self.tile == 64:
                    raise
                torch.cuda.empty_cache()
                self.tile = 768 if not self.tile else max(64, self.tile // 2)
                log(f"GPU belleği yetmedi, karo boyutu {self.tile} px yapıldı")
        s = self.scale
        return y[:, :, p * s:(p + h) * s, p * s:(p + w) * s]


class RifeRunner:
    def __init__(self, net):
        self.net, self.batch = net, (4 if BIG_GPU else 2)

    def __call__(self, a, b, ts):
        """a, b: (1,3,H,W) float32 [0,1] — ts: ara zamanlar (0..1) → (len(ts),3,H,W)."""
        h, w = a.shape[2:]
        scale = 0.5 if min(h, w) > 1080 else 1.0  # 2K/4K'da akış yarım çözünürlükte (Practical-RIFE önerisi)
        m = max(128, int(128 / scale))
        ph, pw = ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m
        pad = (0, pw - w, 0, ph - h)
        a, b = F.pad(a, pad), F.pad(b, pad)
        scales = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        outs, i = [], 0
        while i < len(ts):
            part = ts[i:i + self.batch]
            t = torch.tensor(part, device=a.device, dtype=a.dtype).view(-1, 1, 1, 1)
            try:
                o = self.net(a, b, t, scales)
            except (RuntimeError, MemoryError) as e:
                if not _is_oom(e) or self.batch == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch = max(1, self.batch // 2)
                log(f"GPU belleği yetmedi, RIFE toplu işlem boyutu {self.batch} yapıldı")
                continue
            outs.append(o[:, :, :h, :w])
            i += len(part)
        return torch.cat(outs).clamp_(0, 1)


class ModelStore:
    def __init__(self):
        self.sr_cache, self.rife_runner = {}, None

    def sr(self, name):
        if name not in self.sr_cache:
            path = download(SR_MODELS[name], MODEL_DIR / f"{name}.pth", name)
            try:
                sd = _load_state(path)
            except Exception:  # bozuk indirme → bir kez daha indir
                path.unlink(missing_ok=True)
                sd = _load_state(download(SR_MODELS[name], path, name))
            net, scale = _build_sr(sd)
            net.load_state_dict(sd, strict=True)
            net.eval().requires_grad_(False).to(DEV, memory_format=MEM_FMT)
            if HALF:
                net.half()
            if len(self.sr_cache) >= (4 if BIG_GPU else 2):
                self.sr_cache.clear()
                if HALF:
                    torch.cuda.empty_cache()
            self.sr_cache[name] = SRRunner(net, scale)
        return self.sr_cache[name]

    def rife(self):
        if self.rife_runner is None:
            path = download(RIFE_URL, MODEL_DIR / "rife426.pth", "RIFE 4.26")
            sd = {k: v for k, v in _load_state(path).items() if not k.startswith(("teacher.", "caltime."))}
            net = IFNet()
            net.load_state_dict(sd, strict=True)
            self.rife_runner = RifeRunner(net.eval().requires_grad_(False).to(DEV))
        return self.rife_runner


MODELS = ModelStore()

# =============================================================================
#  Video okuma / yazma (ffmpeg boruları) ve yardımcılar
# =============================================================================
HAS_ZSCALE = False
NVENC = {"h264": False, "hevc": False}
_ENC_CACHE = {}


class Cancelled(Exception):
    pass


def frac_str(fr):
    fr = Fraction(fr)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _rate(s):
    try:
        n, d = (int(v) for v in str(s).split("/"))
        return Fraction(n, d).limit_denominator(1001) if n > 0 and d > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _even(v):
    v = int(round(v))
    return max(2, v - v % 2)


def plan_size(w, h, res):
    target = RES_TARGETS[res]
    if target is None:
        return _even(w), _even(h), False
    k = target / min(w, h)
    return _even(w * k), _even(h * k), k > 1.0001


def probe_video(path):
    r = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=120)
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise ValueError("Dosyada video bulunamadı (desteklenmeyen veya bozuk dosya).")
    v = vids[0]
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if w < 16 or h < 16:
        raise ValueError("Video çözünürlüğü okunamadı.")
    rot = 0
    try:
        rot = int(float(v.get("tags", {}).get("rotate", 0)))
    except ValueError:
        pass
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = _rate(v.get("avg_frame_rate"))
    if not fps or fps > 240 or fps < 1:
        fps = _rate(v.get("r_frame_rate")) or Fraction(30)
    fmt = data.get("format", {})
    dur = float(v.get("duration") or fmt.get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or int(round(dur * float(fps)))
    space = v.get("color_space", "")
    matrix = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "bt2020nc": "bt2020",
              "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}.get(space)
    if not matrix:
        matrix = "auto" if space == "gbr" else ("bt709" if min(w, h) >= 720 else "smpte170m")
    pix = v.get("pix_fmt", "")
    try:
        bits = int(v.get("bits_per_raw_sample") or 0)
    except ValueError:
        bits = 0
    if not bits:
        bits = 10 if re.search(r"p01[06]|p2[01]0|p4[01]0|1[0246][lb]e", pix) else 8
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    def start(s):
        try:
            return float(s.get("start_time") or 0)
        except ValueError:
            return 0.0

    hdr = v.get("color_transfer") in ("smpte2084", "arib-std-b67")
    return {
        "width": w, "height": h, "fps_num": fps.numerator, "fps_den": fps.denominator, "fps": round(float(fps), 3),
        "duration": round(dur, 3), "frames": frames, "vindex": int(v.get("index", 0)), "codec": v.get("codec_name", "?"),
        "has_audio": bool(audio), "acodec": audio[0].get("codec_name", "") if audio else "",
        "matrix": matrix, "range": v.get("color_range", ""), "hdr": hdr, "bits": bits, "deep": bits > 8 or hdr,
        # Ses videodan önce başlıyorsa ne kadar kırpılacağı (A/V senkronu için)
        "audio_trim": max(0.0, start(v) - min([start(v)] + [start(a) for a in audio])) if audio else 0.0,
    }


def build_decode_cmd(path, info, fps, size=None):
    """Kaynak → RGB (8-bit kaynakta rgb24, 10-bit/HDR kaynakta rgb48le) ham kare akışı."""
    vf, matrix, rng = [], info["matrix"], info["range"]
    if info["hdr"] and HAS_ZSCALE:  # HDR (iPhone vb.) → SDR ton eşleme, 10-bit hassasiyette
        vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
                  "zscale=t=bt709:m=bt709:r=tv,format=yuv420p10le")
        matrix, rng = "bt709", "tv"
    vf.append(f"fps=fps={frac_str(fps)}")
    pix = "rgb48le" if info["deep"] else "rgb24"
    sc = "scale=" + (f"{size[0]}:{size[1]}:" if size else "")
    sc += f"flags=lanczos+accurate_rnd+full_chroma_int:in_color_matrix={matrix}"
    if rng in ("tv", "pc"):
        sc += f":in_range={rng}"
    vf += [sc, f"format={pix}"]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:{info['vindex']}",
            "-an", "-sn", "-dn", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", pix, "-"]


def encoder_candidates(codec, quality, w, h, fps):
    """(etiket, ffmpeg argümanları, boruya giden pix_fmt, çıkış pix_fmt) — en iyiden en güvenliye."""
    cq_hevc, cq_avc, crf_avc, crf_hevc = {"maksimum": (16, 15, 14, 16), "yuksek": (20, 19, 18, 20),
                                          "dengeli": (25, 23, 22, 24)}[quality]
    heavy = w * h * float(fps) > 1920 * 1080 * 60
    hq = ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-multipass", "fullres", "-rc-lookahead", "32",
          "-spatial-aq", "1", "-temporal-aq", "1", "-aq-strength", "8"]
    nv265_10 = ("H.265 10-bit · NVENC", ["-c:v", "hevc_nvenc", *hq, "-cq", str(cq_hevc), "-bf", "4", "-b_ref_mode",
                                         "middle", "-profile:v", "main10", "-tag:v", "hvc1"], "p010le", "p010le")
    nv265_10s = ("H.265 10-bit · NVENC", ["-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", str(cq_hevc),
                                          "-b:v", "0", "-profile:v", "main10", "-tag:v", "hvc1"], "p010le", "p010le")
    nv265_8 = ("H.265 · NVENC", ["-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", str(cq_hevc), "-b:v",
                                 "0", "-profile:v", "main", "-tag:v", "hvc1"], "yuv420p", "yuv420p")
    nv264 = ("H.264 · NVENC", ["-c:v", "h264_nvenc", *hq, "-cq", str(cq_avc), "-bf", "3", "-b_ref_mode", "middle",
                               "-profile:v", "high"], "yuv420p", "yuv420p")
    nv264s = ("H.264 · NVENC", ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", str(cq_avc), "-b:v", "0",
                                "-profile:v", "high"], "yuv420p", "yuv420p")
    x264 = ("H.264 · CPU", ["-c:v", "libx264", "-preset", "medium" if heavy else "slow", "-crf", str(crf_avc),
                            "-profile:v", "high"], "yuv420p", "yuv420p")
    x265 = ("H.265 10-bit · CPU", ["-c:v", "libx265", "-preset", "fast" if heavy else "medium", "-crf", str(crf_hevc),
                                   "-profile:v", "main10", "-tag:v", "hvc1", "-x265-params", "log-level=error"],
            "p010le", "yuv420p10le")
    if codec == "h265":
        return [nv265_10, nv265_10s, nv265_8, x265, x264]
    return [nv264, nv264s, nv265_8, x264]


def encoder_works(args, in_pix, out_pix, w, h, fps):
    key = (tuple(args), out_pix, w, h, frac_str(fps))
    if key not in _ENC_CACHE:
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:r={frac_str(fps)}",
               "-frames:v", "3", "-vf", f"format={in_pix}", *args, "-pix_fmt", out_pix, "-f", "null", "-"]
        try:
            _ENC_CACHE[key] = run(cmd, timeout=120).returncode == 0
        except subprocess.TimeoutExpired:
            _ENC_CACHE[key] = False
    return _ENC_CACHE[key]


def pick_encoder(codec, quality, w, h, fps):
    for label, args, in_pix, out_pix in encoder_candidates(codec, quality, w, h, fps):
        if encoder_works(args, in_pix, out_pix, w, h, fps):
            return label, args, in_pix, out_pix
    raise RuntimeError("Uygun video kodlayıcı bulunamadı.")


def _set_pipe_size(f):
    try:
        import fcntl
        fcntl.fcntl(f.fileno(), 1031, 1 << 20)  # F_SETPIPE_SZ = 1 MB
    except Exception:  # noqa: BLE001
        pass


class FrameReader:
    def __init__(self, cmd, w, h, deep, cancel_check):
        self.w, self.h, self.cancel_check = w, h, cancel_check
        self.dtype = np.uint16 if deep else np.uint8
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        _set_pipe_size(self.p.stdout)
        self.q, self.err, self.stop = queue.Queue(maxsize=8), b"", False
        threading.Thread(target=self._drain, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _drain(self):
        self.err = self.p.stderr.read()

    def _put(self, item):
        while not self.stop:
            try:
                self.q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def _loop(self):
        try:
            while not self.stop:
                arr = np.empty((self.h, self.w, 3), self.dtype)
                mv, got = memoryview(arr).cast("B"), 0
                while got < arr.nbytes:
                    n = self.p.stdout.readinto(mv[got:])
                    if not n:
                        break
                    got += n
                if got < arr.nbytes:
                    break
                self._put(arr)
        finally:
            self._put(None)

    def __iter__(self):
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                self.cancel_check()
                continue
            if item is None:
                rc = self.p.wait()
                if rc != 0 and not self.stop:
                    raise RuntimeError("Video çözümlenemedi: " + self.err.decode("utf-8", "ignore")[-400:])
                return
            yield item

    def close(self):
        self.stop = True
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()


class FrameWriter:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _set_pipe_size(self.p.stdin)
        self.q, self.err, self.exc = queue.Queue(maxsize=24), b"", None
        threading.Thread(target=self._drain, daemon=True).start()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _drain(self):
        self.err = self.p.stderr.read()

    def _loop(self):
        try:
            while True:
                a = self.q.get()
                if a is None:
                    break
                self.p.stdin.write(memoryview(a).cast("B"))
        except Exception as e:  # noqa: BLE001
            self.exc = e
            while True:  # kuyruğu boşalt ki üretici takılmasın
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

    def put(self, arr):
        while True:
            if self.exc:
                raise RuntimeError("Video kodlayıcı hata verdi: " + self.err.decode("utf-8", "ignore")[-400:])
            try:
                self.q.put(arr, timeout=0.5)
                return
            except queue.Full:
                pass

    def close(self):
        self.q.put(None)
        self.t.join()
        try:
            self.p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        rc = self.p.wait()
        if rc != 0 or self.exc:
            raise RuntimeError("Video kodlayıcı hata verdi: " + self.err.decode("utf-8", "ignore")[-400:])

    def abort(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()


def to_tensor(arr):
    """(H,W,3) uint8/uint16 → (1,3,H,W) float32 [0,1] GPU'da (bellek düzeni zaten channels_last)."""
    if arr.dtype == np.uint16:
        t = torch.from_numpy(arr.view(np.int16)).to(DEV).to(torch.int32).bitwise_and_(0xFFFF).float().div_(65535.0)
    else:
        t = torch.from_numpy(arr).to(DEV).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def rgb_to_yuv(x, pix):
    """(B,3,H,W) RGB [0,1] → BT.709 sınırlı aralık yuv420p (8-bit) veya p010le (10-bit) — dönüşüm GPU'da."""
    kr, kb = 0.2126, 0.0722
    kg = 1 - kr - kb
    y = kr * x[:, 0] + kg * x[:, 1] + kb * x[:, 2]
    s = F.avg_pool2d(x, 2)
    ys = kr * s[:, 0] + kg * s[:, 1] + kb * s[:, 2]
    u, v = (s[:, 2] - ys) / 1.8556, (s[:, 0] - ys) / 1.5748
    if pix == "p010le":
        yy = (64 + 876 * y).round_().clamp_(0, 1023).to(torch.int32) * 64
        uv = torch.stack((512 + 896 * u, 512 + 896 * v), -1).round_().clamp_(0, 1023).to(torch.int32) * 64
        out = torch.cat((yy.flatten(1), uv.flatten(1)), 1)
        return torch.where(out >= 32768, out - 65536, out).to(torch.int16)  # uint16 bit deseni
    out = torch.cat(((16 + 219 * y).flatten(1), (128 + 224 * u).flatten(1), (128 + 224 * v).flatten(1)), 1)
    return out.round_().clamp_(0, 255).to(torch.uint8)


_SSIM_WIN = {}


def _scene_thumb(x):
    return F.interpolate(x, size=(32, 32), mode="area").mean(1, keepdim=True)


def _dup_thumb(x):
    f = max(1, int(max(x.shape[2:]) / 256))
    return F.avg_pool2d(x, f) if f > 1 else x


def _is_dup(a, b, strict=False):
    """Kopya kare mi? Sıkıştırma gürültüsünü tolere eder; gerçek hareket (kenar kayması) yerel olarak büyük,
    yavaş kararma/kaydırma ise ortalamada fark üretir."""
    d = (a - b).abs()
    return (d.max().item() < (0.08 if strict else 0.2) and d.mean().item() < 0.002
            and (d > 0.05).float().mean().item() < 0.0005)


def _ssim(a, b):
    w = _SSIM_WIN.get(a.device)
    if w is None:
        g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        w = _SSIM_WIN[a.device] = (g[:, None] * g[None, :]).view(1, 1, 11, 11).to(a.device)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a ** 2
    sbb = F.conv2d(b * b, w) - mu_b ** 2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())


def dl_token(jid):
    return hmac.new(DL_SECRET, jid.encode(), hashlib.sha256).hexdigest()[:32]


# =============================================================================
#  İş kuyruğu ve işleme hattı
# =============================================================================
JOBS, UPLOADS = {}, {}
JLOCK = threading.Lock()
JOBQ = queue.Queue()
STATE = {"version": 0}

PUBLIC_KEYS = ("id", "name", "status", "stage", "progress", "fps_proc", "eta", "created", "started", "finished",
               "error", "out_name", "out_size", "in_desc", "out_desc", "opts", "engine", "done", "total", "drive")


def job_set(job, **kw):
    with JLOCK:
        job.update(kw)
        STATE["version"] += 1


def job_public(job):
    d = {k: job.get(k) for k in PUBLIC_KEYS}
    if job.get("status") == "done":
        d["dl_token"] = dl_token(job["id"])
    return d


def snapshot():
    with JLOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
        queued = [j["id"] for j in sorted(JOBS.values(), key=lambda j: j["created"]) if j["status"] == "queued"]
        out = []
        for j in jobs:
            d = job_public(j)
            d["queue_pos"] = queued.index(j["id"]) + 1 if j["id"] in queued else 0
            out.append(d)
        return {"jobs": out, "version": STATE["version"], "dl_base": DL_STATE["base"]}


def pick_sr_model(model, factor):
    if model == "anime":
        return "RealESRGAN_x4plus_anime_6B"
    if model == "hizli":
        return "realesr-general-x4v3"
    return "RealESRGAN_x2plus" if factor <= 2 else "RealESRGAN_x4plus"


def _safe_name(name):
    base = re.sub(r"[^\w\-]+", "_", Path(name).stem, flags=re.UNICODE).strip("_") or "video"
    return base[:60]


def process_job(job):
    info, opt = job["info"], job["opts"]

    def check():
        if job.get("cancel"):
            raise Cancelled()

    src_w, src_h = info["width"], info["height"]
    fps_in = Fraction(info["fps_num"], info["fps_den"])
    out_w, out_h, need_ai = plan_size(src_w, src_h, opt["res"])
    fps_out = fps_in if opt["fps"] == "orig" else Fraction(int(opt["fps"]))
    use_rife = fps_out > fps_in * Fraction(101, 100)
    ratio = fps_in / fps_out  # çıkış karesi k → kaynak konumu k*ratio
    scene = use_rife and opt.get("scene", True)
    dedup = use_rife and opt.get("dedup", True)
    total_est = max(1, round(max(1, info["frames"]) * fps_out / fps_in))
    job_set(job, stage="Hazırlanıyor", total=total_est, done=0)

    sr = None
    engine = []
    if need_ai:
        name = pick_sr_model(opt["model"], min(out_w / src_w, out_h / src_h))
        job_set(job, stage=f"AI modeli yükleniyor ({name})")
        sr = MODELS.sr(name)
        engine.append(f"Büyütme: {name} (×{sr.scale})")
    elif opt["res"] == "orig":
        engine.append("Büyütme: yok (orijinal çözünürlük)")
    else:
        engine.append("Büyütme: gerek yok (kaynak hedeften büyük/eşit) → yalnızca boyutlandırma")
    rife = None
    if use_rife:
        job_set(job, stage="RIFE 4.26 interpolasyon modeli yükleniyor")
        rife = MODELS.rife()
        extras = [e for e, on in (("kopya kare ayıklama", dedup), ("sahne geçişi koruması", scene)) if on]
        engine.append("İnterpolasyon: RIFE 4.26" + (f" + {' + '.join(extras)}" if extras else ""))
    else:
        engine.append("İnterpolasyon: yok")
    check()

    job_set(job, stage="Video kodlayıcı hazırlanıyor")
    enc_label, enc_args, in_pix, out_pix = pick_encoder(opt["codec"], opt["quality"], out_w, out_h, fps_out)
    engine.append(f"Kodlayıcı: {enc_label}")
    if info["hdr"]:
        engine.append("HDR → SDR ton eşleme" if HAS_ZSCALE else "HDR (ton eşleme yok)")
    job_set(job, engine=engine)
    log(f"[{job['id']}] {job['name']}: {src_w}x{src_h}@{float(fps_in):.2f} → {out_w}x{out_h}@{float(fps_out):.2f} · "
        + " · ".join(engine))

    tmp_video = WORK_DIR / f"{job['id']}_video.mp4"
    dec_cmd = build_decode_cmd(job["path"], info, fps_in, None if need_ai else (out_w, out_h))
    dec_w, dec_h = (src_w, src_h) if need_ai else (out_w, out_h)
    enc_cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_pix,
               "-s", f"{out_w}x{out_h}", "-r", frac_str(fps_out), "-i", "-", *enc_args, "-pix_fmt", out_pix,
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-an", str(tmp_video)]

    reader = FrameReader(dec_cmd, dec_w, dec_h, info["deep"], check)
    writer = FrameWriter(enc_cmd)
    t_start = time.time()
    st = {"k": 0, "last_ui": 0.0, "dups": 0, "cuts": 0}

    def emit(batch):
        check()
        data = rgb_to_yuv(batch, in_pix).cpu().numpy()
        for row in data:
            writer.put(row)
        st["k"] += len(data)
        now = time.time()
        if now - st["last_ui"] > 0.5:
            st["last_ui"] = now
            k, el = st["k"], max(now - t_start, 1e-6)
            total = max(job.get("total") or 1, k)
            speed = k / el
            job_set(job, stage="İşleniyor", done=k, progress=min(0.99, k / total), fps_proc=round(speed, 2),
                    eta=int((total - k) / speed) if speed > 0 else None)

    def upscale(x):
        if sr is None:
            return x
        y = sr(x.half() if HALF else x).float()
        if y.shape[-2:] != (out_h, out_w):
            y = F.interpolate(y, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
        return y.clamp_(0, 1)

    def emit_span(prev, cur, ts, cut):
        """prev→cur aralığındaki çıkış karelerini (t ∈ [0,1)) üretip kodlayıcıya gönderir."""
        for c in range(0, len(ts), 8):
            frames, need = [], []
            for t in ts[c:c + 8]:
                if t < 1e-3 or cur is prev:
                    frames.append(prev)
                elif t > 1 - 1e-3:
                    frames.append(cur)
                elif rife is None:
                    frames.append(prev if t < 0.5 else cur)
                elif cut:
                    frames.append(prev)  # sahne değişiminde bulanık ara kare yerine kesme
                else:
                    frames.append(None)
                    need.append(t)
            if need:
                mids = iter(rife(prev, cur, need).split(1))
                frames = [f if f is not None else next(mids) for f in frames]
            emit(torch.cat(frames))

    try:
        with torch.inference_mode():
            prev = prev_dt = prev_st = None
            pj, j, k = 0, -1, 0
            for arr in reader:
                check()
                j += 1
                x = to_tensor(arr)
                dt = _dup_thumb(x)
                cut = False
                if prev is not None and _is_dup(prev_dt, dt, strict=not dedup):
                    st["dups"] += 1
                    if dedup and j - pj < 3:
                        continue  # kopya kare: zaman çizelgesinden çıkar, iki gerçek kare arasında interpole et
                    # önceki AI çıktısını yeniden kullan (titreme olmaz, hızlı); karşılaştırma referansı çapa
                    # karede kalır ki yavaş değişimler birikip yakalansın
                    cur, cur_st, dt = prev, prev_st, prev_dt
                else:
                    cur = upscale(x)
                    cur_st = _scene_thumb(x) if scene else None
                    if scene and prev is not None and _ssim(prev_st, cur_st) < 0.2:
                        cut = True
                        st["cuts"] += 1
                if prev is None:
                    prev, prev_dt, prev_st, pj = cur, dt, cur_st, j
                    continue
                ts = []
                while k * ratio < j:  # [pj, j) aralığına düşen çıkış kareleri
                    ts.append(float((k * ratio - pj) / (j - pj)))
                    k += 1
                if ts:
                    emit_span(prev, cur, ts, cut)
                prev, prev_dt, prev_st, pj = cur, dt, cur_st, j
            if prev is None:
                raise RuntimeError("Videodan hiç kare okunamadı.")
            total = max(1, round((j + 1) * fps_out / fps_in))
            job_set(job, total=total)
            while k < total:  # son kare(ler) — süre kaynakla aynı kalsın
                n = min(8, total - k)
                emit(prev.expand(n, -1, -1, -1))
                k += n
        writer.close()
    except BaseException:
        reader.close()
        writer.abort()
        tmp_video.unlink(missing_ok=True)
        raise
    reader.close()
    if use_rife:
        log(f"[{job['id']}] Kopya kare: {st['dups']} · sahne geçişi: {st['cuts']}")

    check()
    job_set(job, stage="Ses ekleniyor ve dosya hazırlanıyor", progress=0.995, eta=None)
    out_name = (f"{_safe_name(job['name'])}_{RES_LABELS[opt['res']]}_"
                f"{'orjfps' if opt['fps'] == 'orig' else opt['fps'] + 'fps'}_{job['id'][:4]}.mp4")
    final = OUT_DIR / out_name
    base_cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(tmp_video)]
    ok = False
    if info["has_audio"]:
        audio_in = ["-i", str(job["path"]), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
        tail = ["-movflags", "+faststart", str(final)]
        attempts = []
        if info["audio_trim"] <= 0.005 and info["acodec"] in ("aac", "mp3", "ac3", "eac3", "alac"):
            attempts.append(["-c:a", "copy"])  # orijinal ses, kayıpsız
        af = ["-af", f"atrim=start={info['audio_trim']:.4f},asetpts=PTS-STARTPTS"] if info["audio_trim"] > 0.005 else []
        attempts.append(["-c:a", "aac", "-b:a", "320k", *af])
        for a in attempts:
            r = run(base_cmd + audio_in + a + tail)
            if r.returncode == 0:
                ok = True
                break
        if not ok:
            log(f"[{job['id']}] Ses eklenemedi, sessiz kaydediliyor: {r.stderr[-300:]}")
    if not ok:
        r = run(base_cmd + ["-c", "copy", "-movflags", "+faststart", str(final)])
        if r.returncode != 0:
            raise RuntimeError("Çıktı dosyası yazılamadı: " + r.stderr[-400:])
    tmp_video.unlink(missing_ok=True)

    drive_path = None
    if DRIVE_DIR:
        try:
            job_set(job, stage="Google Drive'a kopyalanıyor")
            drive_path = str(Path(shutil.copy2(final, DRIVE_DIR / out_name)))
        except Exception as e:  # noqa: BLE001
            log(f"[{job['id']}] Drive'a kopyalanamadı: {e}")
    el = time.time() - t_start
    job_set(job, status="done", stage="Tamamlandı", progress=1.0, eta=None, finished=time.time(),
            out_name=out_name, out_path=str(final), out_size=final.stat().st_size, drive=drive_path,
            done=st["k"], total=st["k"], fps_proc=round(st["k"] / max(el, 1e-6), 2))
    log(f"[{job['id']}] ✅ Bitti ({el / 60:.1f} dk): {final}")


def worker_loop(stop_event):
    while not stop_event.is_set():
        try:
            jid = JOBQ.get(timeout=1.0)
        except queue.Empty:
            continue
        job = JOBS.get(jid)
        if not job or job["status"] != "queued":
            continue
        job_set(job, status="running", started=time.time(), stage="Başlıyor")
        try:
            process_job(job)
        except Cancelled:
            job_set(job, status="canceled", stage="İptal edildi", finished=time.time(), eta=None)
            log(f"[{jid}] İptal edildi")
        except Exception as e:  # noqa: BLE001
            msg = "GPU belleği yetmedi; daha düşük çözünürlük deneyin." if _is_oom(e) else str(e)
            job_set(job, status="error", stage="Hata", error=msg[-600:], finished=time.time(), eta=None)
            log(f"[{jid}] ❌ Hata: {e}\n{traceback.format_exc()}")
        finally:
            if HALF:
                torch.cuda.empty_cache()


# =============================================================================
#  Web arayüzü (FastAPI)
# =============================================================================
PAGE_HTML = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Video Upscaler</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%8E%AC%3C/text%3E%3C/svg%3E">
<style>
:root{--bg:#0a0d14;--card:#121724;--card2:#181f30;--line:#253049;--text:#e9edf6;--muted:#8d97ae;--acc:#7c5cff;--acc2:#20d3ee;--ok:#22c55e;--err:#f05252;--warn:#f5a524}
*{box-sizing:border-box}
html,body{margin:0}
body{background:radial-gradient(1100px 560px at 0% -10%,#22185099 0,transparent 60%),radial-gradient(900px 480px at 110% -5%,#0c3a4a88 0,transparent 55%),var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;min-height:100vh}
.wrap{max-width:980px;margin:0 auto;padding:26px 16px 70px}
header{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:13px}
.mark{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:grid;place-items:center;font-size:22px;box-shadow:0 8px 30px #7c5cff55}
h1{font-size:21px;margin:0;letter-spacing:.2px}
.sub{color:var(--muted);font-size:13px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:7px 12px;border-radius:99px;background:var(--card);border:1px solid var(--line);font-size:13px;color:var(--muted)}
.chip b{color:var(--text);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok)}
.dot.warn{background:var(--warn);box-shadow:0 0 10px var(--warn)}
.card{background:linear-gradient(180deg,#141a29,#111623);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:18px}
.card h2{font-size:16px;margin:0 0 15px;display:flex;align-items:center;gap:10px}
.num{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:13px;font-weight:700;background:linear-gradient(135deg,var(--acc),var(--acc2))}
.drop{border:2px dashed var(--line);border-radius:15px;padding:34px 16px;text-align:center;cursor:pointer;transition:.18s;background:#0e1320}
.drop:hover,.drop.over{border-color:var(--acc);background:#7c5cff12}
.drop .big{font-size:38px;line-height:1;margin-bottom:8px}
.drop strong{display:block;font-size:16px;margin-bottom:4px}
.drop span{color:var(--muted);font-size:13px}
.file{display:none;margin-top:14px;padding:14px;border-radius:13px;background:var(--card2);border:1px solid var(--line)}
.file .row{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:center}
.fname{font-weight:600;word-break:break-all}
.muted{color:var(--muted);font-size:13px}
.bar{height:8px;background:#0b0f1a;border-radius:99px;overflow:hidden;margin-top:10px}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .35s}
.pills{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.pill{padding:4px 10px;border-radius:99px;background:#0b0f1a;border:1px solid var(--line);font-size:12.5px;color:var(--muted)}
.pill b{color:var(--text);font-weight:600}
.group{margin-bottom:17px}
.group .t{font-size:13px;color:var(--muted);margin-bottom:8px;font-weight:600;letter-spacing:.3px;text-transform:uppercase}
.seg{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:8px}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg label{display:block;padding:10px 12px;border-radius:12px;background:var(--card2);border:1px solid var(--line);cursor:pointer;font-weight:650;font-size:14.5px;user-select:none;transition:.15s}
.seg label:hover{border-color:#3a4766}
.seg label small{display:block;font-weight:400;color:var(--muted);font-size:12px;margin-top:1px}
.seg input:checked+label{border-color:var(--acc);background:linear-gradient(135deg,#7c5cff2e,#20d3ee14);box-shadow:inset 0 0 0 1px var(--acc)}
.seg input:focus-visible+label{outline:2px solid var(--acc2);outline-offset:2px}
.toggles{display:flex;flex-direction:column;gap:9px;margin-bottom:4px}
.toggle{display:flex;align-items:center;gap:10px;cursor:pointer;font-size:14px;user-select:none}
.toggle input{width:18px;height:18px;accent-color:var(--acc);flex:none}
.summary{margin:14px 0 16px;padding:12px 14px;border-radius:12px;background:#0b0f1a;border:1px solid var(--line);font-size:14px}
.summary .arrow{color:var(--acc2);font-weight:700;margin:0 6px}
.note{font-size:12.5px;color:var(--warn);margin-top:6px}
.btn{appearance:none;border:0;border-radius:13px;padding:13px 20px;font-weight:700;font-size:15px;cursor:pointer;color:#fff;background:linear-gradient(135deg,var(--acc),#5b8cff 60%,var(--acc2));box-shadow:0 10px 30px #7c5cff44;transition:.15s;width:100%}
.btn:hover{filter:brightness(1.08)}
.btn:disabled{opacity:.45;cursor:not-allowed;box-shadow:none}
.btn.sm{width:auto;padding:8px 13px;font-size:13px;border-radius:10px;box-shadow:none}
.btn.red{background:#3a1720;border:1px solid #6b2433;color:#ffb4c0}
.jobs-empty{color:var(--muted);text-align:center;padding:22px 10px;font-size:14px}
.job{border:1px solid var(--line);border-radius:15px;padding:15px;margin-top:11px;background:var(--card2)}
.job .top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}
.job .title{font-weight:650;word-break:break-all}
.badge{font-size:12px;font-weight:700;padding:4px 10px;border-radius:99px;white-space:nowrap}
.b-queued{background:#2a2f45;color:#c6cde0}.b-running{background:#1d2c55;color:#9cc0ff}.b-done{background:#123222;color:#7ff0a8}.b-error{background:#3a1720;color:#ffb4c0}.b-canceled{background:#2a2a2a;color:#bbb}
.job .stats{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:8px;font-size:13px;color:var(--muted)}
.job .eng{margin-top:8px;font-size:12.5px;color:var(--muted)}
.job .acts{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
.job .acts a.alt{color:var(--muted);font-size:12.5px}
.job .errm{margin-top:8px;font-size:13px;color:#ffb4c0;white-space:pre-wrap;word-break:break-word}
a.btn{text-decoration:none;display:inline-block;text-align:center}
.toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%) translateY(30px);opacity:0;background:#1d2440;border:1px solid var(--line);padding:11px 16px;border-radius:12px;font-size:14px;transition:.25s;z-index:9;max-width:calc(100% - 32px)}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
footer{color:var(--muted);font-size:12.5px;text-align:center;margin-top:26px;line-height:1.7}
@media (max-width:560px){.seg{grid-template-columns:repeat(2,1fr)}h1{font-size:19px}.card{padding:16px}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="mark">🎬</div>
      <div><h1>AI Video Upscaler</h1><div class="sub">Real-ESRGAN büyütme · RIFE 4.26 kare interpolasyonu</div></div>
    </div>
    <div class="chip" id="gpuChip"><span class="dot" id="gpuDot"></span><span id="gpuTxt">Bağlanıyor…</span></div>
  </header>

  <section class="card">
    <h2><span class="num">1</span>Video yükle</h2>
    <div class="drop" id="drop" tabindex="0" role="button" aria-label="Video seç">
      <div class="big">⬆️</div>
      <strong>Videoyu buraya sürükleyin veya tıklayın</strong>
      <span>MP4, MOV, MKV, WEBM, AVI… · Boyut sınırı yok</span>
      <input type="file" id="fileIn" accept="video/*,.mkv,.mov,.avi,.webm,.flv,.ts,.m4v,.wmv" hidden>
    </div>
    <div class="file" id="fileBox">
      <div class="row"><div class="fname" id="fName"></div><div class="muted" id="fStat"></div></div>
      <div class="bar"><i id="fBar"></i></div>
      <div class="pills" id="fInfo"></div>
    </div>
  </section>

  <section class="card">
    <h2><span class="num">2</span>Ayarlar</h2>
    <div id="opts"></div>
    <div class="toggles">
      <label class="toggle"><input type="checkbox" id="scene" checked> Sahne geçişlerinde bulanık ara kare oluşturma</label>
      <label class="toggle"><input type="checkbox" id="dedup" checked> Tekrarlanan (kopya) kareleri ayıkla — anime ve takılan videolarda çok daha akıcı</label>
    </div>
    <div class="summary" id="summary">Önce bir video yükleyin.</div>
    <button class="btn" id="startBtn" disabled>🚀 İşlemi Başlat</button>
  </section>

  <section class="card">
    <h2><span class="num">3</span>İşler ve indirmeler</h2>
    <div id="jobs"><div class="jobs-empty">Henüz iş yok.</div></div>
  </section>

  <footer>
    İşlem sırasında Colab sekmesini açık tutun. Bu sayfayı kapatsanız bile işler arka planda devam eder; bu linke tekrar girip indirebilirsiniz.<br>
    Upscale: Real-ESRGAN (xinntao) · Interpolasyon: RIFE 4.26 (hzwer/Practical-RIFE) · v__VERSION__
  </footer>
</div>
<div class="toast" id="toast"></div>
<script>
const OPTS = {
  res: {t:"Çözünürlük", items:[["orig","Orijinal","çözünürlüğü koru"],["1080","1080p","Full HD"],["1440","2K","1440p"],["2160","4K","2160p · Ultra HD"]]},
  fps: {t:"Kare hızı (interpolasyon)", items:[["orig","Orijinal","değiştirme"],["60","60 FPS","akıcı"],["90","90 FPS","çok akıcı"],["120","120 FPS","ultra akıcı"]]},
  model: {t:"AI modeli", items:[["kalite","💎 Maks. kalite","RealESRGAN x4plus · önerilen"],["hizli","⚡ Hızlı","uzun videolar için"],["anime","🎨 Anime","çizgi film / animasyon"]]},
  codec: {t:"Video formatı", items:[["h265","H.265 10-bit","en iyi kalite · bantlanma yok"],["h264","H.264","her cihazda oynar"]]},
  quality: {t:"Dosya kalitesi", items:[["maksimum","Maksimum","görsel olarak kayıpsız"],["yuksek","Yüksek","daha küçük dosya"],["dengeli","Dengeli","paylaşım için"]]},
};
const DEF = {res:"2160", fps:"60", model:"kalite", codec:"h265", quality:"maksimum"};
const RES = {orig:null, "1080":1080, "1440":1440, "2160":2160};
const STATUS = {queued:"Sırada", running:"İşleniyor", done:"Tamamlandı", error:"Hata", canceled:"İptal edildi"};
const $ = id => document.getElementById(id);
let upload = null, uploading = false, dlBase = null;

function store(k, v){ try{ if(v===undefined) return localStorage.getItem(k); localStorage.setItem(k, v); }catch(e){ return null; } }
function toast(msg){ const t=$("toast"); t.textContent=msg; t.classList.add("show"); clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"),3500); }
function fmtBytes(b){ if(!b && b!==0) return ""; const u=["B","KB","MB","GB","TB"]; let i=0; while(b>=1024&&i<4){b/=1024;i++;} return b.toFixed(i?1:0)+" "+u[i]; }
function fmtTime(s){ if(s==null||!isFinite(s)) return "—"; s=Math.max(0,Math.round(s)); const h=Math.floor(s/3600), m=Math.floor(s%3600/60), x=s%60; return h? `${h} sa ${m} dk` : m? `${m} dk ${x} sn` : `${x} sn`; }
function ev(v){ v=Math.round(v); return Math.max(2, v - v%2); }
function plan(w,h,res){ const t=RES[res]; if(!t) return [ev(w),ev(h),false]; const k=t/Math.min(w,h); return [ev(w*k),ev(h*k),k>1.0001]; }
function val(name){ const el=document.querySelector(`input[name=${name}]:checked`); return el?el.value:DEF[name]; }
async function api(url, opt={}){
  opt.headers = Object.assign({"ngrok-skip-browser-warning":"1"}, opt.headers||{});
  const r = await fetch(url, opt);
  let d = null; try{ d = await r.json(); }catch(e){}
  if(!r.ok) throw new Error((d&&d.detail)||("HTTP "+r.status));
  return d;
}

function buildOpts(){
  const box=$("opts");
  for(const [name,g] of Object.entries(OPTS)){
    const saved=store("v2_"+name);
    const grp=document.createElement("div"); grp.className="group";
    const t=document.createElement("div"); t.className="t"; t.textContent=g.t; grp.appendChild(t);
    const seg=document.createElement("div"); seg.className="seg";
    for(const [v,l,s] of g.items){
      const id=`o_${name}_${v}`;
      const inp=document.createElement("input"); inp.type="radio"; inp.name=name; inp.value=v; inp.id=id;
      inp.checked = (saved||DEF[name])===v;
      inp.onchange=()=>{ store("v2_"+name, v); updateSummary(); };
      const lab=document.createElement("label"); lab.htmlFor=id; lab.textContent=l;
      const sm=document.createElement("small"); sm.textContent=s; lab.appendChild(sm);
      seg.append(inp,lab);
    }
    grp.appendChild(seg); box.appendChild(grp);
  }
  for(const id of ["scene","dedup"]){
    const s=store("v2_"+id); if(s!==null) $(id).checked = s==="1";
    $(id).onchange=()=>store("v2_"+id, $(id).checked?"1":"0");
  }
}

function updateSummary(){
  const s=$("summary");
  if(!upload||!upload.info){ s.textContent = uploading ? "Video yükleniyor…" : "Önce bir video yükleyin."; $("startBtn").disabled=true; return; }
  const i=upload.info, res=val("res"), fps=val("fps"), model=val("model");
  const [w,h,ai]=plan(i.width,i.height,res);
  const ofps = fps==="orig" ? i.fps : +fps;
  const frames = Math.max(1, Math.round((i.frames||1) * ofps / i.fps));
  s.innerHTML="";
  const a=document.createElement("b"); a.textContent=`${i.width}×${i.height} @ ${i.fps} FPS`;
  const ar=document.createElement("span"); ar.className="arrow"; ar.textContent="→";
  const b=document.createElement("b"); b.textContent=`${w}×${h} @ ${+(+ofps).toFixed(2)} FPS`;
  const c=document.createElement("div"); c.className="muted";
  const parts=[];
  parts.push(ai ? "AI ile büyütülecek" : (res==="orig"||(w===i.width&&h===i.height) ? "çözünürlük aynı kalacak" : "kaynak zaten hedeften büyük: yalnızca boyutlandırılacak"));
  parts.push(ofps>i.fps*1.01 ? `RIFE 4.26 ile ${Math.round(ofps/i.fps*10)/10}× ara kare` : "interpolasyon yok");
  parts.push(`≈ ${frames.toLocaleString("tr-TR")} kare`);
  c.textContent=parts.join(" · ");
  s.append(a,ar,b,c);
  if(ofps<i.fps*0.99){ const n=document.createElement("div"); n.className="note"; n.textContent="Seçilen FPS kaynaktan düşük: bazı kareler atlanacak."; s.appendChild(n); }
  if(val("codec")==="h265"){ const n=document.createElement("div"); n.className="muted"; n.style.marginTop="6px"; n.textContent="ℹ️ H.265 10-bit: telefon, TV, VLC ve modern tarayıcılarda oynar. Eski cihazlar için H.264 seçin."; s.appendChild(n); }
  $("startBtn").disabled=false;
}

async function uploadFile(file){
  if(uploading) return toast("Bir yükleme zaten sürüyor.");
  uploading=true; upload=null; updateSummary();
  $("fileBox").style.display="block"; $("fName").textContent=file.name; $("fInfo").innerHTML=""; $("fBar").style.width="0%";
  $("fStat").textContent="Hazırlanıyor…";
  try{
    const init=await api("/api/upload/init",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:file.name,size:file.size})});
    const CH=init.chunk_size, total=init.chunks; let next=0, sent=0; const t0=performance.now();
    const showProg=()=>{ const p=sent/file.size, el=(performance.now()-t0)/1000, sp=sent/Math.max(el,.001);
      $("fBar").style.width=(p*100).toFixed(1)+"%";
      $("fStat").textContent=`%${(p*100).toFixed(0)} · ${fmtBytes(sent)} / ${fmtBytes(file.size)} · ${fmtBytes(sp)}/sn · kalan ${fmtTime((file.size-sent)/Math.max(sp,1))}`; };
    async function worker(){
      while(true){
        const i=next++; if(i>=total) return;
        const blob=file.slice(i*CH, Math.min(file.size,(i+1)*CH));
        for(let a=0;;a++){
          try{ const r=await fetch(`/api/upload/${init.id}/${i}`,{method:"PUT",body:blob,headers:{"ngrok-skip-browser-warning":"1"}}); if(r.ok) break; throw new Error("HTTP "+r.status); }
          catch(e){ if(a>=6) throw e; await new Promise(r=>setTimeout(r,1000*2**a)); }
        }
        sent+=blob.size; showProg();
      }
    }
    await Promise.all([worker(),worker(),worker()]);
    $("fStat").textContent="Video inceleniyor…";
    const res=await api(`/api/upload/${init.id}/complete`,{method:"POST"});
    upload=res; const i=res.info;
    $("fBar").style.width="100%";
    $("fStat").textContent=`✅ Yüklendi · ${fmtBytes(file.size)}`;
    const pills=[["Çözünürlük",`${i.width}×${i.height}`],["FPS",i.fps],["Süre",fmtTime(i.duration)],["Kare",(i.frames||0).toLocaleString("tr-TR")],["Codec",i.codec+(i.bits>8?` ${i.bits}-bit`:"")],["Ses",i.has_audio?(i.acodec||"var"):"yok"]];
    if(i.hdr) pills.push(["HDR","SDR'ye çevrilecek"]);
    for(const [k,v] of pills){ const p=document.createElement("span"); p.className="pill"; p.append(k+": "); const b=document.createElement("b"); b.textContent=v; p.appendChild(b); $("fInfo").appendChild(p); }
  }catch(e){ $("fStat").textContent="❌ "+e.message; toast("Yükleme başarısız: "+e.message); }
  uploading=false; updateSummary();
}

async function startJob(){
  if(!upload) return;
  const body={upload_id:upload.id,res:val("res"),fps:val("fps"),model:val("model"),codec:val("codec"),quality:val("quality"),scene:$("scene").checked,dedup:$("dedup").checked};
  $("startBtn").disabled=true;
  try{ await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); toast("✅ İş kuyruğa eklendi. Aynı videoyu farklı ayarlarla da başlatabilirsiniz."); refresh(); }
  catch(e){ toast("Başlatılamadı: "+e.message); }
  setTimeout(()=>{$("startBtn").disabled=!upload;},800);
}

function renderJobs(data){
  if(data && "dl_base" in data) dlBase = data.dl_base;
  const box=$("jobs"); const jobs=(data&&data.jobs)||[];
  if(!jobs.length){ box.innerHTML='<div class="jobs-empty">Henüz iş yok.</div>'; return; }
  box.innerHTML="";
  for(const j of jobs){
    const d=document.createElement("div"); d.className="job";
    const top=document.createElement("div"); top.className="top";
    const left=document.createElement("div");
    const tt=document.createElement("div"); tt.className="title"; tt.textContent=j.name;
    const sub=document.createElement("div"); sub.className="muted"; sub.textContent=`${j.in_desc}  →  ${j.out_desc}`;
    left.append(tt,sub);
    const bd=document.createElement("span"); bd.className="badge b-"+j.status;
    bd.textContent = j.status==="queued" && j.queue_pos ? `Sırada (${j.queue_pos})` : STATUS[j.status]||j.status;
    top.append(left,bd); d.appendChild(top);
    if(j.status==="running"||j.status==="queued"){
      const bar=document.createElement("div"); bar.className="bar"; const i=document.createElement("i"); i.style.width=((j.progress||0)*100).toFixed(1)+"%"; bar.appendChild(i); d.appendChild(bar);
      const st=document.createElement("div"); st.className="stats";
      const a=document.createElement("span"); a.textContent=`${j.stage||""}${j.status==="running"?" · %"+((j.progress||0)*100).toFixed(1):""}`;
      const b=document.createElement("span");
      if(j.status==="running" && j.done) b.textContent=`${j.done.toLocaleString("tr-TR")} / ${(j.total||0).toLocaleString("tr-TR")} kare · ${j.fps_proc||0} kare/sn · kalan ${fmtTime(j.eta)}`;
      st.append(a,b); d.appendChild(st);
    }
    if(j.engine&&j.engine.length){ const e=document.createElement("div"); e.className="eng"; e.textContent=j.engine.join(" · "); d.appendChild(e); }
    if(j.status==="done"){ const st=document.createElement("div"); st.className="stats"; const a=document.createElement("span"); a.textContent=`📦 ${fmtBytes(j.out_size)} · ⏱️ ${fmtTime((j.finished||0)-(j.started||0))}${j.drive?" · ☁️ Drive'a kaydedildi":""}`; st.appendChild(a); d.appendChild(st); }
    if(j.status==="error"){ const e=document.createElement("div"); e.className="errm"; e.textContent=j.error||"Bilinmeyen hata"; d.appendChild(e); }
    const acts=document.createElement("div"); acts.className="acts";
    if(j.status==="done"){
      const path=`/api/download/${j.id}?t=${j.dl_token||""}`;
      const a=document.createElement("a"); a.className="btn sm"; a.href=(dlBase||"")+path; a.textContent="⬇️ İndir"; a.setAttribute("download", j.out_name||""); acts.appendChild(a);
      if(dlBase){ const alt=document.createElement("a"); alt.className="alt"; alt.href=path; alt.textContent="alternatif link"; alt.setAttribute("download", j.out_name||""); acts.appendChild(alt); }
    }
    const del=document.createElement("button"); del.className="btn sm red";
    del.textContent = (j.status==="running"||j.status==="queued") ? "✖ İptal et" : "🗑 Sil";
    del.onclick=async()=>{ if(!confirm(j.status==="running"||j.status==="queued"?"Bu iş iptal edilsin mi?":"Bu iş ve çıktı dosyası silinsin mi?")) return; try{ await api(`/api/jobs/${j.id}`,{method:"DELETE"}); refresh(); }catch(e){ toast(e.message); } };
    acts.appendChild(del); d.appendChild(acts);
    box.appendChild(d);
  }
}

async function refresh(){ try{ renderJobs(await api("/api/jobs")); }catch(e){} }
function connectEvents(){
  let errs=0, polling=null;
  const startPoll=()=>{ if(!polling) polling=setInterval(refresh,3000); };
  if(!window.EventSource){ startPoll(); return; }
  const es=new EventSource("/api/events");
  es.onmessage=e=>{ errs=0; try{ renderJobs(JSON.parse(e.data)); }catch(x){} };
  es.onerror=()=>{ if(++errs>=4){ es.close(); startPoll(); } };
}
async function loadInfo(){
  try{
    const i=await api("/api/info");
    dlBase=i.dl_base||null;
    $("gpuTxt").innerHTML=""; const b=document.createElement("b");
    if(i.gpu){ b.textContent=i.gpu; $("gpuTxt").append(b, ` · ${i.vram_gb} GB${i.nvenc?" · NVENC":""}`); }
    else { $("gpuDot").classList.add("warn"); b.textContent="GPU yok"; $("gpuTxt").append(b," · çok yavaş çalışır"); }
  }catch(e){ $("gpuDot").classList.add("warn"); $("gpuTxt").textContent=e.message.includes("401")?"Şifre gerekli":"Sunucuya ulaşılamadı"; }
}

buildOpts(); updateSummary(); loadInfo(); refresh(); connectEvents();
const drop=$("drop"), fin=$("fileIn");
drop.onclick=()=>fin.click();
drop.onkeydown=e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); fin.click(); } };
fin.onchange=()=>{ if(fin.files[0]) uploadFile(fin.files[0]); fin.value=""; };
["dragenter","dragover"].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.add("over");}));
["dragleave","drop"].forEach(n=>drop.addEventListener(n,e=>{e.preventDefault();drop.classList.remove("over");}));
drop.addEventListener("drop",e=>{ const f=e.dataTransfer.files[0]; if(f) uploadFile(f); });
$("startBtn").onclick=startJob;
</script>
</body>
</html>
"""


class BasicAuth:
    """Saf ASGI şifre katmanı (SSE/indirme dahil her isteği korur; imzalı indirme linkleri hariç)."""

    def __init__(self, app, password):
        self.app, self.password = app, password

    def _signed_download(self, scope):
        m = re.fullmatch(r"/api/download/([0-9a-f]+)", scope.get("path", ""))
        if not m:
            return False
        t = urllib.parse.parse_qs(scope.get("query_string", b"").decode("latin-1")).get("t", [""])[0]
        return hmac.compare_digest(t, dl_token(m.group(1)))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self._signed_download(scope):
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        ok = False
        if header[:6].lower() == b"basic ":
            try:
                raw = base64.b64decode(header[6:])
                try:
                    cred = raw.decode("utf-8")
                except UnicodeDecodeError:
                    cred = raw.decode("latin-1")
                ok = secrets.compare_digest(cred.split(":", 1)[-1].encode(), self.password.encode())
            except ValueError:
                ok = False
        if ok:
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"www-authenticate", b'Basic realm="AI Upscaler", charset="UTF-8"'),
                                (b"content-type", b"text/plain; charset=utf-8")]})
        await send({"type": "http.response.body", "body": "Şifre gerekli".encode()})


def create_app(stop_event):
    import asyncio

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    if SIFRE:
        app.add_middleware(BasicAuth, password=SIFRE)
    page = PAGE_HTML.replace("__VERSION__", VERSION)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    async def info():
        return {"gpu": GPU_NAME, "vram_gb": VRAM_GB, "nvenc": NVENC["h264"] or NVENC["hevc"], "version": VERSION,
                "drive": bool(DRIVE_DIR), "dl_base": DL_STATE["base"]}

    @app.post("/api/upload/init")
    async def upload_init(req: Request):
        d = await req.json()
        size = int(d.get("size") or 0)
        name = str(d.get("name") or "video.mp4")[:200]
        if size <= 0:
            raise HTTPException(400, "Dosya boş.")
        if size > shutil.disk_usage(BASE).free - 3 * 1024 ** 3:
            raise HTTPException(507, "Colab diskinde yeterli yer yok.")
        uid = uuid.uuid4().hex[:12]
        ext = re.sub(r"[^\w.]", "", Path(name).suffix)[:10] or ".mp4"
        path = UP_DIR / f"{uid}{ext}"
        with open(path, "wb") as f:
            f.truncate(size)
        UPLOADS[uid] = {"id": uid, "name": name, "size": size, "path": str(path),
                        "chunks": max(1, math.ceil(size / CHUNK)), "received": set(), "info": None}
        return {"id": uid, "chunk_size": CHUNK, "chunks": UPLOADS[uid]["chunks"]}

    def _pwrite(path, offset, data):
        fd = os.open(path, os.O_WRONLY)
        try:
            mv = memoryview(data)
            while mv:
                n = os.pwrite(fd, mv, offset)
                mv, offset = mv[n:], offset + n
        finally:
            os.close(fd)

    @app.put("/api/upload/{uid}/{idx}")
    async def upload_chunk(uid: str, idx: int, req: Request):
        up = UPLOADS.get(uid)
        if not up:
            raise HTTPException(404, "Yükleme bulunamadı.")
        if not 0 <= idx < up["chunks"]:
            raise HTTPException(400, "Geçersiz parça.")
        expected = min(CHUNK, up["size"] - idx * CHUNK)
        body = bytearray()
        async for part in req.stream():
            body += part
            if len(body) > expected:
                raise HTTPException(413, "Parça çok büyük.")
        if len(body) != expected:
            raise HTTPException(400, "Parça eksik geldi.")
        await run_in_threadpool(_pwrite, up["path"], idx * CHUNK, body)
        up["received"].add(idx)
        return {"ok": True}

    @app.post("/api/upload/{uid}/complete")
    async def upload_complete(uid: str):
        up = UPLOADS.get(uid)
        if not up:
            raise HTTPException(404, "Yükleme bulunamadı.")
        missing = up["chunks"] - len(up["received"])
        if missing:
            raise HTTPException(400, f"{missing} parça eksik.")
        try:
            up["info"] = await run_in_threadpool(probe_video, up["path"])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Video okunamadı: {e}") from e
        log(f"Yüklendi: {up['name']} ({up['size'] / 1024 ** 2:.1f} MB, {up['info']['width']}x{up['info']['height']}"
            f" @ {up['info']['fps']} fps)")
        return {"id": uid, "name": up["name"], "info": up["info"]}

    @app.post("/api/jobs")
    async def create_job(req: Request):
        d = await req.json()
        up = UPLOADS.get(str(d.get("upload_id")))
        if not up or not up["info"]:
            raise HTTPException(404, "Önce videoyu yükleyin.")
        opts = {"res": str(d.get("res")), "fps": str(d.get("fps")), "model": str(d.get("model")),
                "codec": str(d.get("codec")), "quality": str(d.get("quality")),
                "scene": bool(d.get("scene", True)), "dedup": bool(d.get("dedup", True))}
        if (opts["res"] not in RES_TARGETS or opts["fps"] not in FPS_CHOICES or opts["model"] not in MODEL_CHOICES
                or opts["codec"] not in CODEC_CHOICES or opts["quality"] not in QUALITY_CHOICES):
            raise HTTPException(400, "Geçersiz ayar.")
        i = up["info"]
        w, h, _ = plan_size(i["width"], i["height"], opts["res"])
        ofps = i["fps"] if opts["fps"] == "orig" else int(opts["fps"])
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "name": up["name"], "path": up["path"], "info": dict(i), "opts": opts, "status": "queued",
               "stage": "Sırada bekliyor", "progress": 0.0, "created": time.time(), "engine": [],
               "in_desc": f"{i['width']}×{i['height']} @ {i['fps']:g} FPS",
               "out_desc": f"{w}×{h} @ {ofps:g} FPS · {'H.265 10-bit' if opts['codec'] == 'h265' else 'H.264'}"}
        with JLOCK:
            JOBS[jid] = job
            STATE["version"] += 1
        JOBQ.put(jid)
        log(f"[{jid}] Kuyruğa eklendi: {up['name']} → {job['out_desc']}")
        return job_public(job)

    @app.get("/api/jobs")
    async def list_jobs():
        return snapshot()

    @app.delete("/api/jobs/{jid}")
    async def delete_job(jid: str):
        job = JOBS.get(jid)
        if not job:
            raise HTTPException(404, "İş bulunamadı.")
        if job["status"] in ("queued", "running"):
            job["cancel"] = True
            if job["status"] == "queued":
                job_set(job, status="canceled", stage="İptal edildi", finished=time.time())
            return {"ok": True}
        if job.get("out_path"):
            Path(job["out_path"]).unlink(missing_ok=True)
        with JLOCK:
            JOBS.pop(jid, None)
            STATE["version"] += 1
        return {"ok": True}

    @app.get("/api/download/{jid}")
    async def download_out(jid: str):
        job = JOBS.get(jid)
        if not job or job.get("status") != "done" or not Path(job.get("out_path", "")).exists():
            raise HTTPException(404, "Dosya bulunamadı.")
        return FileResponse(job["out_path"], media_type="video/mp4", filename=job["out_name"])

    @app.get("/api/events")
    async def events(req: Request):
        async def gen():
            last, last_sent = -1, 0.0
            yield "retry: 3000\n\n"
            while not stop_event.is_set():  # kapanışta açık akışlar sunucuyu bekletmesin
                if await req.is_disconnected():
                    break
                now = time.time()
                if STATE["version"] != last:
                    data = snapshot()
                    last, last_sent = data["version"], now
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif now - last_sent > 15:
                    last_sent = now
                    yield ": ping\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


# =============================================================================
#  Tünel (sabit link) ve başlatma
# =============================================================================
def start_ngrok(port):
    pip_install("pyngrok")
    from pyngrok import conf, ngrok

    token = NGROK_TOKEN.strip()
    if not token:
        raise RuntimeError("NGROK_TOKEN boş. https://dashboard.ngrok.com/get-started/your-authtoken adresinden alın "
                           "veya TUNEL'i 'cloudflare' seçin.")
    conf.get_default().auth_token = token
    try:
        ngrok.kill()
    except Exception:  # noqa: BLE001
        pass
    dom = re.sub(r"^https?://", "", NGROK_DOMAIN.strip()).strip("/")
    attempts = [{"url": f"https://{dom}"}, {"domain": dom}] if dom else [{}]
    last = None
    for extra in attempts:
        try:
            return ngrok.connect(str(port), **extra).public_url
        except Exception as e:  # noqa: BLE001
            last = e
    msg = str(last)
    if "ERR_NGROK_108" in msg or "ERR_NGROK_334" in msg or "already online" in msg:
        msg += ("\n👉 Bu ngrok hesabı/adresi başka bir yerde (ör. başka bir Colab sekmesinde) hâlâ açık. "
                "Oradaki hücreyi durdurun veya https://dashboard.ngrok.com/agents sayfasından oturumu kapatın.")
    elif "ERR_NGROK_4018" in msg or "authtoken" in msg.lower():
        msg += "\n👉 NGROK_TOKEN hatalı görünüyor."
    raise RuntimeError(msg)


def start_cloudflare(port, rt):
    exe = download("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                   BIN_DIR / "cloudflared", "cloudflared")
    exe.chmod(0o755)
    p = subprocess.Popen([str(exe), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rt.procs.append(p)
    found = queue.Queue()

    def pump():
        for line in p.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                found.put(m.group(0))

    threading.Thread(target=pump, daemon=True).start()
    try:
        return found.get(timeout=90)
    except queue.Empty:
        raise RuntimeError("Cloudflare tüneli açılamadı.") from None


def start_download_tunnel(rt):
    """ngrok ücretsiz planda aylık ~1 GB indirme sınırı var; büyük videolar ayrı bir Cloudflare tünelinden iner."""
    try:
        DL_STATE["base"] = start_cloudflare(PORT, rt)
        STATE["version"] += 1
        log(f"⚡ Sınırsız indirme tüneli hazır: {DL_STATE['base']} (İndir butonu bunu otomatik kullanır)")
    except Exception as e:  # noqa: BLE001
        log(f"Ek indirme tüneli açılamadı ({e}); indirmeler ana link üzerinden yapılacak.")


class Runtime:
    def __init__(self):
        self.server, self.thread, self.worker, self.procs = None, None, None, []
        self.stop_event, self.uses_ngrok, self.jobs = threading.Event(), False, JOBS

    def stop(self):
        self.stop_event.set()
        for job in list(self.jobs.values()):
            if job.get("status") in ("queued", "running"):
                job["cancel"] = True
        if self.server:
            self.server.should_exit = True
        for p in self.procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        if self.uses_ngrok:
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except Exception:  # noqa: BLE001
                pass
        for t in (self.thread, self.worker):
            if t:
                t.join(timeout=8)


def detect_tools():
    global HAS_ZSCALE
    if not shutil.which("ffmpeg"):
        log("ffmpeg kuruluyor…")
        subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
    HAS_ZSCALE = " zscale " in run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    if HALF:
        NVENC["h264"] = encoder_works(["-c:v", "h264_nvenc"], "yuv420p", "yuv420p", 320, 240, Fraction(30))
        NVENC["hevc"] = encoder_works(["-c:v", "hevc_nvenc", "-profile:v", "main10"], "p010le", "p010le", 320, 240,
                                      Fraction(30))


def main():
    global DRIVE_DIR
    old = getattr(builtins, "_AI_UPSCALER_RT", None)
    if old:
        log("Önceki oturum kapatılıyor…")
        old.stop()
        time.sleep(1.5)
    rt = Runtime()
    builtins._AI_UPSCALER_RT = rt

    print("=" * 64 + f"\n🎬  AI Video Upscaler v{VERSION}\n" + "=" * 64, flush=True)
    for d in (UP_DIR, OUT_DIR, WORK_DIR, MODEL_DIR, BIN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for f in WORK_DIR.glob("*"):
        f.unlink(missing_ok=True)
    pip_install("fastapi", "uvicorn")

    if HALF:
        log(f"GPU: {GPU_NAME} · {VRAM_GB} GB" + (" → büyük GPU profili (toplu RIFE, geniş model önbelleği)"
                                                 if BIG_GPU else ""))
    else:
        log("⚠️  GPU bulunamadı! Colab menüsü: Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 GPU. "
            "(GPU olmadan çok yavaş çalışır.)")
    detect_tools()
    log(f"ffmpeg hazır · NVENC H.264: {'✓' if NVENC['h264'] else '✗'} · NVENC H.265 10-bit: "
        f"{'✓' if NVENC['hevc'] else '✗'} · HDR ton eşleme: {'✓' if HAS_ZSCALE else '✗'}")

    if DRIVE_KAYDET and IN_COLAB:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            DRIVE_DIR = Path("/content/drive/MyDrive/AI_Upscaler")
            DRIVE_DIR.mkdir(parents=True, exist_ok=True)
            log(f"Çıktılar Drive'a da kaydedilecek: {DRIVE_DIR}")
        except Exception as e:  # noqa: BLE001
            log(f"Drive bağlanamadı ({e}); yalnızca Colab diskine kaydedilecek.")

    # Varsayılan (maksimum kalite) modelleri önceden indirip yükle; diğerleri ilk kullanımda iner.
    MODELS.rife()
    for name in ("RealESRGAN_x4plus", "RealESRGAN_x2plus"):
        MODELS.sr(name)
    log("AI modelleri hazır (RealESRGAN x4plus/x2plus + RIFE 4.26)")

    import uvicorn

    class _Server(uvicorn.Server):
        def install_signal_handlers(self):  # iş parçacığında çalıştığımız için
            pass

    rt.server = _Server(uvicorn.Config(create_app(rt.stop_event), host="0.0.0.0", port=PORT, log_level="warning",
                                       timeout_keep_alive=30))
    rt.thread = threading.Thread(target=rt.server.run, daemon=True)
    rt.thread.start()
    for _ in range(100):
        if rt.server.started:
            break
        if not rt.thread.is_alive():
            raise RuntimeError(f"Web sunucusu başlatılamadı (port {PORT} dolu olabilir).")
        time.sleep(0.1)
    rt.worker = threading.Thread(target=worker_loop, args=(rt.stop_event,), daemon=True)
    rt.worker.start()
    log(f"Web sunucusu çalışıyor (port {PORT})")

    tunnel = TUNEL.strip().lower()
    url = f"http://127.0.0.1:{PORT}"
    try:
        if tunnel == "ngrok":
            rt.uses_ngrok = True
            url = start_ngrok(PORT)
            threading.Thread(target=start_download_tunnel, args=(rt,), daemon=True).start()
        elif tunnel == "cloudflare":
            url = start_cloudflare(PORT, rt)
    except Exception as e:  # noqa: BLE001
        log(f"❌ Tünel açılamadı: {e}")
        if tunnel == "ngrok":
            log("Cloudflare geçici tüneli deneniyor (link bu sefer değişken olacak)…")
            url = start_cloudflare(PORT, rt)

    print("\n" + "=" * 64, flush=True)
    print(f"🌐  WEB ARAYÜZÜ:  {url}", flush=True)
    if "ngrok" in url:
        print("    (Bu link sabittir; her çalıştırmada aynı kalır. İlk girişte ngrok uyarı sayfasında "
              "'Visit Site' butonuna basın.)", flush=True)
    if SIFRE:
        print("    🔒 Şifre korumalı — kullanıcı adı: herhangi bir şey, şifre: SIFRE alanındaki değer", flush=True)
    print("=" * 64 + "\nHücre çalıştığı sürece site açık kalır. Kapatmak için ■ durdur butonuna basın.\n", flush=True)
    if IN_COLAB:
        try:
            from IPython.display import HTML, display
            display(HTML(f'<a href="{url}" target="_blank" style="display:inline-block;padding:12px 20px;'
                         f'border-radius:12px;background:linear-gradient(135deg,#7c5cff,#20d3ee);color:#fff;'
                         f'font:600 16px system-ui;text-decoration:none">🎬 AI Video Upscaler\'ı aç → {url}</a>'))
        except Exception:  # noqa: BLE001
            pass

    try:
        while not rt.stop_event.is_set():
            try:
                print(_LOGQ.get(timeout=1.0), flush=True)
            except queue.Empty:
                pass
            if not rt.thread.is_alive():
                log("Web sunucusu durdu.")
                break
    except KeyboardInterrupt:
        print("\n⏹️  Kapatılıyor…", flush=True)
    finally:
        rt.stop()
        while not _LOGQ.empty():
            print(_LOGQ.get_nowait(), flush=True)
        print("Site kapatıldı. Tekrar açmak için hücreyi yeniden çalıştırın.", flush=True)


if __name__ == "__main__" or IN_COLAB:  # Colab'da her koşulda başlasın
    main()
