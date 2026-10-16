# model-server

RTX 4090 (24 GB) üzerinde iki GGUF modeli tek adreste çalıştırır. OpenCode bu adrese bağlanır. İki model aynı anda karta sığmaz; istek hangi modeli istiyorsa diğeri kapanır, VRAM boşalır, yenisi yüklenir.

Adres: `http://127.0.0.1:8080/v1`

Modeller:

- `qwen3.8-27b` — Qwen3.8 27B Q4
- `muse-glimmer` — Muse Glimmer 30B Q4

Pencere 64K bağlam, cevap tavanı 8K. Bu kadarı proje içinde kod yazmaya yeter. Daha geniş pencere kartı doldurup selamı dakikalarca bekletir.

## Repoda olmayanlar

Model dosyaları (~36 GB) ve `runtime/` altındaki llama.cpp / CUDA kütüphaneleri git'e girmez. GitHub tek dosyada 100 MB sınırı koyuyor; CUDA kütüphaneleri bundan büyük. İkisi de yerelde kalır, silinmez.

## İlk kurulum

Python 3.11 ve güncel NVIDIA sürücüsü gerekir. `curl` Windows'ta hazır gelir.

```powershell
pip install httpx huggingface_hub filelock
python download_models.py
```

İndirme `models.json` içindeki dosyaları `models/` altına yazar. Yarıda kesilirse aynı komut kaldığı yerden devam eder. Token gerekirse:

```powershell
$env:HF_TOKEN = "hf_..."
python download_models.py
```

## Ayağa kaldırma

```powershell
python serve.py
```

İlk seferde llama.cpp CUDA paketini `runtime/` içine indirir. Model açılırken terminalde saniye sayacı ilerler:

```text
YUKLENIYOR  qwen3.8-27b  8s  model aciliyor, donmedi
HAZIR       qwen3.8-27b  ctx 65536  kv q8_0  http://127.0.0.1:8080/v1
```

`HAZIR` görünce OpenCode bağlanabilir. Sunucu açık kaldığı sürece kartta yaklaşık 22 GB dolu durur. Bu normal.

## Terminalde ne görürsün

Satır akıyorsa çalışıyordur. Saat ilerliyorsa donmamıştır.

| Satır | Anlamı |
| --- | --- |
| `YUKLENIYOR  8s` | Model açılıyor |
| `ISTEK` | OpenCode istek gönderdi |
| `PROMPT processing` | Prompt işleniyor, çubuk ve token/sn |
| `YAZIYOR` | Cevap yazılıyor |
| `BITTI` | Bitti, süre ve hız |
| `bosta` | Boşta, yeni istek bekliyor |
| `SES YOK` | 10 saniyedir sayaç ilerlemedi |
| `KAPANDI` | OpenCode bağlantıyı kesti, sunucu ayakta |

Kendi PowerShell penceresinde çalıştırırsan alt satır yerinde güncellenir. Tuşlar: `s` durum, `h` yardım, `q` kapat.

## OpenCode

`opencode.jsonc` dosyasını şuraya koy:

```text
%USERPROFILE%\.config\opencode\opencode.jsonc
```

OpenCode'u bir kez kapatıp aç. Sağlayıcı adı `local`, adres `http://127.0.0.1:8080/v1`.

Model değiştirmek için OpenCode içinde şunlardan birini seç:

- `local/qwen3.8-27b`
- `local/muse-glimmer`

Geçişte terminal `YUKLENIYOR` der. Eski model prosesi kapanır, VRAM boşalır, yeni model yüklenir. Bu 10–20 saniye sürer. İkisi birden açık kalmaz.

`serve.py` kapalıyken OpenCode bu adrese ulaşamaz. Önce sunucu, sonra OpenCode.

## Kapatma ve VRAM boşaltma

Sunucunun olduğu pencerede `Ctrl+C`. Gerçek konsoldaysa `q`. İkisi de `llama-server` prosesini kapatır ve karttaki belleği bırakır.

Pencere kapanmış ama kart hâlâ doluysa:

```powershell
taskkill /IM llama-server.exe /T /F
```

Kontrol:

```powershell
nvidia-smi
```

`llama-server` satırı yoksa ve bellek düşünce kart boştur. `python serve.py` tekrar ayağa kaldırır.

Model değiştirmek için sunucuyu kapatmana gerek yok. OpenCode'da diğer modeli seçmen yeterli. Sunucuyu sadece işin bitince kapat.
