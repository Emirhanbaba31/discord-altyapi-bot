# 🎬 AI Video Upscaler + FPS İnterpolasyon (Google Colab · Tek Hücre)

Google Colab'da **tek hücreyle** çalışan, **linki değişmeyen** bir web arayüzü üzerinden video yükleyip
yapay zekâ ile **1080p / 2K / 4K**'ya büyüten ve **60 / 90 / 120 FPS**'e ara kare üreten bot.

| | |
|---|---|
| **Büyütme (upscale)** | Real-ESRGAN — `RealESRGAN_x4plus` / `x2plus` (maks. kalite), `x4plus_anime_6B` (anime), `realesr-general-x4v3` (hızlı) |
| **Ara kare (interpolasyon)** | RIFE **4.26** (Practical-RIFE'ın en yeni modeli) + kopya kare ayıklama + sahne geçişi koruması |
| **Kodlama** | GPU'da NVENC **H.265 10-bit** (p7 + multipass + uzamsal/zamansal AQ) veya H.264 · NVENC yoksa otomatik CPU yedeği |
| **Ses** | Orijinal ses kayıpsız kopyalanır (gerekirse AAC 320k) |
| **Arayüz** | Sürükle-bırak yükleme (büyük dosyalar parça parça), canlı ilerleme/kalan süre, iş kuyruğu, indirme |
| **Link** | ngrok ücretsiz sabit adres → **her çalıştırmada aynı link** |

Hepsi ücretsiz araçlardır; ek bir ücretli hizmet gerekmez.

---

## 🚀 Kurulum (5 dakika)

### 1) ngrok (sabit link için, ücretsiz)
1. https://dashboard.ngrok.com/signup adresinden ücretsiz hesap açın.
2. **Your Authtoken** sayfasındaki token'ı kopyalayın: https://dashboard.ngrok.com/get-started/your-authtoken
3. **Domains** sayfasında hesabınıza tanımlı ücretsiz sabit adresi kopyalayın (ör. `benim-adim.ngrok-free.app`):
   https://dashboard.ngrok.com/domains

### 2) Colab
1. https://colab.research.google.com → **Yeni not defteri**.
2. Menü: **Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 GPU** (T4 de çalışır, daha yavaştır).
3. [`colab_upscaler.py`](colab_upscaler.py) dosyasının **tamamını** tek bir hücreye yapıştırın.
4. Sağdaki formda `NGROK_TOKEN` ve `NGROK_DOMAIN` alanlarını doldurun, ▶ ile çalıştırın.
5. Çıktıda görünen `🌐 WEB ARAYÜZÜ: https://…` linkine tıklayın. İlk girişte ngrok uyarı sayfasında **Visit Site**'a basın.

> Hazır not defteri: [AI_Video_Upscaler.ipynb](AI_Video_Upscaler.ipynb) — GitHub'da açıp **"Open in Colab"** ile
> doğrudan Colab'a aktarabilirsiniz. Tek hücre, GPU ayarı içinde hazır gelir.

**Alternatif (kodu yapıştırmadan, 3 satır):**

```python
%env UPSCALER_NGROK_TOKEN=buraya_token
%env UPSCALER_NGROK_DOMAIN=benim-adim.ngrok-free.app
!wget -q -O up.py https://raw.githubusercontent.com/Emirhanbaba31/discord-altyapi-bot/claude/ai-video-upscaler-bot-ilz55b/ai-video-upscaler/colab_upscaler.py && echo indirildi
%run up.py
```

(Branch `master`'a birleştirildiyse linkteki `claude/ai-video-upscaler-bot-ilz55b` kısmını `master` yapın.)

---

## 🎛️ Arayüzdeki seçenekler

| Seçenek | Değerler | Açıklama |
|---|---|---|
| Çözünürlük | Orijinal · 1080p · 2K · 4K | Kısa kenar 1080/1440/2160 olacak şekilde, en-boy oranı korunur (dikey videolar da desteklenir). Kaynak hedeften büyükse AI atlanır, yalnızca boyutlandırılır. |
| Kare hızı | Orijinal · 60 · 90 · 120 | RIFE 4.26 ile ara kare. 24→60 gibi tam bölünmeyen oranlar da doğru zamanlamayla üretilir; süre ve ses senkronu korunur. |
| AI modeli | 💎 Maks. kalite · ⚡ Hızlı · 🎨 Anime | Maks. kalite: ≤2× büyütmede `x2plus`, daha fazlasında `x4plus`. |
| Video formatı | H.265 10-bit · H.264 | 10-bit, büyütülmüş gökyüzü/gradyanlarda bantlanmayı önler. Eski cihazlar için H.264. |
| Dosya kalitesi | Maksimum · Yüksek · Dengeli | Maksimum = görsel olarak kayıpsız (büyük dosya). |
| Sahne geçişi koruması | açık/kapalı | Sahne kesmelerinde iki farklı görüntüyü karıştıran bulanık ara kare yerine temiz kesme. |
| Kopya kare ayıklama | açık/kapalı | Anime ("2'de/3'te bir çizim") ve takılan/VFR videolarda kopya kareler atlanıp gerçek kareler arasında interpolasyon yapılır → çok daha akıcı. |

Varsayılanlar en iyi çıktı içindir: **4K · 60 FPS · Maks. kalite · H.265 10-bit · Maksimum**.

---

## ⚡ L4 GPU için yapılan optimizasyonlar

- **fp16 + channels_last** (Ada tensör çekirdekleri) Real-ESRGAN, **TF32** RIFE; cuDNN otomatik ayar.
- 24 GB VRAM'de **toplu RIFE** (aynı kare çiftinden birden fazla ara kare tek seferde), karo bölme gerekmez;
  bellek yetmezse otomatik küçültme/karo bölmeye geçer.
- RGB→YUV (8-bit ve **10-bit P010**) dönüşümü GPU'da yapılır; CPU darboğazı olmaz.
- 10-bit kaynaklar (HEVC 10-bit, HDR) **16-bit** hassasiyetle okunur; HDR (HLG/PQ) otomatik SDR'ye ton eşlenir.
- Kopya karelerde AI tekrar çalıştırılmaz (hem hız hem titreşimsiz çıktı).
- NVENC: `p7` (en yüksek kalite ön ayarı), `multipass fullres`, 32 kare ileri bakış, uzamsal + zamansal AQ,
  B-kare referans. Seçenek desteklenmiyorsa otomatik olarak güvenli ayara düşer.

Tahmini hız (L4, ölçülmüş değil): 720p→4K **Maks. kalite** büyütme saniyede ~1–2 kaynak kare civarındadır;
uzun videolarda **⚡ Hızlı** model birkaç kat daha hızlıdır. Gerçek hız arayüzde "kare/sn" olarak canlı görünür.

---

## 📌 Bilmeniz gerekenler

- **Hücre çalıştığı sürece site açıktır.** Colab sekmesini açık tutun; ücretsiz/ücretli Colab oturum süresi
  sınırları geçerlidir. İş bittikten sonra sayfayı kapatıp linke tekrar girip indirebilirsiniz.
- **ngrok ücretsiz plan** aylık giden trafiği (~1 GB) sınırlar. Bu yüzden bot, büyük dosyaları **ayrı bir
  Cloudflare tüneli** üzerinden indirtir (İndir butonu bunu otomatik kullanır, sınırsızdır). Yüklemeler
  ngrok'un sınırına dahil değildir.
- `DRIVE_KAYDET` açılırsa biten videolar Google Drive'da `AI_Upscaler` klasörüne de kopyalanır
  (Colab kapansa da kaybolmaz).
- `SIFRE` doldurulursa site şifre ister (kullanıcı adı önemsiz). İndirme linkleri imzalıdır.

## 🛠️ Sorun giderme

| Sorun | Çözüm |
|---|---|
| `ERR_NGROK_108` / `ERR_NGROK_334` | Aynı ngrok hesabı başka bir yerde açık. Diğer Colab hücresini durdurun veya https://dashboard.ngrok.com/agents sayfasından kapatın. |
| `ERR_NGROK_4018` | Token hatalı; yeniden kopyalayın. |
| "GPU bulunamadı" | Çalışma zamanı türünü **L4 GPU** yapın ve hücreyi yeniden çalıştırın. |
| Video oynamıyor | H.265 desteklemeyen eski bir cihazdır; **H.264** seçin. |
| Çok yavaş | **⚡ Hızlı** modeli seçin veya hedef çözünürlüğü düşürün. |

## Lisanslar
Real-ESRGAN (BSD-3-Clause, xinntao), Practical-RIFE / RIFE (MIT, hzwer). Model ağırlıkları ilk çalıştırmada
resmi yayın adreslerinden indirilir.
