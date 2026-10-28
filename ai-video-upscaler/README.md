# 🎬 AI Video Upscaler + FPS İnterpolasyon (Google Colab · Tek Hücre)

Google Colab'da **tek hücreyle** çalışan, **linki değişmeyen** bir web arayüzü üzerinden video yükleyip
yapay zekâ ile **1080p / 2K / 4K**'ya büyüten ve **60 / 90 / 120 FPS**'e ara kare üreten bot.

| | |
|---|---|
| **Büyütme (upscale)** | Real-ESRGAN — `RealESRGAN_x4plus` / `x2plus` (maks. kalite), `x4plus_anime_6B` (anime), `realesr-general-x4v3` (hızlı) |
| **Ara kare (interpolasyon)** | RIFE **4.26** (Practical-RIFE'ın en yeni modeli) + kopya kare ayıklama + sahne geçişi koruması |
| **Kodlama** | GPU'da NVENC **H.265 10-bit** (p7 + multipass + uzamsal/zamansal AQ) veya H.264 · NVENC yoksa otomatik CPU yedeği |
| **Ses** | Orijinal ses kayıpsız kopyalanır (gerekirse AAC 320k); çekim tarihi gibi meta veriler korunur |
| **Video ekleme** | Çoklu sürükle-bırak yükleme (büyük dosyalar parça parça) · **link ile ekleme** (YouTube, Google Drive paylaşım linki, doğrudan video linki) · **Drive'dan seçme** |
| **Önizleme** | İşlemeden önce **önce/sonra sürgülü karşılaştırma**, %100 detay görünümü, kare konumu seçimi |
| **Ekstralar** | Kırpma (başlangıç/bitiş) · **ağır çekim 2×/4×/8×** · aynı çözünürlükte AI iyileştirme · toplu işlem · tekrar dene |
| **Arayüz** | Canlı ilerleme/kalan süre, canlı GPU kullanımı/VRAM/sıcaklık ve boş disk, tarayıcı bildirimi, iş kuyruğu |
| **Kalıcılık** | Hücre yeniden çalıştırılsa da (aynı Colab oturumunda) yüklenen videolar ve biten işler listede kalır |
| **Link** | ngrok ücretsiz sabit adres → **her çalıştırmada aynı link** |

Hepsi ücretsiz araçlardır; ek bir ücretli hizmet gerekmez.

---

## 🚀 Kurulum (5 dakika)

### 1) ngrok (sabit link için, ücretsiz)
1. https://dashboard.ngrok.com/signup adresinden ücretsiz hesap açın.
2. **Your Authtoken** sayfasındaki token'ı kopyalayın: https://dashboard.ngrok.com/get-started/your-authtoken
3. **Domains** sayfasında hesabınıza tanımlı ücretsiz sabit adresi kopyalayın (ör. `benim-adim.ngrok-free.app`):
   https://dashboard.ngrok.com/domains

### 2) Colab (önerilen: kısa başlatıcı hücre)
1. https://colab.research.google.com → **Yeni not defteri**.
2. Menü: **Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 GPU** (T4 de çalışır, daha yavaştır).
3. [`colab_baslat.py`](colab_baslat.py) içeriğini (20 satır) tek bir hücreye yapıştırın. Bu hücre asıl kodu
   GitHub'dan indirip çalıştırır; uzun kodu elle kopyalamanız gerekmez.
4. Formda `NGROK_TOKEN` (ve isterseniz `NGROK_DOMAIN`) alanlarını doldurun, ▶ ile çalıştırın.
5. İlk saniyede `⏳ … yükleniyor` yazısı çıkar; modeller indikten sonra `🌐 WEB ARAYÜZÜ: https://…` linki görünür.
   İlk girişte ngrok uyarı sayfasında **Visit Site**'a basın.

**Hücre çalıştığı sürece site açıktır** — hücrenin dönmeye devam etmesi normaldir. Hücre birkaç saniyede
"bitti" görünüyorsa site kapanmış demektir; çıktıdaki hata mesajına bakın.

> Hazır not defteri: [AI_Video_Upscaler.ipynb](AI_Video_Upscaler.ipynb) — GitHub'da açıp **"Open in Colab"** ile
> doğrudan Colab'a aktarabilirsiniz (başlatıcı hücre ve L4 GPU ayarı hazır gelir).

**Alternatif — kodun tamamını yapıştırmak:** [`colab_upscaler.py`](colab_upscaler.py) dosyasını GitHub'da açıp
**Raw** → `Ctrl+A` → `Ctrl+C` (veya "Copy raw file" butonu) ile kopyalayın. Ekranda seçerek kopyalarsanız dosyanın
sonu eksik kalabilir; o zaman hücre **0 sn'de biter** ve hiçbir şey olmaz.

(Branch `master`'a birleştirildiyse `colab_baslat.py` içindeki linkte `claude/ai-video-upscaler-bot-ilz55b`
kısmını `master` yapın.)

---

## 🎬 Kullanım akışı

1. **Videoları ekleyin:** sürükleyip bırakın (birden fazla olabilir), bir link yapıştırın ya da
   **📁 Drive'dan seç** ile Drive'daki `AI_Upscaler/girdi` klasöründeki videoları seçin (`DRIVE_KAYDET` açıksa).
   Eklenen her video küçük resmiyle listelenir ve otomatik seçilir (☑).
2. **Ayarları seçin** ve **👁️ Önizleme** ile bir karenin önce/sonra halini sürgüyle karşılaştırın. Görüntüye çift
   tıklarsanız o bölgenin %100 detayını gösterir; "Kare konumu" sürgüsüyle videonun başka bir anını seçebilirsiniz.
3. Uzun bir videoda önce kısa bir parçayı denemek için **Başlangıç/Bitiş** girin (ör. `1:30` – `1:40`).
4. **🚀 Başlat** — seçili her video için ayrı bir iş kuyruğa eklenir. Bitince tarayıcı bildirimi gelir; **⬇️ İndir**.

## 🎛️ Arayüzdeki seçenekler

| Seçenek | Değerler | Açıklama |
|---|---|---|
| Çözünürlük | Orijinal · 1080p · 2K · 4K | Kısa kenar 1080/1440/2160 olacak şekilde, en-boy oranı korunur (dikey videolar da desteklenir). Kaynak hedeften büyükse AI atlanır, yalnızca boyutlandırılır. |
| Kare hızı | Orijinal · 60 · 90 · 120 | RIFE 4.26 ile ara kare. 24→60 gibi tam bölünmeyen oranlar da doğru zamanlamayla üretilir; süre ve ses senkronu korunur. |
| Ağır çekim | Kapalı · 2× · 4× · 8× | Ara karelerle gerçek ağır çekim (ör. 30 FPS videodan 4× → 120 kare/sn üretilip seçilen FPS'te oynatılır). Ses kaldırılır. |
| Başlangıç / Bitiş | `90`, `1:30`, `0:01:30.5` | Yalnızca bu aralık işlenir; ses aynı aralıktan alınır. |
| AI iyileştirme | açık/kapalı | Kaynak zaten hedef boyuttaysa da AI uygular (netleştirme/gürültü temizleme), sonra hedef boyuta indirir. |
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
  (Colab kapansa da kaybolmaz) ve `AI_Upscaler/girdi` klasörüne koyduğunuz videolar arayüzden seçilebilir.
  Büyük videoları kendi internetinizden yüklemek yerine Drive'a koyup buradan seçmek çok daha hızlıdır.
- **Link ile ekleme** yt-dlp kullanır (ilk kullanımda otomatik kurulur/güncellenir). Yalnızca kullanma hakkınız
  olan videoları işleyin.
- **YouTube "Sign in to confirm you're not a bot" hatası:** Colab sunucuları veri merkezinde olduğu için YouTube
  giriş doğrulaması isteyebilir. Arayüzdeki **🍪 YouTube çerezi** butonuna basın ve adımları izleyin:
  1. Tarayıcınıza **Get cookies.txt LOCALLY** (Chrome/Edge) veya **cookies.txt** (Firefox) eklentisini kurun ve
     gizli pencerede çalışmasına izin verin. ("LOCALLY" olmayan eski Chrome eklentisini kullanmayın.)
  2. Gizli/özel pencerede YouTube'a — mümkünse **yedek bir Google hesabıyla** — giriş yapın.
  3. Aynı sekmede `https://www.youtube.com/robots.txt` adresini açıp çerezleri **Export** edin, gizli pencereyi kapatın.
  4. Dosyayı arayüzden yükleyin, başarısız linkte **↻ Tekrar dene**'ye basın.

  yt-dlp ile kullanılan hesapların YouTube tarafından kısıtlanma riski vardır (yt-dlp de yedek hesap önerir).
  Çerez dosyası yalnızca o Colab oturumunda saklanır; çerez yüklediyseniz `SIFRE` kullanmanız önerilir.
- `SIFRE` doldurulursa site şifre ister (kullanıcı adı önemsiz). İndirme ve önizleme linkleri imzalıdır.

## 🛠️ Sorun giderme

| Sorun | Çözüm |
|---|---|
| Hücre 0 sn'de bitiyor, link çıkmıyor | Kod eksik yapıştırılmış. [`colab_baslat.py`](colab_baslat.py) başlatıcı hücresini kullanın. |
| `ERR_NGROK_108` / `ERR_NGROK_334` | Aynı ngrok hesabı başka bir yerde açık. Diğer Colab hücresini durdurun veya https://dashboard.ngrok.com/agents sayfasından kapatın. |
| `ERR_NGROK_4018` | Token hatalı; yeniden kopyalayın. |
| "GPU bulunamadı" | Çalışma zamanı türünü **L4 GPU** yapın ve hücreyi yeniden çalıştırın. |
| Video oynamıyor | H.265 desteklemeyen eski bir cihazdır; **H.264** seçin. |
| Çok yavaş | **⚡ Hızlı** modeli seçin veya hedef çözünürlüğü düşürün; önce **Başlangıç/Bitiş** ile kısa bir parça deneyin. |
| YouTube: "bot olmadığını doğrula" | **🍪 YouTube çerezi** ile cookies.txt yükleyip **↻ Tekrar dene** (yukarıdaki adımlar). Çerez yüklüyken de oluyorsa çerezleri gizli pencere yöntemiyle yeniden dışa aktarın. |
| Link indirilemedi (diğer) | Hata kartındaki Türkçe açıklamaya bakın; site Colab'ı engelliyorsa videoyu indirip yükleyin veya Drive'a koyun. |
| "Drive'dan seç" butonu yok | Colab formunda `DRIVE_KAYDET`'i açıp hücreyi yeniden çalıştırın. |

## Lisanslar
Real-ESRGAN (BSD-3-Clause, xinntao), Practical-RIFE / RIFE (MIT, hzwer). Model ağırlıkları ilk çalıştırmada
resmi yayın adreslerinden indirilir.
