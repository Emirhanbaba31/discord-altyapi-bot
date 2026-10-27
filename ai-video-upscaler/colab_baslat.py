#@title 🎬 AI Video Upscaler — Başlat (tek hücre) { display-mode: "form" }
#@markdown **1)** Çalışma zamanı → *Çalışma zamanı türünü değiştir* → **L4 GPU** · **2)** Alanları doldurun · **3)** ▶ ile çalıştırın.
#@markdown Hücre çalıştığı sürece site açık kalır (hücrenin dönmeye devam etmesi normaldir). Kapatmak için ■ durdur.
NGROK_TOKEN = ""  #@param {type:"string"}
NGROK_DOMAIN = ""  #@param {type:"string"}
SIFRE = ""  #@param {type:"string"}
DRIVE_KAYDET = False  #@param {type:"boolean"}
TUNEL = "ngrok"  #@param ["ngrok", "cloudflare"]

import os
import urllib.request

os.environ.update(UPSCALER_NGROK_TOKEN=NGROK_TOKEN.strip(), UPSCALER_NGROK_DOMAIN=NGROK_DOMAIN.strip(),
                  UPSCALER_SIFRE=SIFRE, UPSCALER_TUNEL=TUNEL, UPSCALER_DRIVE_KAYDET=str(DRIVE_KAYDET))
KOD_URL = ("https://raw.githubusercontent.com/Emirhanbaba31/discord-altyapi-bot/"
           "claude/ai-video-upscaler-bot-ilz55b/ai-video-upscaler/colab_upscaler.py")
KOD_YOLU = "/content/colab_upscaler.py" if os.path.isdir("/content") else "colab_upscaler.py"
print("⏳ Kod indiriliyor…", flush=True)
urllib.request.urlretrieve(KOD_URL, KOD_YOLU)
exec(compile(open(KOD_YOLU, encoding="utf-8").read(), KOD_YOLU, "exec"), {"__name__": "__main__"})
