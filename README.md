# 📚 Türkçe Kitap Tavsiye Chatbot

Yapay zeka destekli kişiselleştirilmiş kitap danışmanı uygulaması, kullanıcılara akıllı kitap tavsiyeleri sunmak için tasarlanmıştır.

## 🌐 Deploy Link
- 

## ✨ Özellikler

- 📚 Kişisel kitap analizi ve öneriler
- 🎯 İlgi alanına göre kitap danışmanlığı
- 💬 Konuşmaya bağlı bağlam-farkındalı öneriler
- 🔍 Semantik arama (RAG teknolojisi)
- 🎨 Modern ve kullanıcı dostu arayüz
- ⚡ Gerçek zamanlı cevaplar

## 🛠️ Teknik Altyapı

- **Backend**: Python LangChain
- **Frontend**: Streamlit
- **AI**: Google Gemini 2.0 Flash
- **Vector Database**: Chroma
- **Embedding Model**: Google Text Embedding 004
- **Veri Seti**: YTÜ COSMOS Turkish Book Dataset (60,000+ kitap)

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.9+
- pip (Python paket yöneticisi)
- Google Gemini API anahtarı 

### 2. Repoyu Klonla

```bash
git clone https://github.com/yourusername/turkce-kitap-chatbot.git
cd turkce-kitap-chatbot
```

### 3. Sanal Ortam Oluştur ve Etkinleştir

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Gereksinimleri Yükle

```bash
pip install -r requirements.txt
```

### 5. Environment Dosyasını Yapılandır

Kök dizinde `.env` dosyası oluştur:

```env
GEMINI_API_KEY=your-api-key-here
```

### 6. Vector Database Oluştur

Chatbot'u ilk kez çalıştırmadan önce veritabanını oluştur:

```bash
python database.py
```

Bu işlem:
- `kitaplar_temiz.csv` dosyasından 60,000+ kitap verisi yükler
- Google Embedding modeli ile embeddings oluşturur
- Chroma vector database oluşturur
- Tahmini süre: 5-10 dakika

### 7. Uygulamayı Çalıştır

```bash
streamlit run chatbot_app.py
```

Tarayıcın otomatik açılacak: `http://localhost:8501`

## 🎯 Kullanım

1. **Kitap Kategorisi Seç:**
   - Başlangıçta sunulan seçeneklerden birini tıkla
   - Veya doğrudan sorunu yaz

2. **Soruları Yazarak Başla:**
   - "Tarih konusunda kitap öner"
   - "Çocuklar için felsefe kitabı arıyorum"
   - "Türk edebiyatı klasikleri neler?"
   - "Macera romanı arıyorum"

3. **Önerilen Kitaplardan Seç:**
   - Chatbot kitap adı, yazar ve neden önerdiğini gösterecek
   - Benzer kitaplar hakkında sorular sorabilirsin

4. **Sohbeti Devam Ettir:**
   - Geçmiş konuşmaları hatırlıyor
   - Daha detaylı tavsiyeleri kişiselleştiriyor

## 📁 Proje Yapısı

```
turkce-kitap-chatbot/
├── README.md                      # Proje belgelendirmesi
├── requirements.txt               # Python bağımlılıkları
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore dosyası
├── chatbot_app.py                 # Streamlit uygulaması (Ana dosya)
├── database.py                    # Vector DB oluşturucu
├── kitaplar_temiz.csv            # Kitap verisi (60,000+)
└── chroma_db/                     # Vector database (otomatik oluşturulur)
```

## 🔧 Yapılandırma

### `chatbot_app.py` içinde düzenlenebilir:

```python
MODEL_NAME = "gemini-2.0-flash"    # LLM modeli
DB_PATH = "./chroma_db"            # Database yolu
```

### `database.py` içinde düzenlenebilir:

```python
CSV_FILE = "kitaplar_temiz.csv"    # Kitap verisi dosyası
```

## 🐛 Sorun Giderme

### "GEMINI_API_KEY bulunamadı" hatası

```bash
# .env dosyasının var olduğunu kontrol et
# GEMINI_API_KEY=... yazılı olduğundan emin ol
```

### "Veritabanı yüklenemedi" hatası

```bash
# Eski database'i sil ve yeniden oluştur
Remove-Item -Recurse -Force chroma_db  # Windows PowerShell
python database.py
```

### Chatbot yavaş cevap veriyor

- Google API kotasını kontrol et
- İnternet bağlantısını kontrol et
- Vector database dosyasının mevcut olduğundan emin ol

## 📊 Dataset Bilgisi

- **Kaynak**: [YTÜ COSMOS Turkish Book Dataset](https://huggingface.co/datasets/ytu-ce-cosmos/turkce-kitap)
- **Toplam Kitap**: 60,000+
- **Veri Formatı**: Kitap adı, yazar, görsel
- **Veri Kalitesi**: Manuel olarak temizlenmiş ve doğrulanmış

## 🤝 Katkıda Bulun

Katkılarınızı bekliyoruz! Lütfen:

1. Repository'yi fork et
2. Feature branch oluştur (`git checkout -b feature/YeniOzellik`)
3. Değişiklikleri commit et (`git commit -m 'Add YeniOzellik'`)
4. Branch'i push et (`git push origin feature/YeniOzellik`)
5. Pull Request aç

## 📝 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bak

##  Teşekkürler

- YTÜ COSMOS ekibine dataset için
- Google Gemini ve Embedding modelleri için
- LangChain, Streamlit ve Chroma kütüphaneleri için
- Tüm Türkçe kitap severlere ilham için


---

**Not**: Bu proje Claude AI asistanının yardımıyla geliştirilmiştir.
