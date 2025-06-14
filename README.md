# Makine Öğrenmesi Soru-Cevap Sistemi

Bu proje, makine öğrenmesi konularında soru-cevap yapabilen, öğrencilerin anlamadığı konuları tespit eden ve öğretmenlere bilgi veren bir yapay zeka sistemidir.

## Özellikler

- Makine öğrenmesi konularında soru-cevap
- ChromaDB ile benzer soruları bulma
- GPT entegrasyonu ile detaylı cevaplar
- Öğrenci takip sistemi
- Öğretmen bilgilendirme sistemi
- Öğrenme kaynakları önerisi

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/proje-adi.git
cd proje-adi
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
# veya
.\venv\Scripts\activate  # Windows için
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. `.env` dosyası oluşturun ve gerekli API anahtarlarını ekleyin:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Kullanım

Programı çalıştırmak için:
```bash
python Hafta-1/soru-cevap.py
```

## Gereksinimler

- Python 3.8+
- OpenAI API anahtarı
- Gerekli Python paketleri (requirements.txt dosyasında listelenmiştir)

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın. 