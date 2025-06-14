# Soru-Cevap Sistemi

Bu proje, makine öğrenmesi konularında soru-cevap yapabilen bir sistemdir. Sistem, önceden tanımlanmış sorulara cevap verebilir ve bilinmeyen sorular için GPT-3.5 Turbo modelini kullanır.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız vardır:

```bash
pip install -r requirements.txt
```

## Veri Yapısı

Sistem şu sütunları içeren bir veri seti kullanır:
- Ticket ID: Soru numarası
- Soru Başlığı: Sorunun kısa başlığı
- Sorular: Detaylı soru metni
- Cevaplar: Her soruya karşılık gelen cevap
- Embedding: Soru başlıklarının vektör temsili

## Kullanım

Programı çalıştırmak için:

```bash
python Hafta-1/soru-cevap.py
```

## Özellikler

- Sistem çok dilli destek sunar (paraphrase-multilingual-MiniLM-L12-v2 modeli kullanılmaktadır)
- Veri seti genişletilebilir, yeni soru-cevap çiftleri eklenebilir
- Sistem performansı, veri setinin kalitesi ve büyüklüğüne bağlıdır
- Bilinmeyen sorular için GPT-3.5 Turbo entegrasyonu
- Airtable veritabanı entegrasyonu
- Benzerlik skoruna dayalı akıllı cevap eşleştirme

## Çalışma Notları

- Sistem, soruları embedding vektörlerine dönüştürerek benzerlik hesaplaması yapar
- Benzerlik skoru 0.5'in altında olan sorular için GPT-3.5 Turbo kullanılır
- Veriler Airtable'da saklanabilir ve yönetilebilir
- Sistem, makine öğrenmesi konularında uzmanlaşmıştır
- 3 kategorili çalışmanın sebebi, derste işlenen örneğin 2 kategoriden oluşması ve fazla kategorili sistemin embedding hesaplamasını test etmek istememdir.

## API Anahtarı

Sistem, OpenAI API anahtarı gerektirir. API anahtarınızı çevresel değişken olarak ayarlayabilirsiniz:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Bu Hafta İzlediğim Yol Notları
-Sorularımı data.json dosyasına aktardım ve sistemin soruları json formatında okuması için optimizasyonlar yaptım.
-Sorularımı ve soruların embedding değerlerini Airtable ortamına aktardım.
-Soru sorabilmek için gerekli fonksiyonlarımı yazdım.
-data.json dosyasında bulunmayan soruların cevabı için 3.5 Turbo modelini entegre ettim ve sohbet rollerini belirledim.