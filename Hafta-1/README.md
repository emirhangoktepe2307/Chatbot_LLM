# AI Eğitim Soruları Vektör Veritabanı

Bu proje, yapay zeka eğitim sorularını ChromaDB vektör veritabanında saklamak ve sorgulamak için kullanılır.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. ChromaDB'yi başlatın:
```bash
python chroma_setup.py
```

## Kullanım

`chroma_setup.py` dosyası şunları yapar:
1. ChromaDB istemcisini oluşturur
2. Yeni bir koleksiyon oluşturur
3. `data.json` dosyasındaki verileri vektör veritabanına yükler
4. Test sorgusu çalıştırır

## Veri Yapısı

Her belge şu bilgileri içerir:
- Soru
- Cevap
- Başlık (metadata)
- Anahtar kelimeler (metadata)

## Sorgulama

Vektör veritabanında semantik arama yapabilirsiniz. Örnek:
```python
results = collection.query(
    query_texts=["Yapay zeka nedir?"],
    n_results=3
)
``` 