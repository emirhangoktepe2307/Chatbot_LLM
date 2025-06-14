import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from email_request import send_teacher_email_inform
import re
import chromadb
from chromadb.utils import embedding_functions

# .env dosyasını yükle
load_dotenv()

#API birden fazla kullanıcıda mevcut ve deadlock-fork süreç zarflarının uyarısını veriyordu çıktıyı sadeleştirdim burada
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
model_gpt=os.getenv("OPENAI_MODEL")

client = OpenAI(api_key=api_key)

# Sohbet geçmişini tutacak liste
conversation_history = []

# ChromaDB istemcisini RAM üzerinde oluştur
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()

# Koleksiyonu oluştur veya var olanı al
collection = chroma_client.get_or_create_collection(
    name="ai_egitim_sorulari",
    embedding_function=embedding_function
)

# JSON dosyasından verileri oku
with open('Hafta-1/data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# DataFrame'e çevir
df = pd.DataFrame(data)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Embedding'leri hesapla (Sorular üzerinden)
embeddings = model.encode(df["Sorular"].tolist())

def is_relevant(question, topic_keywords=None):
    if topic_keywords is None:
        topic_keywords = [
    # Genel kavramlar
    "makine öğrenmesi", "machine learning",
    "yapay zeka", "artificial intelligence",
    "model", "veri", "data", "dataset", "veri seti", "veri ön işleme", "preprocessing", "API",
    "tahmin", "prediction", "algoritma", "algorithm", "eğitim", "training",
    
    # Öğrenme türleri
    "denetimli", "denetimsiz", "supervised", "unsupervised",
    "pekiştirmeli", "reinforcement", "etiketli veri", "labeled data",
    "etiketsiz", "unlabeled", "kümeleme", "clustering",
    
    # Algoritmalar
    "regresyon", "lineer regresyon", "lojistik regresyon",
    "sınıflandırma", "karar ağacı", "decision tree",
    "random forest", "rf", "knn", "naive bayes",
    "ann", "yapay sinir ağı", "neural network", "cnn", "rnn",
    "svm", "destek vektör makineleri", "gradient boosting", "xgboost",
    "polinom regresyon", "ridge regresyon", "lasso regresyon",
    "elastic net", "ensemble", "bagging", "boosting",

    # Büyük modeller
    "llm", "large language model", "transformer", "bert", "gpt",
    "deep learning", "derin öğrenme", "nöron", "neuron",

    # Kütüphaneler / Framework'ler
    "tensorflow", "pytorch", "scikit-learn", "scikit", "Sklearn", "numpy", "np", "pandas", "pd", "matplotlib", "seaborn", "sns",
    "keras", "sequential", "model oluşturma", "model creation",

    # Veri işleme
    "encode", "encoding", "one-hot encoding", "label encoding",
    "ölçekleme", "scaling", "min-max scaling", "z-score", "robust scaling",
    "logaritmik dönüşüm", "feature engineering", "normalizasyon", "standardizasyon",
    "imputation", "eksik veri", "missing data", "veri temizleme", "data cleaning",

    # Değerlendirme metrikleri
    "r2", "f1 score", "rmse", "mae", "mse", "doğruluk", "accuracy", "precision", "recall",
    "confusion matrix", "karmaşıklık matrisi", "true positive", "false positive",

    # Diğer teknik terimler
    "confusion matrix", "cross-validation", "hyperparameter", "overfitting", "underfitting",
    "loss function", "activation function", "optimizer", "learning rate", "epoch", "batch size", "logloss",
    "veri bölme", "train test split", "veri seti", "dataset", "eğitim", "test",
    "görselleştirme", "visualization", "plot", "grafik", "chart"
    ]

    user_q = question.lower()
    return any(kw in user_q for kw in topic_keywords)

# Embedding'i DataFrame'e ekle
df["Embedding"] = embeddings.tolist()

def ask_gpt(question):
    try:
        print("\nGPT'ye soru soruluyor, lütfen bekleyiniz...")

        # Sohbet geçmişini mesajlara ekle
        messages = [
            {"role": "system", "content": "Sen deneyimli bir makine öğrenmesi uzmanısın. Sorulara Türkçe olarak, teknik detayları açıklayarak ve örnekler vererek cevap ver. Cevapların kısa ve öz olsun."}
        ]
        
        # Son 5 mesajı ekle (bağlamı korumak için)
        messages.extend(conversation_history[-5:] if conversation_history else [])
        
        # Yeni soruyu ekle
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=model_gpt,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.3
        )

        answer = response.choices[0].message.content
        
        # Sohbet geçmişine ekle
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})
        
        # Yeni soru-cevap çiftini ChromaDB'ye ekle
        collection.add(
            documents=[question],
            metadatas=[{
                'title': 'GPT Cevabı',
                'keywords': ', '.join(re.findall(r'\w+', question.lower()))
            }],
            ids=[f"gpt_{len(collection.get()['ids']) + 1}"]
        )
        
        return answer

    except Exception as e:
        return f"⚠️ GPT'ye soru sorulurken bir hata oluştu:\n{str(e)}"

def get_learning_resources(topic):
    """Konuyla ilgili öğretici web sitelerini önerir."""
    try:
        response = client.chat.completions.create(
            model=model_gpt,
            messages=[
                {"role": "system", "content": "Sen bir eğitim danışmanısın. Verilen konuyla ilgili en iyi öğretici web sitelerini, kursları ve kaynakları öner. Sadece Türkçe ve İngilizce kaynakları öner."},
                {"role": "user", "content": f"{topic} konusuyla ilgili öğretici web siteleri ve kaynaklar önerir misin?"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Kaynak önerileri alınırken bir hata oluştu:\n{str(e)}"
    
def keyword_match(question, df, min_overlap=2):
    question_tokens = re.findall(r'\w+', question.lower())

    best_match_index = -1
    max_overlap = 0

    for idx, row in df.iterrows():
        keywords = row.get("Keywords", [])
        if not keywords:
            continue
        overlap = len(set(question_tokens) & set([kw.lower() for kw in keywords]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match_index = idx

    if max_overlap >= min_overlap:
        return df.iloc[best_match_index]["Cevaplar"]
    else:
        return None

def is_topic_change(question):
    """Yeni bir konu başlığı olup olmadığını kontrol eder."""
    # Eğer sohbet geçmişi boşsa, yeni konu değildir
    if not conversation_history:
        return False
        
    # Son soruyu al
    last_question = conversation_history[-2]["content"] if len(conversation_history) >= 2 else ""
    
    # Son soru ve yeni soru arasında anahtar kelime benzerliği kontrol et
    last_tokens = set(re.findall(r'\w+', last_question.lower()))
    new_tokens = set(re.findall(r'\w+', question.lower()))
    
    # Ortak kelime sayısı
    common_words = len(last_tokens.intersection(new_tokens))
    
    # Eğer ortak kelime sayısı çok azsa veya hiç yoksa, yeni konu olarak kabul et
    return common_words < 2

def find_answer(question):
    # Yeni konu kontrolü
    if is_topic_change(question):
        print("\n🔄 Yeni bir konuya geçildiği tespit edildi. Sohbet geçmişi temizleniyor...")
        clear_conversation()
    
    # 0️⃣ İlk olarak konuyla alakalı mı kontrol et
    if not is_relevant(question):
        return "⚠️ Bu soru makine öğrenmesi veya yapay zeka konularıyla ilgili görünmüyor."

    # 1️⃣ Sonra keyword eşleşmesine bak
    keyword_based_answer = keyword_match(question, df)
    if keyword_based_answer:
        print("✅ Anahtar kelimelerle eşleşme bulundu.")
        return keyword_based_answer

    # 2️⃣ Eşleşme yoksa ChromaDB'de ara
    print("🔍 ChromaDB'de benzer sorular aranıyor...")
    results = collection.query(
        query_texts=[question],
        n_results=1
    )
    
    if results['documents'] and len(results['documents'][0]) > 0:
        print("✅ ChromaDB'de benzer soru bulundu.")
        return results['documents'][0][0]
        
    else:
        print("🤖 GPT'ye yönlendiriliyor...")
        return ask_gpt(question)

def clear_conversation():
    """Sohbet geçmişini temizler."""
    global conversation_history
    conversation_history = []
    print("\n✅ Sohbet geçmişi temizlendi.")

# Ana menü
def show_menu():
    print("\n=== ANA MENÜ ===")
    print("1. Soru Sor")
    print("2. Sohbet Geçmişini Temizle")
    print("3. Çıkış")
    return input("Seçiminiz (1-3): ")

# Ana döngü
while True:
    choice = show_menu()
    
    if choice == '1':
        user_question = input("\nSorunuzu yazın (Çıkmak için 'q' yazın): ")
        if user_question.lower() == 'q':
            continue
            
        answer = find_answer(user_question)
        print("\nCevap:", answer)
        if not answer.startswith("⚠️"):
            user_select = input("\nAnlamadıysan eğer daha detaylı bir cevap oluşturmamı ister misin? (y/n)")
            
            if user_select.lower() == 'y':
                new_answer = ask_gpt(user_question)
                print("\nDetaylı Cevap:", new_answer)
                user_control = input("\nBu detaylı açıklamayı anladınız mı? (y/n)")
                if user_control.lower() == 'n':
                    send_teacher_email_inform(user_question)
                    print("\nHocanıza Konuyla Alakalı Bilgilendirme Yapıldı. İletişime Geçene Kadar Alternatif Kaynak Önermemi İster Misin.(y/n)")
                    if input().lower() == 'y':
                        resources = get_learning_resources(user_question)
                        print("\n📚 Önerilen Kaynaklar:")
                        print(resources)
                
    elif choice == '2':
        clear_conversation()
        
    elif choice == '3':
        print("\nProgram sonlandırılıyor...")
        break
        
    else:
        print("\nGeçersiz seçim! Lütfen 1-3 arasında bir sayı girin.")
        