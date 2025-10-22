# -*- coding: utf-8 -*-
# Dosya: database.py
# Amaç: CSV'den kitap verisi yükle ve RAG veritabanı oluştur

import os
import shutil
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

# =====================================================================
# 1. YAPILANDIRMA
# =====================================================================

CSV_FILE = "kitaplar_temiz.csv"  # Temiz CSV dosyası
DB_PATH = "./chroma_db"

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("HATA: GEMINI_API_KEY ayarlanmadı!")
    exit()

genai.configure(api_key=api_key)

# =====================================================================
# 2. VERİTABANI OLUŞTURMA
# =====================================================================

def create_database():
    print("📚 Vector Database Oluşturuluyor...\n")
    
    # CSV Yükle
    print("1️⃣ CSV dosyası yükleniyor...")
    if not os.path.exists(CSV_FILE):
        print(f"❌ Hata: {CSV_FILE} bulunamadı!")
        print(f"   Lütfen önce 'clean_dataset.py' çalıştırın")
        return
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ {len(df)} kitap yüklendi.\n")
    except Exception as e:
        print(f"❌ Hata: {e}")
        return
    
    # Belge Oluştur
    print("2️⃣ Belge oluşturuluyor...")
    documents_list = []
    
    for idx, row in df.iterrows():
        if idx % 5000 == 0 and idx > 0:
            print(f"   {idx}/{len(df)} işlendi...")
        
        book_name = str(row['book_name']).strip()
        author = str(row['author']).strip()
        
        # RAG için optimize edilmiş belge
        doc_text = f"""
KİTAP ADI: {book_name}
YAZAR: {author}

Bu kitap "{book_name}" adlı eser, {author} tarafından yazılmıştır.
"""
        documents_list.append(doc_text.strip())
    
    print(f"✅ {len(documents_list)} belge hazırlandı.\n")
    
    # Metin Parçala
    print("3️⃣ Metinler parçalanıyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    documents = text_splitter.create_documents(documents_list)
    print(f"✅ {len(documents)} chunk oluşturuldu.\n")
    
    # Embedding
    print("4️⃣ Embedding modeli yükleniyor...")
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    print("✅ Embedding hazır.\n")
    
    # Eski DB Temizle
    print("5️⃣ Eski veritabanı temizleniyor...")
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"✅ Temizlendi.\n")
        except Exception as e:
            print(f"⚠️ Hata: {e}\n")
    
    # DB Oluştur
    print("6️⃣ Chroma DB oluşturuluyor...")
    try:
        vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_function,
            collection_name="turkce-kitap-chatbot",
        )
        
        print("   Belgeler ekleniyor (batch halinde)...")
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectordb.add_documents(batch)
            print(f"   {min(i + batch_size, len(documents))}/{len(documents)} eklendi...")
        
        print(f"✅ DB oluşturuldu.\n")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return
    
    # Sonuç
    print("="*70)
    print("✅ BAŞARILI!")
    print("="*70)
    print(f"📊 Toplam kitap: {len(documents_list)}")
    print(f"📦 Toplam chunk: {len(documents)}")
    print(f"💾 Veritabanı: {DB_PATH}")
    print("="*70)

if __name__ == "__main__":
    create_database()
