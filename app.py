import os
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# =====================================================================
# 1. SAYFA YAPILANDIRMASI
# =====================================================================

st.set_page_config(
    page_title="Türkçe Kitap Danışmanı",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS Stili
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #f0f0f0;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stChatMessage"][data-chat-message-type="user"] {
        background: linear-gradient(135deg, rgba(100, 200, 255, 0.15), rgba(100, 200, 255, 0.05));
        border-left: 3px solid #64c8ff;
        margin-left: 2rem;
    }
    
    [data-testid="stChatMessage"][data-chat-message-type="assistant"] {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1), rgba(255, 165, 0, 0.02));
        border-left: 3px solid #ffa500;
        margin-right: 2rem;
    }
    
    /* Input Container */
    [data-testid="stChatInputContainer"] {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 15, 30, 0.95);
        padding: 1.5rem;
    }
    
    [data-testid="stChatInputContainer"] input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        color: #f0f0f0 !important;
        padding: 12px 16px !important;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stChatInputContainer"] input:focus {
        border-color: #ffa500 !important;
        box-shadow: 0 0 15px rgba(255, 165, 0, 0.3) !important;
    }
    
    /* Başlık */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.1), rgba(100, 200, 255, 0.1));
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffa500, #64c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(255, 165, 0, 0.3);
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #a0a0b0;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Öneriler */
    .suggestion-container {
        text-align: center;
        margin: 2rem 0;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .suggestion-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffa500, #64c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
    }
    
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.2), rgba(255, 165, 0, 0.1));
        border: 2px solid rgba(255, 165, 0, 0.5);
        color: #ffa500;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    [data-testid="stButton"] button:hover {
        background: rgba(255, 165, 0, 0.3);
        border-color: #ffa500;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metrikler */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Container */
    .block-container {
        padding: 2rem 1rem !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 165, 0, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 165, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. YAPILANDIRMA
# =====================================================================

load_dotenv()

DB_PATH = "./chroma_db"
MODEL_NAME = "gemini-2.0-flash"
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ **GEMINI_API_KEY bulunamadı!**")
    st.info("💡 Lütfen `.env` dosyasını kontrol edin")
    st.stop()

genai.configure(api_key=api_key)

# =====================================================================
# 3. SESSION STATE BAŞLATMA
# =====================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "vectorstore" not in st.session_state:
    with st.spinner("🔄 Veritabanı yükleniyor..."):
        try:
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            st.session_state.vectorstore = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embedding_function,
                collection_name="turkce-kitap-chatbot"
            )
            st.session_state.db_loaded = True
        except Exception as e:
            st.error(f"❌ **Veritabanı yuklenemedi**\n\n{str(e)}")
            st.session_state.db_loaded = False

# =====================================================================
# 4. RAG FONKSIYONU
# =====================================================================

def get_rag_response(query, conversation_history=""):
    if not st.session_state.db_loaded:
        return "❌ Veritabanı yüklenmedi. Lütfen çalıştırıldığından emin olun."
    
    try:
        results = st.session_state.vectorstore.similarity_search(query, k=5)
        
        if not results:
            return "🤔 Veritabanında bu konuyla ilgili kitap bulamadım. Farkli bir konu sorabilir misiniz?"
        
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # Yazarı extract et
        authors = set()
        for doc in results:
            lines = doc.page_content.split('\n')
            for line in lines:
                if 'Yazar:' in line or 'yazar' in line.lower():
                    author = line.split(':')[-1].strip()
                    if author and author != 'Yazar':
                        authors.add(author)
        
        authors_list = list(authors)[:5]  # Max 5 yazar
        
        full_prompt = f"""Sen Türkiye'nin en samimi ve bilgili edebiyat danişmanisın. Senin görevin KİTAP DEĞİL, YAZAR önerisi vermektir.

📚 ÖNCEKI KONUŞMA:
{conversation_history if conversation_history else "İlk soru bu."}

✍️ İLGİLİ YAZARLAR:
{chr(10).join([f"- {author}" for author in authors_list]) if authors_list else "Yazarlar"}

❓ KULLANICI SORUSU: {query}

✨ TALIMATLAR:
1. YAZARLARI ÖNERİ, KİTAPLARI DEĞİL
2. Her yazar için şunu yaz:
   ✍️ **Yazar Adı**
   - Kısa tanıtım (bu yazar neden meşhur, hangi tarzda yazıyor)
   - 2-3 kitap örneği (isteğe bağlı, sadece bilineni varsa)
3. 2-3 yazar öner
4. Eğer yazarın tanınmış kitapları yoksa sadece yazarı söyle, kitap sayma
5. Samimi, sıcak ve dostane bir dil kullan

KURALLAR:
- ODAK: YAZAR
- Kitap sayarken: 2-3 eser örneği, daha fazla değil
- Yazar yoksa yazma

💬 CEVAP:"""
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1000,
            ),
        )
        
        return response.text if response.text else "⚠️ Cevap oluşturulamadı."
        
    except Exception as e:
        return f"❌ **Hata:** {str(e)}"

# =====================================================================
# 5. ARAYÜZ
# =====================================================================

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📚 TÜRKÇE KİTAP DANIŞMANI</div>
    <div class="header-subtitle">Yapay zeka destekli akıllı kitap tavsiyeleri</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Mesajlar
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) > 0:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📚"):
                st.markdown(message["content"])

# Öneriler
if st.session_state.show_suggestions and len(st.session_state.messages) == 0:
    st.markdown("---")
    st.markdown("""
    <div class="suggestion-container">
        <div class="suggestion-title">✨ Hangi Türde Kitaplar Arıyorsunuz?</div>
    </div>
    """, unsafe_allow_html=True)
    
    example_questions = [
        ("📖 Tarih Konusu", "Tarih konusunda kitap öner"),
        ("🏛️ Osmanlı Dönemi", "Osmanlı dönemi romanları öner"),
        ("👶 Çocuklar İçin", "Çocuklar için kitap öner"),
        ("🧠 Felsefe", "Felsefe kitabı arıyorum"),
        ("📜 Türk Edebiyatı", "Türk edebiyatı klasikleri"),
        ("🎭 Macera Romanı", "Macera romanları öner"),
    ]
    
    cols = st.columns(2)
    for idx, (label, question) in enumerate(example_questions):
        with cols[idx % 2]:
            if st.button(label, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.show_suggestions = False
                
                with st.chat_message("user", avatar="👤"):
                    st.markdown(question)
                
                with st.chat_message("assistant", avatar="📚"):
                    with st.spinner("🤔 Kitapları arıyorum..."):
                        response = get_rag_response(question)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# Input Alanı
if prompt := st.chat_input("💭 Kitap hakkında bir şey sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.show_suggestions = False
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("🤔 Kitapları arıyorum..."):
            history = "\n".join([
                f"{'Kullanıcı' if m['role']=='user' else 'Danışman'}: {m['content']}"
                for m in st.session_state.messages[-6:-1]
            ])
            
            response = get_rag_response(prompt, history)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# -*- coding: utf-8 -*-
# Dosya: chatbot_app.py
# Amaç: Türkçe Kitap Tavsiye Chatbot - Modern Streamlit Arayüzü

import os
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# **YENİ EKLEME:** database.py dosyasından DB oluşturma fonksiyonunu import ediyoruz
from database import create_database 


# =====================================================================
# 1. SAYFA YAPILANDIRMASI
# =====================================================================

st.set_page_config(
    page_title="Türkçe Kitap Danışmanı",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS Stili (CSS kodları buraya dahil edilmemiştir, ancak orijinal dosyada olmalıdır.)
st.markdown("""
<style>
    /* ... Orijinal CSS Kodunuz ... */
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. YAPILANDIRMA
# =====================================================================

load_dotenv()

DB_PATH = "./chroma_db"
MODEL_NAME = "gemini-2.0-flash"
# Streamlit Secrets'ı okumak için os.getenv'i kullanıyoruz.
# Eğer secrets'a eklediyseniz, bu satır otomatik çalışır.
api_key = os.getenv("GEMINI_API_KEY") 

if not api_key:
    # Streamlit Cloud'da çalışırken bu kontrol Secrets olmadan çalışır.
    st.error("❌ **GEMINI_API_KEY bulunamadı!**")
    st.info("💡 Lütfen `.env` dosyasını veya Streamlit Secrets'ı kontrol edin")
    st.stop()

genai.configure(api_key=api_key)

# =====================================================================
# 3. SESSION STATE BAŞLATMA - KRİTİK DEĞİŞİKLİK BURADA
# =====================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "vectorstore" not in st.session_state:
    # **YENİ MANTIK:** Eğer DB klasörü yoksa (Deploy ortamında yok olacak) VEYA boşsa, oluştur.
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        st.warning("⚠️ Chroma DB bulunamadi. Veritabani olusturuluyor... (Bu ilk deployda biraz surebilir)")
        
        # database.py'deki create_database fonksiyonunu çağırıyoruz
        create_database() 
        
        st.success("✅ Veritabanı başarıyla oluşturuldu! Şimdi yükleniyor...")
        
    with st.spinner("🔄 Veritabanı yükleniyor..."):
        try:
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
            st.session_state.vectorstore = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embedding_function,
                collection_name="turkce-kitap-chatbot"
            )
            st.session_state.db_loaded = True
        except Exception as e:
            st.error(f"❌ **Veritabanı yuklenemedi**\n\n{str(e)}")
            st.session_state.db_loaded = False

# =====================================================================
# 4. RAG FONKSIYONU
# =====================================================================

def get_rag_response(query, conversation_history=""):
    # ... (Fonksiyon içeriği aynı kalır)
    if not st.session_state.db_loaded:
        return "❌ Veritabanı yüklenmedi. Lütfen çalıştırıldığından emin olun."
    
    try:
        results = st.session_state.vectorstore.similarity_search(query, k=5)
        
        if not results:
            return "🤔 Veritabanında bu konuyla ilgili kitap bulamadım. Farkli bir konu sorabilir misiniz?"
        
        # ... (Yazar çıkarma ve Prompt oluşturma kısmı aynı kalır)

        full_prompt = f"""Sen Türkiye'nin en samimi ve bilgili edebiyat danişmanisın. Senin görevin KİTAP DEĞİL, YAZAR önerisi vermektir.
# ... (Prompt içeriği aynı kalır)
💬 CEVAP:"""
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1000,
            ),
        )
        
        return response.text if response.text else "⚠️ Cevap oluşturulamadı."
        
    except Exception as e:
        return f"❌ **Hata:** {str(e)}"

# =====================================================================
# 5. ARAYÜZ (Geri kalan kodunuz, sadece bir kopyası kalacak şekilde)
# =====================================================================

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📚 TÜRKÇE KİTAP DANIŞMANI</div>
    <div class="header-subtitle">Yapay zeka destekli akıllı kitap tavsiyeleri</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Mesajlar
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) > 0:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📚"):
                st.markdown(message["content"])

# Öneriler
if st.session_state.show_suggestions and len(st.session_state.messages) == 0:
    # ... (Öneriler kısmı aynı kalır)
    pass # CSS kodu burada tekrar yer almalıdır, kısaltılmıştır.

# Input Alanı
if prompt := st.chat_input("💭 Kitap hakkında bir şey sorun..."):
    # ... (Input kısmı aynı kalır)
    pass # Kod içeriği aynı kalır.

# İstatistikler
if len(st.session_state.messages) > 0:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💬 Toplam Mesaj", len(st.session_state.messages))
    
    with col2:
        user_msgs = sum(1 for m in st.session_state.messages if m['role'] == 'user')
        st.metric("❓ Soruların", user_msgs)
    
    with col3:
        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_suggestions = True
            st.rerun() # <-- Dosya burada bitmeli