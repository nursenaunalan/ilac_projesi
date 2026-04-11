import streamlit as st
from PIL import Image
import io
import os
import re
import json
import time
import numpy as np
import torch
import easyocr
from google import genai
from groq import Groq
from duckduckgo_search import DDGS
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💊 İlaç Analiz Asistanı",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# .env dosyasını yükle
load_dotenv()

# ── Özel CSS (mobil uyum) ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-weight: bold;
    }
    .result-box {
        background: #f0f7ff;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    @media (max-width: 600px) {
        .stButton > button { width: 100% !important; }
        h1 { font-size: 1.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("💊 İlaç Analiz Asistanı")
st.caption("Fotoğraf çek veya yükle → İlaç hakkında her şeyi öğren")

# ── Uyarı Bandı ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
⚠️ Bu uygulama yalnızca bilgilendirme amaçlıdır.
Tıbbi tavsiye niteliği taşımaz. İlaç kullanmadan önce
mutlaka doktorunuza veya eczacınıza danışınız.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── UTILS & MODULES (CONSOLIDATED) ───────────────────────────────────────────

def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    if len(text) < 20:
        text = text.replace('0', 'O').replace('1', 'I')
    return text.strip()

def extract_drug_name(text: str) -> str:
    words = text.split()
    candidates = [w for w in words if w.isupper() and len(w) > 3]
    if candidates:
        return candidates[0]
    for w in words:
        if w[0].isupper() and len(w) > 3:
            return w
    return text[:50] if text else "Bilinmiyor"

@st.cache_resource
def get_ocr_reader():
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(['tr', 'en'], gpu=use_gpu)

def extract_text_from_image(image: Image.Image) -> str:
    try:
        reader = get_ocr_reader()
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        return " ".join(results).strip()
    except Exception as e:
        return f"OCR okuma hatası: {str(e)}"

def extract_json(text: str) -> dict:
    try:
        text = re.sub(r'```json\s*|\s*```', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        return {}

def analyze_image_with_gemini(image: Image.Image) -> dict:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = """Bu görüntüde bir ilaç kutusu/ambalajı var. Lütfen şunları çıkar ve JSON formatında döndür:
    { "ilac_adi": "İlacın tam adı", "etken_madde": "Etken madde(ler)", "firma": "Üretici firma", "doz": "Dozaj", "form": "Tablet/Şurup vb.", "tum_metin": "Okunan tüm metin" }
    Sadece JSON döndür, açıklama ekleme."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        return extract_json(response.text.strip())
    except Exception as e:
        return {"hata": str(e), "tum_metin": ""}

def search_drug_info(drug_name: str) -> str:
    if not drug_name or len(drug_name) < 2: return "Geçersiz ilaç adı."
    queries = [f"{drug_name} ilaç prospektüs endikasyon yan etki", f"{drug_name} etken maddesi nedir"]
    results_text = ""
    for query in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if results:
                    for r in results:
                        results_text += f"\nBaşlık: {r.get('title','')}\nÖzet: {r.get('body','')}\n"
        except: pass
    return results_text if results_text else "Arama sonucu yok."

def analyze_drug_llm(drug_name: str, active_ingredient: str, web_info: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = "Sen uzman bir farmakologsun. Bilimsel ve tarafsız bir rapor sun. Dozaj tavsiye etme, uyarıları belirt."
    prompt = f"İsim: {drug_name}, Etken: {active_ingredient}\nWeb Bilgisi: {web_info}\nİçerik: Genel bakış, Endikasyonlar, Yan etkiler, Etkileşimler, Türkiye muadilleri."
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e: return f"LLM hatası: {str(e)}"

def generate_pdf_report(drug_name: str, analysis_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    font_path = "arial.ttf"
    font_bold_path = "arialbd.ttf"
    
    if os.path.exists(font_path) and os.path.exists(font_bold_path):
        pdf.add_font("ArialCustom", "", font_path)
        # Bazen bold varyasyon eklendiğinde FPDF style='B' algılar
        pdf.add_font("ArialCustom", "B", font_bold_path)
        font_family = "ArialCustom"
    else:
        font_family = "Helvetica"
        
    pdf.set_font(font_family, "B", 16)
    
    # Geleneksel font ise latin-1 dönüşümü, TTF özel font ise orijinal unicode bırak
    def safe_text(txt):
        if font_family == "ArialCustom":
            return txt
        return txt.encode('latin-1', 'replace').decode('latin-1')

    pdf.cell(0, 10, safe_text(f"Analiz Raporu: {drug_name}"), ln=True, align="C")
    
    pdf.set_font(font_family, "", 10)
    pdf.cell(0, 10, safe_text(f"Tarih: {datetime.now().strftime('%d.%m.%Y')}"), ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font(font_family, "", 12)
    clean_text = analysis_text.replace("#", "").replace("*", "")
    pdf.multi_cell(0, 7, safe_text(clean_text))
    
    return bytes(pdf.output())

# ── UI MANTIĞI ─────────────────────────────────────────────────────────────

# Görsel Giriş
col1, col2 = st.columns(2)
with col1:
    camera_photo = st.camera_input("📷 Kamera ile Çek", key="camera_widget")
with col2:
    uploaded_file = st.file_uploader("📁 Dosya Yükle", type=["jpg", "png", "jpeg"], key="uploader_widget")

image_source = camera_photo or uploaded_file
image = None
if image_source:
    image = Image.open(io.BytesIO(image_source.getvalue()))
    image = preprocess_image(image)
    st.image(image, caption="Yüklenen Görsel", width="stretch")

with st.expander("✍️ Manuel Giriş"):
    manual_drug = st.text_input("İlaç adı girin", key="manual_input")

# Analiz Butonu
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

analyze_btn = st.button("🔍 Analiz Et", type="primary", disabled=(image is None and not manual_drug), key="main_analyze_btn")

# İşlem Alanı (Stabil React DOM için st.empty kullanıldı)
status_placeholder = st.empty()

if analyze_btn:
    drug_name, active_ingredient, gemini_data, analysis = "", "", {}, ""
    with status_placeholder.status("🔄 Analiz yapılıyor...", expanded=True) as status:
        if image:
            status.write("🖼️ Görsel analiz ediliyor...")
            gemini_data = analyze_image_with_gemini(image)
            drug_name = gemini_data.get("ilac_adi", "")
            active_ingredient = gemini_data.get("etken_madde", "")
            if not drug_name:
                status.write("🔡 OCR deneniyor...")
                raw_text = extract_text_from_image(image)
                drug_name = extract_drug_name(clean_ocr_text(raw_text))
        elif manual_drug:
            drug_name = manual_drug
        
        status.write(f"🌐 '{drug_name}' aranıyor...")
        web_info = search_drug_info(drug_name)
        
        status.write("🤖 Rapor oluşturuluyor...")
        analysis = analyze_drug_llm(drug_name, active_ingredient or drug_name, web_info)
        
        st.session_state.analysis_result = {"name": drug_name, "gemini": gemini_data, "analysis": analysis}
        status.update(label="✅ Tamamlandı!", state="complete")

# Sonuç Alanı
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    st.divider()
    st.subheader(f"📋 {res['name']} - Sonuç")
    
    if res['gemini'] and "hata" not in res['gemini']:
        c1, c2 = st.columns(2)
        c1.metric("İlaç", res['gemini'].get("ilac_adi", "-"))
        c2.metric("Etken Madde", res['gemini'].get("etken_madde", "-"))
    
    st.markdown(res['analysis'])
    st.error("️⚠️ UYARI: Bu analiz bilgilendirme amaçlıdır. Doktora danışın.", icon="⚠️")
    
    # İndirme
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button("📄 PDF İndir", data=generate_pdf_report(res['name'], res['analysis']), file_name="rapor.pdf", mime="application/pdf", key="pdf_btn")
    with dcol2:
        st.download_button("📝 Metin İndir", data=res['analysis'].encode("utf-8"), file_name="rapor.txt", mime="text/plain", key="txt_btn")

# Footer
st.divider()
st.caption("💡 Powered by Groq · Gemini · EasyOCR · Streamlit")
st.caption("🎓 Python Yapay Zeka Kursu — Görüntü İşleme Projesi")
st.caption("✍️ Hazırlayan: **Nursena Ünalan**")
