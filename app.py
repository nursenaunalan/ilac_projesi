import streamlit as st
from PIL import Image
import io
import os
from dotenv import load_dotenv

from modules.ocr_reader import extract_text_from_image
from modules.gemini_vision import analyze_image_with_gemini
from modules.web_search import search_drug_info
from modules.llm_analyzer import analyze_drug, quick_ingredient_analysis
from modules.report_generator import generate_pdf_report
from utils.image_utils import preprocess_image
from utils.text_utils import clean_ocr_text, extract_drug_name

load_dotenv()

st.set_page_config(
      page_title="Ilac Analiz Asistani",
      page_icon="pill",
      layout="centered",
      initial_sidebar_state="collapsed",
)

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

st.title("Ilac Analiz Asistani")
st.caption("Fotograf cek veya yukle -> Ilac hakkinda her seyi ogren")

st.markdown("""
<div class="warning-box">
UYARI: Bu uygulama yalnizca bilgilendirme amaclidir.
Tibbi tavsiye niteligi tasimaz. Ilac kullanmadan once
mutlaka doktorunuza veya eczaciniza danisiniz.
</div>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)
with col1:
      camera_photo = st.camera_input("Kamera ile Cek")
  with col2:
        uploaded_file = st.file_uploader(
                  "Dosya Yukle",
                  type=["jpg", "jpeg", "png", "webp", "bmp"],
                  help="Ilac kutusunun net fotografini yukleyin"
        )

image_source = camera_photo or uploaded_file
image = None
if image_source:
      image = Image.open(io.BytesIO(image_source.getvalue()))
      image = preprocess_image(image)
      st.image(image, caption="Yuklenen Gorsel", use_container_width=True)

st.divider()

with st.expander("Manuel ilac adi gir (gorsel yoksa)"):
      manual_drug = st.text_input("Ilac adi veya etken madde", placeholder="orn: Aspirin, Ibuprofen, Parasetamol...")

analyze_btn = st.button(
      "Analiz Et",
      type="primary",
      use_container_width=True,
      disabled=(image is None and not manual_drug)
)

if "analysis_result" not in st.session_state:
      st.session_state.analysis_result = None

if analyze_btn:
      drug_name = ""
      active_ingredient = ""
      gemini_data = {}
      analysis = ""
      with st.status("Analiz yapiliyor...", expanded=True) as status:
                if image:
                              st.write("Gorsel analiz ediliyor (Gemini Vision)...")
                              try:
                                                gemini_data = analyze_image_with_gemini(image)
                                                drug_name = gemini_data.get("ilac_adi", "")
                                                active_ingredient = gemini_data.get("etken_madde", "")
                                                st.write(f"Ilac tespit edildi: **{drug_name}**")
except Exception as e:
                st.write(f"Gemini hatasi: {str(e)}, OCR deneniyor...")
            if not drug_name:
                              st.write("OCR ile metin okunuyor...")
                              raw_text = extract_text_from_image(image)
                              cleaned = clean_ocr_text(raw_text)
                              drug_name = extract_drug_name(cleaned)
                              active_ingredient = cleaned
                              st.write(f"Metin okundu: {cleaned[:100]}...")

elif manual_drug:
              drug_name = manual_drug
              active_ingredient = manual_drug
          st.write(f"'{drug_name}' internette araniyor...")
        web_info = search_drug_info(drug_name)
        if "bulunamadi" in web_info or not web_info.strip():
                      st.write("Internet bilgisi sinirli, etken maddeye gore yorum yapilacak.")
else:
              st.write("Web bilgisi bulundu.")
          st.write("Groq LLM ile detayli analiz yapiliyor...")
        if drug_name:
                      analysis = analyze_drug(drug_name, active_ingredient, web_info)
else:
              analysis = quick_ingredient_analysis(active_ingredient)
          st.session_state.analysis_result = {
                        "drug_name": drug_name,
                        "gemini_data": gemini_data,
                        "analysis": analysis
          }
        status.update(label="Analiz tamamlandi!", state="complete")

if st.session_state.analysis_result:
      res = st.session_state.analysis_result
      drug_name = res["drug_name"]
      gemini_data = res["gemini_data"]
      analysis = res["analysis"]
      st.divider()
      st.subheader(f"{drug_name} - Analiz Sonucu")
      if gemini_data and "hata" not in gemini_data:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                              st.metric("Ilac Adi", gemini_data.get("ilac_adi", "-"))
                          with col_b:
                    st.metric("Etken Madde", gemini_data.get("etken_madde", "-"))
                                    with col_c:
                                        st.metric("Form", gemini_data.get("form", "-"))
                                          st.markdown(analysis)
                                          st.error(
                                              "ONEMLI UYARI: Bu analiz yapay zeka tarafindan olusturulmustur. Tibbi teshis veya tedavi tavsiyesi degildir. Ilac kullanimi icin mutlaka doktorunuza danisiniz.",
                                              icon="warning"
                                          )
                                          st.divider()
                                          st.subheader("Raporu Indir")
                                          dl_col1, dl_col2 = st.columns(2)
                                          with dl_col1:
                                              st.download_button(
                                                            label="PDF Indir",
                                                            data=generate_pdf_report(drug_name, analysis),
                                                            file_name=f"ilac_raporu_{drug_name.replace(' ', '_')}.pdf",
                                                            mime="application/pdf",
                                                            use_container_width=True
                                              )
                                                with dl_col2:
                                                    st.download_button(
                                                                  label="Metin Indir",
                                                                  data=analysis.encode("utf-8"),
                                                                  file_name=f"ilac_raporu_{drug_name.replace(' ', '_')}.txt",
                                                                  mime="text/plain",
                                                                  use_container_width=True
                                                    )
                                                  st.divider()
                                            st.caption("Powered by Groq LLaMA - Google Gemini - EasyOCR - Streamlit")
                                      st.caption("Bilisim Ogretmeni Python Yapay Zeka Kursu - Goruntu Isleme Projesi")
                            
