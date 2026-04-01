import easyocr
import numpy as np
import torch
from PIL import Image
import streamlit as st
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
        return f"OCR okuma hatasi: {str(e)}"
