import google.generativeai as genai
import os
import re
import json
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

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
      genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
      model = genai.GenerativeModel("gemini-1.5-flash")
      prompt = """
      Bu goruntude bir ilac kutusu/ambalaji var.
      Lutfen sunlari cikar ve JSON formatinda dondur:
      {
        "ilac_adi": "Ilacin tam adi",
        "etken_madde": "Etken madde(ler)",
        "firma": "Uretici firma",
        "doz": "Dozaj (orn: 500mg, 10ml)",
        "form": "Tablet/Surup/Kapsul vb.",
        "tum_metin": "Kutudaki tum okunabilir metinler"
      }
      Sadece JSON dondur, baska hicbir aciklama ekleme.
      """
      try:
                response = model.generate_content(
                              [prompt, image],
                              generation_config=genai.GenerationConfig(
                                                temperature=0.1,
                                                top_p=0.95,
                              )
                )
                return extract_json(response.text.strip())
except Exception as e:
        return {"hata": str(e), "tum_metin": ""}
