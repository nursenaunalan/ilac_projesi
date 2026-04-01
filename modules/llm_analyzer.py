from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SYSTEM_PROMPT = """Sen uzman bir farmakolog ve tibbi bilgi asistanisin. 
Gorevin, kullanicinin sagladigi ilac bilgilerini ve web arama sonuclarini analiz ederek dogru, 
tarafsiz ve kolay anlasilir bir rapor sunmaktadir.
Kurallar:
1. Bilimsel ve profesyonel bir dil kullan.
2. Web sonuclarinda celiski varsa guvenilir tibbi kaynaklara (FDA, EMA, TITCK) oncelik ver.
3. Kesinlikle dozaj tavsiyesi verme, sadece genel kullanim rehberini acikla.
4. Yan etkileri "Yaygin", "Seyrek" ve "Ciddi" olarak kategorize et.
5. Her zaman Turkiye'deki muadilleri de (mumkunse) belirt.
"""
def analyze_drug(drug_name: str, active_ingredient: str, web_info: str) -> str:
      prompt = f"Ilac Adi: {drug_name}\nEtken Madde: {active_ingredient}\n\nInternetten derlenen ham bilgiler:\n{web_info[:4000]}\n\nYukaridaki verilere dayanarak su yapida bir rapor hazirlamaz misin? (Rapor basligini Ilac kelimesiyle baslat)..."
      try:
                response = client.chat.completions.create(
                              model="llama-3.3-70b-versatile",
                              messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                              temperature=0.2,
                              max_tokens=2500,
                )
                return response.choices[0].message.content
except Exception as e:
        return f"Analiz hatasi (LLM): {str(e)}"
def quick_ingredient_analysis(ingredients_text: str) -> str:
      prompt = f"Su ilac bilesenlerini kisaca analiz et ve ne ise yaradiklarini acikla:\n\n{ingredients_text}"
      try:
                response = client.chat.completions.create(
                              model="llama-3.3-70b-versatile",
                              messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                              temperature=0.2,
                              max_tokens=1000,
                )
                return response.choices[0].message.content
except Exception as e:
        return f"Hizli analiz hatasi: {str(e)}"
