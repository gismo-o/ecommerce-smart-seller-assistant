# 🤖 AI-Powered Smart Seller Assistant

📦 **Akıllı Satıcı Asistanı**, e-ticaret platformlarında satıcıların ürünlerine gelen yorumları analiz ederek, ürün geliştirme ve müşteri memnuniyeti için **yapay zeka destekli stratejik öneriler** sunan bir karar destek sistemidir.

---

## 📌 Proje Özeti

Bu uygulama, ürünlere gelen müşteri yorumlarını:

- Öncelikle duygusal açıdan analiz eder (Pozitif / Negatif / Nötr),
- Duygu dağılımı özetini çıkarır,
- Gemini modeli kullanarak yorumlara dayalı stratejik öneriler üretir,
- ElevenLabs API ile bu önerileri sesli hale getirir.

Satıcılar bu sayede ürün açıklamalarını güncelleyebilir, paketleme sorunlarını fark edebilir veya hedef kitleye yönelik reklam stratejilerini yeniden planlayabilir.

---

## Uygulama Özellikleri

✅ Kullanıcı dostu Streamlit arayüzü  
✅ Ürün bazlı yorum görüntüleme  
✅ Otomatik duygu analizi (LightGBM + TF-IDF)  
✅ Gemini ile öneri üretimi  
✅ ElevenLabs ile sesli öneri sunumu  
✅ Colab destekli analiz defteri (Data Science süreci)

---

## Kullanılan Teknolojiler

| Teknoloji       | Açıklama                                      |
|----------------|-----------------------------------------------|
| `Python`        | Backend ve ML modeli geliştirme               |
| `Streamlit`     | Web uygulaması arayüzü                        |
| `LightGBM`      | Duygu sınıflandırması modeli                  |
| `Gemini API`    | Stratejik öneri üretimi (LLM)                 |
| `ElevenLabs`    | Metni sese çevirme                            |
| `GitHub`        | Versiyon kontrolü ve kod paylaşımı            |

---

## Veri Seti
Projede kullanılan örnek veriler, [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) üzerinden türetilmiştir.

---

## Proje Yapısı

```bash
ecommerce_smart_seller_assistant
├── app.py                         # Streamlit arayüzü
├── preprocessing.py              # Veri ön işleme ve tahmin
├── llm_gemini.py                 # Gemini API entegrasyonu
├── tts_elevenlabs.py             # ElevenLabs sesli çıktı modülü
├── requirements.txt              # Gerekli Python kütüphaneleri
├── .gitignore                    # Dosya hariç tutma kuralları
├── data/
│   └── final_cleaned_data_sample.csv    # Örnek yorum verisi
├── models/
│   ├── lightgbm_model_v2.pkl     # Eğitimli ML modeli
│   ├── tfidf_vectorizer.pkl      # TF-IDF vektörleştirici
│   └── label_encoder.pkl         # Label encoder
└── colab_notebooks/
    └── ecommerce_smart_seller_assistant.ipynb  # Veri analizi defteri

```
---

## 📸 Ekran Görüntüleri

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf9c38c2-7bbb-4808-990f-59ec78737699" width="30%" alt="Overview">
  <img src="https://github.com/user-attachments/assets/42213792-d231-4d97-a6f1-8320a769e1b7" width="30%" alt="Reviews">
  <img src="https://github.com/user-attachments/assets/1088d78a-d089-4992-83a1-6796bcaa82a4" width="30%" alt="Insights">
</p>

---

> 📝 **Not:** Bu proje, **BTK Hackathon 2025** kapsamında geliştirilmiştir.
