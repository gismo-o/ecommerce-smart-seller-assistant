import pandas as pd

# Büyük dosyayı oku
df = pd.read_csv("data/final_cleaned_data.csv")

# İlk 10.000 satırı al
df_sample = df.head(10000)  # veya sample(n=10000, random_state=42)

# Yeni küçültülmüş CSV dosyasını kaydet
df_sample.to_csv("data/final_cleaned_data_sample.csv", index=False)

print("Yeni örnek dosya başarıyla kaydedildi.")
