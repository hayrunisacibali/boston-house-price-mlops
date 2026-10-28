
# Boston House Price Prediction — MLOps

Bu proje, veri bilimi sürecinde geliştirilen en iyi makine öğrenmesi modelinin üretime alınmasından sonraki yaşam döngüsüne odaklanır. XGBoost ile ev fiyatı tahmini yapan sistem; MLflow ile deney ve model yönetimi, Streamlit ile tahmin arayüzü, SQLite ile tahmin kayıtlarının tutulması, veri kayması (data drift) izleme ve aday modellerin mevcut modelle karşılaştırılması süreçlerini bir araya getirir.

Amaç, modelin yalnızca geliştirilmesini değil, üretimde izlenmesini, performansının değerlendirilmesini ve daha iyi bir aday bulunduğunda kontrollü biçimde güncellenmesini sağlayan uçtan uca bir MLOps iş akışı oluşturmaktır.

## Kullanılan Teknolojiler

- **Python:** Veri işleme ve makine öğrenmesi uygulamaları
- **XGBoost:** Ev fiyatı tahmin modeli
- **Scikit-learn:** Veri hazırlama ve model değerlendirme
- **Pandas & NumPy:** Veri işleme
- **MLflow:** Deney takibi ve model yönetimi
- **Streamlit:** Etkileşimli tahmin arayüzü
- **SQLite:** Yerel veri ve kayıt yönetimi

## Proje Özellikleri

1. **Veri hazırlama:** Veri seti yüklenir ve eğitim, doğrulama ve test kümelerine ayrılır.
2. **Model eğitimi ve değerlendirme:** XGBoost modeli eğitilir; tahmin performansı MAE, RMSE ve R² gibi metriklerle değerlendirilir.
3. **MLflow ile deney ve model yönetimi:** Eğitim sonuçları, performans metrikleri ve model MLflow'a kaydedilir. Model, `BostonHousePriceModel` adıyla Model Registry üzerinden yönetilir.
4. **Production modelinin kullanılması:** Streamlit uygulaması, MLflow Model Registry'de `champion` alias'ı ile işaretlenen modeli kullanarak ev fiyatı tahmini üretir.
5. **Tahmin kayıtları:** Kullanıcıdan alınan girdiler ve üretilen tahminler SQLite veritabanında saklanır.
6. **Drift izleme:** Üretimden gelen veriler, belirlenen production baseline ile karşılaştırılarak veri dağılımındaki değişimler izlenir.
7. **Aday model değerlendirme ve terfi:** Her 50 yeni tahmin kaydından sonra aday model değerlendirme süreci başlatılır. Aday modelin performansı mevcut `champion` modelle karşılaştırılır. Belirlenen değerlendirme koşulunu sağlıyor ve daha iyi sonuç veriyorsa `champion` alias'ı adaya atanır; aksi durumda mevcut model korunur.

## Ekran Görüntüleri

### 1. Streamlit Tahmin Uygulaması

Uygulama üzerinden ev özellikleri girilerek fiyat tahmini yapılabilir.

![Streamlit tahmin uygulaması](screenshots/streamlit-app.png)

### 2. MLflow Runs

MLflow arayüzü üzerinden model eğitimlerine ait deney kayıtları incelenebilir.

![MLflow Runs](screenshots/mlflow-runs.png)

### 3. Mlflow Model Kayıtları

Mlflow Model Kayıtları ve Champion model MLflow arayüzü üzerinden takip edilebilir.

![MLflow Metrics](screenshots/mlflow-model-registry.png)

## Kurulum ve Çalıştırma

Tüm komutları proje ana klasöründen çalıştırın.

```bash
# 1. Kütüphaneleri yükle
pip install -r requirements.txt

# 2. İlk modeli eğit ve MLflow'a kaydet
python create_production_baseline.py

# 3. Modeli champion olarak işaretle
python scripts/registry.py

# 4. Uygulamayı başlat
streamlit run app.py
```

Deneyleri ve kayıtlı modelleri görüntülemek için:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Notlar

- `mlruns/`, `mlflow.db` ve `predictions.db` yerel çalışma dosyaları olduğu için repoya eklenmemiştir.
- Uygulama, modeli MLflow Model Registry üzerinden `models:/BostonHousePriceModel@champion` adresiyle yükler.
