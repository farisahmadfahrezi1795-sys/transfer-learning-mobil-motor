# Klasifikasi Mobil dan Motor Menggunakan Transfer Learning ResNet-18

Project ini merupakan implementasi tugas Computer Vision dan Deep Learning menggunakan metode **Transfer Learning** untuk melakukan klasifikasi gambar kendaraan menjadi dua kelas:

- **Mobil (Car)**
- **Motor (Bike/Motorcycle)**

Model yang digunakan adalah **ResNet-18** dengan pretrained weights dari ImageNet.

Eksperimen dilakukan menggunakan tiga pendekatan:

1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Tujuan eksperimen adalah membandingkan performa transfer learning dengan pelatihan model dari awal pada dataset yang relatif kecil.

---

## 1. Dataset

Dataset sumber yang digunakan adalah **Car vs Bike Classification Dataset** dari Kaggle.

**Sumber dataset:**

https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset

Dataset memiliki dua kelas:

| Class | Label |
|---|---:|
| Mobil | 0 |
| Motor | 1 |

Dalam project ini digunakan **120 gambar unik**, terdiri dari:

- 60 gambar mobil
- 60 gambar motor

### Pembagian Dataset

| Split | Mobil | Motor | Total |
|---|---:|---:|---:|
| Train | 42 | 42 | 84 |
| Validation | 9 | 9 | 18 |
| Test | 9 | 9 | 18 |
| **Total** | **60** | **60** | **120** |

Folder `dataset/raw` berisi salinan gambar sumber yang digunakan dalam proses pembagian dataset.

Metadata final hanya mencatat 120 gambar unik yang digunakan pada train, validation, dan test.

Metadata dataset disimpan pada:

```text
dataset/metadata.csv
```

---

## 2. Metode

Model yang digunakan adalah **ResNet-18** dengan pretrained weights dari ImageNet.

Tiga pendekatan dibandingkan:

- Feature Extraction
- Partial Fine-Tuning
- Training from Scratch

### A. Feature Extraction

Pada pendekatan ini, backbone ResNet-18 dibekukan dan hanya fully connected layer (`fc`) yang dilatih.

| Parameter | Nilai |
|---|---|
| Pretrained weights | ImageNet |
| Trainable layer | fc |
| Learning rate | 1e-3 |
| Epoch | 10 |

### B. Partial Fine-Tuning

Pada pendekatan ini, pretrained ResNet-18 digunakan dan layer terakhir (`layer4`) serta fully connected layer (`fc`) dilatih kembali.

| Parameter | Nilai |
|---|---|
| Pretrained weights | ImageNet |
| Trainable layers | layer4 + fc |
| Learning rate | 1e-4 |
| Epoch | 10 |

### C. Training from Scratch

Pada pendekatan ini, ResNet-18 dilatih dari awal tanpa menggunakan pretrained weights.

| Parameter | Nilai |
|---|---|
| Pretrained weights | Tidak digunakan |
| Trainable layers | Semua layer |
| Learning rate | 1e-3 |
| Epoch | 10 |

---

## 3. Preprocessing dan Augmentation

Ukuran input gambar yang digunakan adalah **224 × 224 piksel**.

Data training menggunakan augmentation:

- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter

Setelah augmentation, gambar dikonversi menjadi tensor dan dinormalisasi menggunakan mean dan standard deviation ImageNet.

Data validation dan test menggunakan preprocessing tanpa augmentation.

---

## 4. Hasil Eksperimen

Hasil training dari ketiga pendekatan:

| Metode | Best Val Accuracy | Test Accuracy | Epoch ≥90% | Training Time |
|---|---:|---:|---:|---:|
| Feature Extraction | 100.00% | 72.22% | 3 | 82.98 s |
| Partial Fine-Tuning | 100.00% | 100.00% | 1 | 108.65 s |
| Training from Scratch | 66.67% | 55.56% | Tidak tercapai | 192.26 s |

### Analisis Hasil

**Partial Fine-Tuning** memberikan hasil terbaik dengan test accuracy sebesar **100.00%**.

Model mencapai validation accuracy ≥90% sejak epoch pertama dan memiliki waktu training sebesar **108.65 detik**.

Feature Extraction mencapai validation accuracy 100%, tetapi test accuracy hanya 72.22%. Hal ini menunjukkan adanya perbedaan performa antara validation dan test pada dataset yang relatif kecil.

Training from Scratch memberikan hasil paling rendah dengan test accuracy sebesar 55.56% dan tidak mencapai validation accuracy 90% selama 10 epoch.

Hasil eksperimen menunjukkan bahwa penggunaan pretrained weights ImageNet membantu model memperoleh performa yang lebih baik dibandingkan melatih ResNet-18 dari awal pada dataset yang relatif kecil.

---

## 5. Model Terbaik

Berdasarkan hasil eksperimen, model terbaik adalah:

**ResNet-18 Partial Fine-Tuning**

Konfigurasi model:

| Parameter | Nilai |
|---|---|
| Model | ResNet-18 |
| Pendekatan | Partial Fine-Tuning |
| Pretrained weights | ImageNet |
| Trainable layers | layer4 + fc |
| Learning rate | 1e-4 |
| Epoch | 10 |
| Validation Accuracy | 100.00% |
| Test Accuracy | 100.00% |

File model terbaik:

```text
models/resnet18_partial.pth
```

---

## 6. Grafik Hasil

Grafik hasil eksperimen disimpan pada folder:

```text
results/
```

Grafik yang dihasilkan:

```text
accuracy_feature.png
accuracy_partial.png
accuracy_scratch.png
accuracy_comparison.png
```

File hasil training dalam format CSV:

```text
history_feature.csv
history_partial.csv
history_scratch.csv
comparison.csv
```

---

## 7. Latency

Pengukuran latency dilakukan menggunakan model terbaik, yaitu **ResNet-18 Partial Fine-Tuning**.

Pengujian dilakukan menggunakan CPU dengan 100 kali pengukuran.

| Parameter | Hasil |
|---|---|
| Model | ResNet-18 Partial Fine-Tuning |
| Device | CPU |
| Average latency | 76.69 ms/image |
| Minimum latency | 64.85 ms/image |
| Maximum latency | 102.40 ms/image |
| Measurements | 100 |

Hasil pengukuran latency disimpan dalam:

```text
results/latency.csv
```

---

## 8. Struktur Project

Struktur project yang digunakan:

```text
transfer-learning-mobil-motor/
│
├── dataset/
│   ├── raw/
│   │   ├── mobil/
│   │   └── motor/
│   ├── train/
│   │   ├── mobil/
│   │   └── motor/
│   ├── val/
│   │   ├── mobil/
│   │   └── motor/
│   ├── test/
│   │   ├── mobil/
│   │   └── motor/
│   └── metadata.csv
│
├── models/
│   ├── resnet18_feature.pth
│   ├── resnet18_partial.pth
│   └── resnet18_scratch.pth
│
├── results/
│   ├── accuracy_feature.png
│   ├── accuracy_partial.png
│   ├── accuracy_scratch.png
│   ├── accuracy_comparison.png
│   ├── comparison.csv
│   ├── history_feature.csv
│   ├── history_partial.csv
│   ├── history_scratch.csv
│   └── latency.csv
│
├── src/
│   ├── create_metadata.py
│   ├── train.py
│   ├── plot_results.py
│   └── measure_latency.py
│
├── .gitignore
└── README.md
```

---

## 9. Cara Menjalankan Project

Aktifkan virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependency:

```powershell
pip install torch torchvision torchaudio
```

Membuat metadata dataset:

```powershell
python src\create_metadata.py
```

Melatih semua eksperimen:

```powershell
python src\train.py
```

Membuat grafik hasil training:

```powershell
python src\plot_results.py
```

Mengukur latency model terbaik:

```powershell
python src\measure_latency.py
```

---

## 10. Kesimpulan

Berdasarkan hasil eksperimen, **Partial Fine-Tuning ResNet-18 dengan pretrained weights ImageNet** merupakan pendekatan terbaik untuk klasifikasi mobil dan motor pada dataset ini.

Model memperoleh:

- **100.00% validation accuracy**
- **100.00% test accuracy**
- Mencapai accuracy ≥90% pada **epoch 1**
- Training time **108.65 detik**
- Average inference latency **76.69 ms/image**

Hasil tersebut menunjukkan bahwa penggunaan transfer learning memberikan performa yang lebih baik dibandingkan Training from Scratch pada dataset yang relatif kecil.

---

## 11. Referensi Dataset

Dataset yang digunakan berasal dari Kaggle:

**Car vs Bike Classification Dataset**

Sumber:

https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset
