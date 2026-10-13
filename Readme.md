# Studio 2: Klasifikasi Sampah 

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

Repositori ini berisi kode pelatihan dan submission kelompok untuk Kompetisi Klasifikasi Sampah Studio Proyek 2. Model menerima satu foto sampah dan memprediksi salah satu dari lima kelas: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, dan `styrofoam`.

## Anggota Kelompok

| No | Nama Lengkap | NIM | 
|----|--------------|-----|
| 1  |  Ahmad Ali Mukti| 123140155 |
| 2  | Anisah Octa Rohila | 123140137 |


## Ringkasan Pendekatan

* **Arsitektur / backbone**: ConvNeXt-Tiny dari `torchvision` dengan bobot pretrained ImageNet-1K (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`), head klasifikasi diganti menjadi 5 kelas.
* **Strategi pelatihan**: fine-tuning dua tahap. Tahap pertama hanya melatih head (3 epoch), tahap kedua membuka seluruh layer dengan learning rate kecil (10 epoch).
* **Data tambahan**: tidak ada.
* **Ukuran input & preprocessing**: `Resize(236)`, `CenterCrop(224)`, `ToTensor()`, lalu normalisasi dengan mean dan standar deviasi ImageNet.

### Arsitektur yang Dipakai

ConvNeXt adalah Convolutional Neural Network (CNN) murni yang desainnya dimodernisasi mengikuti gagasan Vision Transformer, tanpa memakai mekanisme attention. Varian Tiny dipilih karena akurasi top-1 ImageNet-1K sebesar 82,52% dengan ukuran yang masih jauh di bawah batas 500 MB kompetisi.

Aliran data untuk input 224 × 224:

| Bagian | Isi | Ukuran keluaran |
|--------|-----|-----------------|
| Stem | Konvolusi 4 × 4 stride 4, lalu Layer Normalization (LayerNorm) | 56 × 56 × 96 |
| Stage 1 | 3 blok ConvNeXt | 56 × 56 × 96 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 28 × 28 × 192 |
| Stage 2 | 3 blok ConvNeXt | 28 × 28 × 192 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 14 × 14 × 384 |
| Stage 3 | 9 blok ConvNeXt | 14 × 14 × 384 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 7 × 7 × 768 |
| Stage 4 | 3 blok ConvNeXt | 7 × 7 × 768 |
| Head | Global average pooling, LayerNorm, Flatten, `Linear(768, 5)` | 5 logits |

Satu blok ConvNeXt terdiri atas:

1. Konvolusi depthwise 7 × 7 untuk menangkap konteks spasial yang lebar.
2. LayerNorm.
3. Layer linear yang memperbesar jumlah kanal menjadi 4 kali lipat.
4. Aktivasi Gaussian Error Linear Unit (GELU).
5. Layer linear yang mengembalikan jumlah kanal semula.
6. Layer scale dan stochastic depth, kemudian dijumlahkan dengan input blok melalui residual connection.

Jumlah parameter setelah head diganti menjadi 5 kelas adalah 27.823.973.

Alasan pemilihan untuk kasus ini:

1. Data latih hanya sekitar 2.900 gambar, sehingga fitur pretrained ImageNet jauh lebih bisa diandalkan daripada melatih CNN dari nol.
2. ConvNeXt memakai LayerNorm, bukan Batch Normalization, sehingga pelatihan tetap stabil ketika ukuran batch kecil.
3. Batas kompetisi longgar (file bobot maksimal 500 MB dan evaluasi maksimal 15 menit), sehingga tidak ada alasan memilih backbone yang lebih kecil dengan akurasi lebih rendah.

### Strategi Pelatihan

| Komponen | Pengaturan awal |
|----------|-----------------|
| Split lokal | 85% latih dan 15% validasi, stratified per kelas, seed 42 |
| Augmentasi | `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip`, `ColorJitter(0.2, 0.2, 0.2)` |
| Tahap 1 | Backbone dibekukan, hanya head yang dilatih selama beberapa epoch |
| Tahap 2 | Seluruh layer dibuka, optimizer AdamW, learning rate 1e-4, weight decay 0,05 |
| Scheduler | Cosine annealing |
| Loss | Cross entropy dengan label smoothing 0,1 untuk meredam label noise |
| Ketidakseimbangan kelas | Pemantauan F1 per kelas, bobot kelas pada loss bila diperlukan |
| Pemilihan model | Checkpoint dengan Macro-F1 tertinggi pada split validasi lokal |

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | 0.9751 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

* Preview 1: ConvNeXt-Tiny pretrained, fine-tuning dua tahap (head 3 epoch, full 10 epoch). AdamW lr 1e-4, weight decay 0.05, cosine annealing, label smoothing 0.1. Best checkpoint di epoch 4 tahap 2. Macro-F1 0.9751, akurasi 0.9780. Kelas terlemah: kertas_kardus (F1 0.9423), paling sering salah prediksi ke gelas_plastik dan styrofoam.

## Bobot Model

Bobot disimpan di `submission/model.pth` sebagai `state_dict` dan dimuat oleh `evaluate.py` dengan `load_state_dict(strict=True)`.

* State dict disimpan dalam format fp16 agar ukuran file di bawah 100 MB (sekitar 56 MB) sehingga tidak perlu Git LFS. `evaluate.py` tetap bisa memuat bobot fp16 karena `load_state_dict` otomatis menyesuaikan dtype.
* Ukuran: 53.1 MB (fp16). Versi fp32 asli 106.2 MB, Macro-F1 tidak berubah setelah konversi.

## Cara Reproduksi

Dataset tidak disertakan di repositori ini sesuai aturan kompetisi. Unduh train set dari Google Drive mata kuliah, lalu letakkan folder `train/` di root repositori.

```bash
pip install -r requirements.txt
python train.py --data ./train
python evaluate.py --submission ./submission --data ./my_val
```

`train.py` membuat split lokal `my_train/` dan `my_val/`, lalu menyimpan bobot terbaik ke `submission/model.pth`.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() untuk ConvNeXt-Tiny
  model.pth         # state_dict model terbaik
evaluate.py         # salinan skrip evaluasi resmi, tidak diubah
train.py            # skrip pelatihan
requirements.txt    # paket untuk pelatihan dan evaluasi
README.md
```

Folder `train/`, `my_train/`, dan `my_val/` hanya ada di komputer lokal dan diabaikan oleh Git.