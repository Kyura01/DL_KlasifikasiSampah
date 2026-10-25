# Studio 2: Klasifikasi Sampah 

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

Repositori ini berisi kode pelatihan dan submission kelompok untuk Kompetisi Klasifikasi Sampah Studio Proyek 2. Model menerima satu foto sampah dan memprediksi salah satu dari lima kelas: `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, dan `styrofoam`.

## Anggota Kelompok

| No | Nama Lengkap | NIM | 
|----|--------------|-----|
| 1  |  Ahmad Ali Mukti| 123140155 |
| 2  | Anisah Octa Rohila | 123140137 |


## Ringkasan Pendekatan

* **Arsitektur / backbone**: ConvNeXt-Tiny dari `timm` dengan bobot pretrained ImageNet-22k yang di-fine-tune ke ImageNet-1k (`convnext_tiny.fb_in22k_ft_in1k`), head klasifikasi diganti menjadi 5 kelas. Saat evaluasi, model merata-rata logits gambar asli dan gambar yang di-flip horizontal (TTA).
* **Strategi pelatihan**: fine-tuning satu tahap seluruh layer selama 25 epoch dengan warmup, cosine annealing, MixUp/CutMix, dan EMA bobot. Model final dilatih ulang di seluruh data latih (`my_train` + `my_val`).
* **Data tambahan**: tidak ada.
* **Ukuran input & preprocessing**: `Resize(304)`, `CenterCrop(288)`, `ToTensor()`, lalu normalisasi dengan mean dan standar deviasi ImageNet.
* **Paket tambahan**: `timm` (tercantum di `submission/requirements.txt`).

### Arsitektur yang Dipakai

ConvNeXt adalah Convolutional Neural Network (CNN) murni yang desainnya dimodernisasi mengikuti gagasan Vision Transformer, tanpa memakai mekanisme attention. Varian Tiny dipilih karena ukurannya jauh di bawah batas 500 MB kompetisi. Bobot pretrained ImageNet-22k → 1k dari timm memberi akurasi top-1 ImageNet-1K 84,2%, lebih tinggi dari bobot torchvision ImageNet-1K (82,5%) dengan arsitektur yang identik.

Aliran data untuk input 288 × 288:

| Bagian | Isi | Ukuran keluaran |
|--------|-----|-----------------|
| Stem | Konvolusi 4 × 4 stride 4, lalu Layer Normalization (LayerNorm) | 72 × 72 × 96 |
| Stage 1 | 3 blok ConvNeXt | 72 × 72 × 96 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 36 × 36 × 192 |
| Stage 2 | 3 blok ConvNeXt | 36 × 36 × 192 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 18 × 18 × 384 |
| Stage 3 | 9 blok ConvNeXt | 18 × 18 × 384 |
| Downsample | LayerNorm, konvolusi 2 × 2 stride 2 | 9 × 9 × 768 |
| Stage 4 | 3 blok ConvNeXt | 9 × 9 × 768 |
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
| Augmentasi | `RandomResizedCrop(288, scale=(0.5, 1.0))`, `RandomHorizontalFlip`, `RandomVerticalFlip(0.2)`, `TrivialAugmentWide`, `RandomErasing(0.25)` |
| MixUp / CutMix | Diterapkan pada 50% batch (MixUp alpha 0,2 atau CutMix alpha 1,0, dipilih acak) |
| Optimizer | AdamW, learning rate backbone 1e-4, head 1e-3, weight decay 0,05 (tanpa weight decay untuk bias, norm, dan layer scale) |
| Scheduler | Warmup linear 1 epoch lalu cosine annealing per step, 25 epoch |
| Regularisasi | Stochastic depth (drop path) 0,1 |
| Loss | Cross entropy dengan label smoothing 0,1 untuk meredam label noise |
| EMA | Exponential moving average bobot, decay 0,998, bobot EMA yang dievaluasi dan disimpan |
| Mixed precision | AMP (`torch.autocast` + `GradScaler`) di GPU |
| Pemilihan model | Run validasi: EMA dengan Macro-F1 tertinggi di split lokal (dengan TTA flip). Run final (`--full`): dilatih ulang di seluruh data dengan jumlah epoch dari run validasi, EMA epoch terakhir disimpan |

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | 0.9751 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | 0.9767 (run validasi) | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

* Preview 1: ConvNeXt-Tiny pretrained, fine-tuning dua tahap (head 3 epoch, full 10 epoch). AdamW lr 1e-4, weight decay 0.05, cosine annealing, label smoothing 0.1. Best checkpoint di epoch 4 tahap 2. Macro-F1 0.9751, akurasi 0.9780. Kelas terlemah: kertas_kardus (F1 0.9423), paling sering salah prediksi ke gelas_plastik dan styrofoam.
* Final: arsitektur tetap ConvNeXt-Tiny. Perubahannya: bobot pretrained ImageNet-22k → 1k (timm), resolusi 224 → 288, satu tahap 25 epoch dengan warmup + cosine, augmentasi lebih kuat (TrivialAugment, RandomErasing, MixUp/CutMix), drop path 0,1, EMA bobot, dan TTA flip horizontal. Run validasi: Macro-F1 0.9767 di epoch 13 (akurasi 0.9780), F1 kertas_kardus 0.9615; setelah epoch 13 skor turun ke sekitar 0.965 sehingga training 25 epoch terlalu lama. Model final dilatih ulang di seluruh data (my_train + my_val) selama 13 epoch. Dari 9 gambar my_val yang salah prediksi, sebagian besar berlabel meragukan (poster daur ulang berlabel kertas_kardus, gelas "polystyrene" yang tampak seperti plastik bening).

## Bobot Model

Bobot disimpan di `submission/model.pth` sebagai `state_dict` dan dimuat oleh `evaluate.py` dengan `load_state_dict(strict=True)`.

* State dict disimpan dalam format fp16 agar ukuran file di bawah 100 MB sehingga tidak perlu Git LFS. `evaluate.py` tetap bisa memuat bobot fp16 karena `load_state_dict` otomatis menyesuaikan dtype.
* Ukuran: sekitar 53 MB (fp16). Versi fp32 sekitar 106 MB.
* Key `state_dict` mengikuti timm `convnext_tiny`. `get_model()` membangun kelas `ConvNeXtTTA` (subclass `timm.models.convnext.ConvNeXt`) tanpa mengunduh bobot.

## Cara Reproduksi

```bash
pip install -r requirements.txt
# Run validasi: pilih epoch terbaik di my_val, cetak gambar yang salah prediksi
python train.py --data ./train --dump-errors
python evaluate.py --submission ./submission --data ./my_val
# Run final: latih ulang di my_train + my_val dengan jumlah epoch dari run validasi
python train.py --data ./train --full --epochs <N>
```

`train.py` membuat split lokal `my_train/` dan `my_val/`, lalu menyimpan bobot ke `submission/model.pth`. Bobot pretrained `convnext_tiny.fb_in22k_ft_in1k` diunduh otomatis oleh timm saat pelatihan.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() untuk ConvNeXt-Tiny + TTA
  model.pth         # state_dict model terbaik
  requirements.txt  # paket tambahan (timm)
evaluate.py         # salinan skrip evaluasi resmi, tidak diubah
train.py            # skrip pelatihanrequirements.txt    # paket untuk pelatihan dan evaluasi
README.md
```

Folder `train/`, `my_train/`, dan `my_val/` hanya ada di komputer lokal dan diabaikan oleh Git.