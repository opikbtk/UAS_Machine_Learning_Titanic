# UAS Machine Learning - Titanic Survival Prediction

Repository ini dibuat untuk memenuhi Tugas UAS Mata Kuliah Machine Learning.
Proyek ini berisi implementasi algoritma **Decision Tree** untuk memprediksi keselamatan penumpang kapal Titanic, lengkap dengan analisis teori dan eksperimen.

## Isi Repository
1.  **Laporan**:
    *   `Laporan UAS Machine Learning.pdf`: Laporan lengkap mencakup jawaban teori, metodologi, dan analisis hasil.
2.  **Source Code**:
    *   `uas_titanic_model.ipynb`: **(Recommended)** Jupyter Notebook interaktif. Berisi kode, output, dan visualisasi dalam satu file.
    *   `uas_titanic_model.py`: Script Python standar jika ingin menjalankan via terminal.
3.  **Data**:
    *   `titanic.csv`: Dataset lengkap (raw).
    *   `titanic_train.csv`: Data latih (80% dari total).
    *   `titanic_test.csv`: Data uji (20% dari total).
4.  **Artifacts**:
    *   `titanic_tree_structure.png`: Gambar visualisasi pohon keputusan.
    *   `requirements.txt`: Daftar library Python yang dibutuhkan.

## Cara Menjalankan Code

### Cara 1: Menggunakan Jupyter Notebook (Disarankan)
1.  Pastikan VS Code dan Extension **Jupyter** sudah terinstall.
2.  Buka file `uas_titanic_model.ipynb`.
3.  Klik tombol **"Run All"** untuk menjalankan seluruh analisa.

### Cara 2: Menggunakan Terminal
1.  Install library yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```
2.  Jalankan script python:
    ```bash
    python uas_titanic_model.py
    ```
    *(Pastikan koneksi internet aktif saat run pertama kali untuk mendownload dataset)*

## Overview Hasil
Model Decision Tree (dengan `max_depth=3`) mencapai akurasi **~80%** pada data testing.
Faktor paling signifikan yang ditemukan oleh model adalah **Gender (Sex)**, diikuti oleh **Kelas Tiket (Pclass)** dan **Umur (Age)**. Detail lengkap bisa dibaca di `Laporan_UAS.md`.

---
**Identitas Mahasiswa:**
*   **Nama**: [Mohamad Taufik Wibowo]
*   **NIM**: [231011400164]
*   **Kelas**: [05 TPLE 004]
