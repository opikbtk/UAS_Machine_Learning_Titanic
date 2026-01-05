# Laporan UAS Machine Learning: Studi Kasus Titanic Survival Prediction

**Nama:** [Isi Nama Anda Disini]
**NIM:** [Isi NIM Anda Disini]
**Mata Kuliah:** Machine Learning
**Tanggal:** [Isi Tanggal]

---

## Bagian 1 – Pemahaman Konsep (Teori)

### 1. Apa yang dimaksud dengan Decision Tree?
Decision Tree (Pohon Keputusan) adalah metode pembelajaran terawasi (*supervised learning*) non-parametrik yang digunakan untuk tugas klasifikasi dan regresi. Secara intuitif, Decision Tree bekerja menyerupai logika pengambilan keputusan manusia, yaitu memecah masalah kompleks menjadi serangkaian keputusan sederhana yang berurutan.

Struktur model ini berbentuk flowchart menyerupai struktur pohon, di mana setiap:
*   **Internal Node** merepresentasikan tes atau pengujian pada sebuah atribut (fitur).
*   **Branch (Cabang)** merepresentasikan hasil dari pengujian tersebut.
*   **Leaf Node (Daun)** merepresentasikan label kelas (keputusan akhir) atau nilai numerik prediksi.

Tujuan utama algoritma ini adalah membuat model yang dapat memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang disimpulkan dari fitur data.

### 2. Penjelasan Konsep Inti:
*   **Node**: Unit dasar dari struktur pohon. Setiap node berisi kondisi atau aturan tertentu yang diterapkan pada data.
*   **Root (Akar)**: Node paling atas dari pohon keputusan. Root node adalah titik awal di mana seluruh dataset dievaluasi sebelum mengalami pemisahan apa pun. Algoritma akan mencari fitur yang paling signifikan (memiliki *Information Gain* tertinggi atau *Gini Impurity* terendah) untuk menjadi Root.
*   **Leaf (Daun)**: Node terminal yang tidak memiliki cabang lagi. Leaf node adalah hasil akhir dari proses penelusuran pohon, yang berisi prediksi kelas (misal: "Selamat" atau "Tidak Selamat").
*   **Splitting (Pemisahan)**: Proses membagi sebuah node menjadi dua atau lebih sub-node berdasarkan kondisi tertentu. Strategi splitting yang baik adalah yang menghasilkan sub-group yang se-homogen mungkin (*pure*).
*   **Pruning (Pemangkasan)**: Teknik untuk mengatasi overfitting dengan cara menghapus bagian pohon (cabang/sub-node) yang lemah atau tidak signifikan secara statistik. Pruning bisa dilakukan saat pohon dibangun (*pre-pruning*) atau setelah pohon selesai dibangun (*post-pruning*).

### 3. Perbedaan Decision Tree, Random Forest, dan Gradient Boosting
Ketiga algoritma ini masuk dalam keluarga *Tree-based Methods*, namun memiliki pendekatan berbeda:

| Fitur | Decision Tree (DT) | Random Forest (RF) | Gradient Boosting (GBM/XGBoost) |
| :--- | :--- | :--- | :--- |
| **Prinsip Dasar** | Algoritma tunggal sederhana. | *Ensemble Bagging* (Bootstrap Aggregating). | *Ensemble Boosting*. |
| **Konstruksi Model** | Membangun satu pohon secara mendalam. | Membangun banyak pohon secara paralel (independen). | Membangun pohon satu per satu secara sekuensial (bertahap). |
| **Cara Kerja** | Mencari split terbaik di setiap langkah glob. | Setiap pohon dilatih pada subset data acak, hasil diputuskan lewat *voting* (mayoritas). | Pohon baru ditambahkan untuk memperbaiki *error* (residual) dari pohon sebelumnya. |
| **Kelemahan Utama** | Mudah **Overfitting** (menghafal data) & tidak stabil (High Variance). | Komputasi lebih berat daripada DT tunggal. | Sensitif terhadap noise & outlier, parameter tuning lebih rumit. |
| **Kecenderungan Error** | Bias rendah, Varians tinggi. | Varians berkurang drastis (lebih stabil). | Bias berkurang drastis (sangat akurat). |

### 4. Kelebihan dan Kekurangan Tree-based Methods
**Kelebihan:**
*   **Interpretabilitas Tinggi**: Sangat mudah dimengerti, divisualisasikan, dan dijelaskan kepada pemangku kepentingan (*stakeholders*) non-teknis. Logikanya transparan ("White box model").
*   **Preprocessing Minimal**: Tidak memerlukan normalisasi atau scaling data (seperti Standarisasi/MinMax), karena algoritma ini berbasis aturan logis (rule-based) bukan jarak (distance-based).
*   **Robust terhadap Data Campuran**: Mampu menangani kombinasi tipe data numerik dan kategorikal dengan baik tanpa rekayasa fitur yang rumit.
*   **Feature Selection Otomatis**: Fitur yang tidak penting cenderung tidak dipilih sebagai splitter di bagian atas pohon.

**Kekurangan:**
*   **Instabilitas (Instability)**: Perubahan kecil pada data training bisa menghasilkan struktur pohon yang sangat berbeda.
*   **Overfitting**: Tanpa pembatasan (pruning), pohon cenderung tumbuh sangat dalam dan kompleks, menangkap noise dalam data sebagai pola.
*   **Bias pada Imbalanced Data**: Cenderung bias ke arah kelas dominan jika dataset tidak seimbang.

---

## Bagian 2 – Metodologi (Implementasi)

Implementasi dilakukan menggunakan bahasa pemrograman **Python** dengan library **Scikit-Learn**.

### Langkah Pengerjaan dan Justifikasi:
1.  **Data Acquisition**:
    Dataset diambil menggunakan library `seaborn` yang memuat dataset Titanic standar. Data ini kemudian disimpan lokal menjadi `titanic.csv` untuk keperluan dokumentasi.

2.  **Exploratory Data Analysis (EDA)**:
    Dilakukan pengecekan distribusi target `survived`. Ditemukan bahwa dataset tidak seimbang (lebih banyak korban meninggal dibanding selamat), namun rasionya masih wajar untuk klasifikasi standar.

3.  **Data Preprocessing (Pembersihan Data)**:
    Tahap ini sangat krusial untuk performa model:
    *   **Feature Selection**: Kolom seperti `deck` (terlalu banyak missing value >70%), `embark_town` (redundan dengan `embarked`), dan `who`/`adjult_male` (redundan dengan `sex`/`age`) dibuang untuk menyederhanakan model dan mencegah *data leakage*.
    *   **Missing Value Imputation**:
        *   `age`: Diisi menggunakan **median** (nilai tengah) karena distribusi umur biasanya *skewed* (miring), sehingga median lebih robust terhadap outlier dibanding mean.
        *   `embarked`: Diisi menggunakan **modus** (nilai terbanyak) karena merupakan data kategorikal.
    *   **Encoding**: Mengubah variabel kategorikal menjadi numerik agar bisa diproses mesin. `sex` (male/female) diubah menjadi 0/1, begitu juga dengan `embarked`.

4.  **Modeling Strategy**:
    *   Menggunakan `DecisionTreeClassifier` dengan parameter `criterion='gini'` (default standar efisien).
    *   **Hyperparameter Tuning**: Parameter `max_depth` diset ke **3**.
        *   *Alasan*: Membatasi kedalaman pohon adalah bentuk *Pruning* sederhana. Pohon yang terlalu dalam (depth > 5) pada dataset kecil seperti Titanic cenderung overfitting (hafal data latih tapi gagal di data uji). Depth 3 memberikan keseimbangan (trade-off) yang baik antara bias dan variance.

5.  **Evaluasi**:
    Data dibagi menjadi **80% Training** dan **20% Testing**. Evaluasi dilakukan menggunakan metrik Accuracy, Precision, Recall, dan F1-Score untuk mendapatkan gambaran performa yang komprehensif.

---

## Bagian 3 – Hasil Analisis dan Kesimpulan

### Analisis Performa Model
Berdasarkan eksperimen, model **Decision Tree (max_depth=3)** menghasilkan akurasi sekitar **80%** pada data testing.

**Interpretasi Confusion Matrix:**
Dari 179 data pengujian:
*   Model berhasil memprediksi dengan benar mayoritas penumpang yang tidak selamat (**True Negative** tinggi).
*   Kesalahan prediksi (**False Negative**) terjadi di mana model memprediksi "Meninggal" padahal aslinya "Selamat". Ini wajar karena fitur "Survival" pada Titanic memang memiliki unsur acak/keberuntungan yang sulit dipola hanya dengan fitur dasar.

### Analisis Visualisasi Pohon (Tree Interpretation)
Visualisasi pohon keputusan memberikan wawasan menarik (lihat file `titanic_tree_structure.png`):
1.  **Root Node (Faktor Terpenting)**: `sex` (Jenis Kelamin).
    *   Pemisahan pertama terjadi berdasarkan gender. Jika laki-laki (`sex > 0.5`), probabilitas selamat langsung turun drastis. Ini konsisten dengan aturan sejarah "Women and Children First".
2.  **Level Kedua**:
    *   Bagi populasi Wanita, faktor penentu berikutnya adalah `pclass` (Kelas Tiket). Wanita di Kelas 1 & 2 memiliki peluang selamat jauh lebih tinggi dibanding Kelas 3.
    *   Bagi populasi Pria, faktor umur (`age`) menjadi pembeda, di mana anak-anak laki-laki memiliki peluang selamat lebih baik dibanding pria dewasa.

### Kesimpulan Akhir
Studi kasus ini menunjukkan kekuatan algoritma Decision Tree dalam **Explainable AI**. Meskipun akurasi 80% mungkin bisa ditingkatkan lagi dengan Random Forest (menjadi ~83-85%), Decision Tree memberikan nilai tambah berupa **pemahaman "Mengapa"**.

Kita dapat menyimpulkan profil keselamatan Titanic dengan aturan sederhana:
> *"Wanita, terutama di kelas atas, memiliki peluang selamat tertinggi. Pria dewasa di kelas bawah memiliki peluang terendah."*

Kesimpulannya, Tree-based methods sangat cocok untuk kasus ini karena mampu menangkap interaksi non-linear antara Gender, Kelas Ekonomi, dan Umur tanpa memerlukan transformasi data yang rumit.
