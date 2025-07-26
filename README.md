# Sentiment Analysis dari Wawancara Audio

## Proyek ini melakukan analisis sentimen secara otomatis dari rekaman wawancara berbahasa Indonesia. Sistem ini dapat:

- Mentranskripsikan audio wawancara (format .ogg, .mp3, dll) menjadi teks
- Menganalisis sentimen dari teks transkripsi
- Mengekstrak kata kunci penting
- Membuat visualisasi hasil analisis (word cloud dan distribusi sentimen)

## Fitur Utama

- Transkripsi Audio: Menggunakan model Whisper OpenAI untuk konversi audio ke teks
- Analisis Sentimen: Menggunakan model RoBERTa khusus bahasa Indonesia
- Ekstraksi Kata Kunci: Menggunakan TF-IDF dengan stopwords bahasa Indonesia
- Word Cloud: Visualisasi kata-kata penting dalam wawancara
- Visualisasi Sentimen: Distribusi sentimen dalam bentuk grafik batang
- Export Otomatis: Menyimpan hasil dalam format CSV dan gambar

## Output yang Dihasilkan

Program akan menghasilkan file-file berikut:
| file | Direktori | Deskripsi |
| --- | --- | --- |
| `transkrip.txt` | `data/` | Teks hasil transkripsi audio wawancara |
| `keywords.csv` | `data/` | Kata kunci penting dari teks |
| `processed_reviews.csv` | `data/` | Teks yang telah diproses (cleaned) |
| `wordcloud.png` | `plots/` | Visualisasi word cloud dari teks |
| `sentiment_distribution.png` | `plots/` | Grafik distribusi sentimen |

## Kelas dan Fungsi Utama

### AudioTranscriber

#### Menangani transkripsi audio ke teks:

- `transkrip_audio(file_path)`: Transkripsi file audio
- `save_transcription()`: Menyimpan hasil transkripsi ke file

### ReviewProcessor

#### Melakukan analisis teks:

- `preprocess_text()`: Membersihkan dan memproses teks
- `label_sentiment()`: Menganalisis sentimen teks
- `extract_keywords()`: Mengekstrak kata kunci penting
- `generate_wordcloud()`: Membuat visualisasi word cloud
- `plot_sentiment_distribution()`: Membuat grafik distribusi sentimen

## Proses Flow

1. Pengguna mengunggah file audio wawancara.
2. Sistem melakukan transkripsi audio ke teks.
3. Teks hasil transkripsi diproses untuk analisis lebih lanjut.
4. Sistem menganalisis sentimen dari teks.
5. Kata kunci penting diekstrak dari teks.
6. Visualisasi hasil analisis dibuat (word cloud dan distribusi sentimen).
7. Hasil akhir disimpan dalam format yang ditentukan.

## Visualisasi

![processing flow](/images/proses.svg)
