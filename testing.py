import whisper
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from transformers import pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class AudioTranscriber:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def transkrip_audio(self, file_path):
        result = self.model.transcribe(file_path, language="id")
        return result["text"]
    
    def transcribe_long_audio(self, file_path, segment_length):
        result = self.model.transcribe(file_path, language="id", segment_length=segment_length)
        return result["text"]

    def save_transcription(self, file_path, output_path, filename="text_wawancara.txt"):
        transcription = self.transkrip_audio(file_path)
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"Transcription saved to {full_path}")
        return full_path

class ReviewProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords_id = StopWordRemoverFactory().get_stop_words()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = self.stopword_remover.remove(text)
        text = self.stemmer.stem(text) # Stemming
        return text

    def process_reviews(self):
        self.df['review'] = self.df['review'].fillna('')
        self.df['cleaned_review'] = self.df['review'].apply(self.preprocess_text)
        return self

    def label_sentiment(self):
        def get_sentiment_id(text):
            if len(text) < 3:
                return "neutral"
            try:
                result = self.sentiment_analyzer(text)[0]
                return result['label']
            except:
                return "neutral"
        
        self.df['sentiment'] = self.df['review'].apply(get_sentiment_id)
        return self

    def extract_keywords(self, max_features=10, use_tfidf=True):
        if use_tfidf:
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words=self.stopwords_id)
        else:
            vectorizer = CountVectorizer(max_features=max_features, stop_words=self.stopwords_id)
        
        X = vectorizer.fit_transform(self.df['cleaned_review'])
        keywords = vectorizer.get_feature_names_out()
        
        def get_keywords_for_row(row):
            row_vector = vectorizer.transform([row])
            indices = row_vector.toarray().flatten().argsort()[-max_features:][::-1]
            return [keywords[i] for i in indices]
        
        self.df['keywords'] = self.df['cleaned_review'].apply(get_keywords_for_row)
        return self
    
    def generate_wordcloud(self, output_dir, filename='wordcloud.png'):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        text = " ".join(self.df['cleaned_review'].tolist())
        wordcloud = WordCloud(
            width=1200, 
            height=600,
            background_color='white',
            stopwords=self.stopwords_id,
            collocations=False
        ).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Reviews", fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"WordCloud saved to {save_path}")
        return self

    def save_keywords_to_csv(self, output_path, filename="keywords.csv"):
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        keywords_df = self.df[['speaker', 'review', 'keywords', 'sentiment']]
        keywords_df.to_csv(full_path, index=False)
        print(f"Keywords saved to {full_path}")
        return self
    
    def save_processed_reviews(self, output_path, filename="processed_reviews.csv"):
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        self.df.to_csv(full_path, index=False)
        print(f"Processed reviews saved to {full_path}")
        return full_path
    
    def plot_sentiment_distribution(self, output_dir, filename='sentiment_distribution.png'):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
        
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(
            kind='bar', 
            color=[colors.get(s, 'gray') for s in sentiment_counts.index]
        )
        
        plt.title('Distribusi Sentimen', fontsize=16)
        plt.xlabel('Sentimen', fontsize=12)
        plt.ylabel('Jumlah', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tambah label nilai di atas bar
        for i, count in enumerate(sentiment_counts):
            plt.text(i, count + 0.1, str(count), ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Sentiment distribution plot saved to {save_path}")
        return self

if __name__ == "__main__":
    audio_dir = "media/audio"
    data_dir = "media/data"
    text_dir = "media/text"
    plots_dir = "media/plots"
    
    for dir_path in [audio_dir, data_dir, text_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    transcriber = AudioTranscriber("base")
    audio_file = os.path.join(audio_dir, "wawancara.ogg")
    
    if os.path.exists(audio_file):
        teks_wawancara = transcriber.transkrip_audio(audio_file)
        transcriber.save_transcription(audio_file, data_dir, "transkrip.txt")
    else:
        print(f"File audio tidak ditemukan: {audio_file}")
        teks_wawancara = "Wawancara tidak tersedia"
    
    df = pd.DataFrame({
        'speaker': ['Responden 1'],
        'review': [teks_wawancara],
        'timestamp': [pd.Timestamp.now()]
    })
    
    # Proses analisis
    processor = ReviewProcessor(df)
    
    processor.process_reviews()\
             .label_sentiment()\
             .extract_keywords(max_features=15, use_tfidf=True)\
             .generate_wordcloud(plots_dir, "wordcloud.png")\
             .plot_sentiment_distribution(plots_dir, "sentiment_distribution.png")\
             .save_keywords_to_csv(data_dir, "keywords.csv")\
             .save_processed_reviews(data_dir, "processed_reviews.csv")
    
    print("Processing complete.")