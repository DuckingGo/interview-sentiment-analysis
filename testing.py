"""
Audio Transcription and Text Analysis Tool

This module provides tools for transcribing audio files and analyzing text reviews
with sentiment analysis, keyword extraction, and visualization capabilities.

The module contains two main classes:
1. AudioTranscriber: For transcribing audio files using OpenAI Whisper
2. ReviewProcessor: For processing and analyzing text reviews

Dependencies:
    - whisper: For audio transcription
    - pandas: For data manipulation
    - matplotlib: For plotting and visualization
    - scikit-learn: For text vectorization
    - wordcloud: For word cloud generation
    - transformers: For sentiment analysis
    - Sastrawi: For Indonesian text preprocessing

Author: David
Date: July 2025
"""

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
    """
    A class for transcribing audio files using OpenAI Whisper model.
    
    This class provides functionality to transcribe audio files with support for
    long audio files by segmenting them into manageable parts. It uses the 
    Indonesian language model for transcription and can save results to text files.
    
    Attributes:
        model: The loaded Whisper model for transcription
        
    Args:
        model_name (str): Name of the Whisper model to use. Options include:
                         "tiny", "base", "small", "medium", "large"
                         
    Example:
        >>> transcriber = AudioTranscriber("medium")
        >>> text = transcriber.transkrip_audio("audio.wav")
        >>> transcriber.save_transcription("audio.wav", "output/", "transcript.txt")
    """
    def __init__(self, model_name):
        """
        Initialize the AudioTranscriber with a specified Whisper model.
        
        Args:
            model_name (str): Name of the Whisper model to load
        """
        self.model = whisper.load_model(model_name)

    def transkrip_audio(self, file_path):
        """
        Transcribe audio from a file using Indonesian language model.
        
        Args:
            file_path (str): Path to the audio file to transcribe
            
        Returns:
            str: The transcribed text from the audio file
            
        Example:
            >>> transcriber = AudioTranscriber("base")
            >>> text = transcriber.transkrip_audio("interview.wav")
            >>> print(text)
        """
        result = self.model.transcribe(file_path, language="id")
        return result["text"]
    
    def transcribe_long_audio(self, file_path, segment_length):
        """
        Transcribe long audio files by segmenting them into smaller parts.
        
        This method is useful for very long audio files that might cause
        memory issues or timeout errors when processed as a whole.
        
        Args:
            file_path (str): Path to the audio file to transcribe
            segment_length (int): Length of each segment in seconds
            
        Returns:
            str: The complete transcribed text from the segmented audio
            
        Example:
            >>> transcriber = AudioTranscriber("medium")
            >>> text = transcriber.transcribe_long_audio("long_interview.wav", 300)
        """
        result = self.model.transcribe(file_path, language="id", segment_length=segment_length)
        return result["text"]

    def save_transcription(self, file_path, output_path, filename="text_wawancara.txt"):
        """
        Transcribe audio and save the result to a text file.
        
        This method combines transcription and file saving in one operation.
        It creates the output directory if it doesn't exist.
        
        Args:
            file_path (str): Path to the audio file to transcribe
            output_path (str): Directory where the transcription will be saved
            filename (str, optional): Name of the output text file. 
                                    Defaults to "text_wawancara.txt"
                                    
        Returns:
            str: Full path to the saved transcription file
            
        Example:
            >>> transcriber = AudioTranscriber("base")
            >>> saved_path = transcriber.save_transcription(
            ...     "interview.wav", 
            ...     "output/transcripts/", 
            ...     "interview_transcript.txt"
            ... )
        """
        transcription = self.transkrip_audio(file_path)
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"Transcription saved to {full_path}")
        return full_path

class ReviewProcessor:
    """
    A comprehensive class for processing and analyzing text reviews in Indonesian.
    
    This class provides a complete pipeline for text analysis including preprocessing,
    sentiment analysis, keyword extraction, word cloud generation, and visualization.
    It uses Indonesian-specific NLP tools and pre-trained models for accurate analysis.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing the reviews data
        sentiment_analyzer: HuggingFace pipeline for Indonesian sentiment analysis
        stopword_remover: Sastrawi stop word remover for Indonesian text
        stemmer: Sastrawi stemmer for Indonesian text
        stopwords_id: Set of Indonesian stop words
        
    Args:
        df (pd.DataFrame): DataFrame containing reviews with required columns:
                          'speaker', 'review', and 'timestamp'
                          
    Example:
        >>> df = pd.DataFrame({
        ...     'speaker': ['User1', 'User2'],
        ...     'review': ['Produk bagus sekali', 'Tidak puas dengan layanan'],
        ...     'timestamp': [pd.Timestamp.now(), pd.Timestamp.now()]
        ... })
        >>> processor = ReviewProcessor(df)
        >>> processor.process_reviews().label_sentiment().extract_keywords()
    """
    def __init__(self, df):
        """
        Initialize the ReviewProcessor with a DataFrame containing reviews.
        
        Sets up Indonesian NLP tools including sentiment analyzer, stop word remover,
        and stemmer for text preprocessing.
        
        Args:
            df (pd.DataFrame): DataFrame with columns 'speaker', 'review', 'timestamp'
        """
        self.df = df.copy()
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords_id = StopWordRemoverFactory().get_stop_words()

    def preprocess_text(self, text):
        """
        Clean and preprocess Indonesian text for analysis.
        
        Performs the following preprocessing steps:
        1. Convert to lowercase
        2. Remove punctuation and special characters
        3. Remove Indonesian stop words
        4. Apply stemming to reduce words to their root form
        
        Args:
            text (str): Raw text to be preprocessed
            
        Returns:
            str: Cleaned and preprocessed text
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> clean_text = processor.preprocess_text("Saya sangat menyukai produk ini!")
            >>> print(clean_text)  # "suka produk"
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = self.stopword_remover.remove(text)
        text = self.stemmer.stem(text) # Stemming
        return text

    def process_reviews(self):
        """
        Apply text preprocessing to all reviews in the DataFrame.
        
        Creates a new column 'cleaned_review' containing the preprocessed text.
        Handles missing values by filling them with empty strings.
        
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.process_reviews()
            >>> print(processor.df['cleaned_review'].head())
        """
        self.df['review'] = self.df['review'].fillna('')
        self.df['cleaned_review'] = self.df['review'].apply(self.preprocess_text)
        return self

    def label_sentiment(self):
        """
        Analyze and label the sentiment of each review using a pre-trained model.
        
        Uses the Indonesian RoBERTa-based sentiment classifier to determine
        sentiment (positive, negative, or neutral) for each review.
        Handles short texts and errors gracefully by defaulting to 'neutral'.
        
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.process_reviews().label_sentiment()
            >>> print(processor.df['sentiment'].value_counts())
        """
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
        """
        Extract keywords from cleaned reviews using vectorization techniques.
        
        Uses either TF-IDF or Count Vectorization to identify the most important
        keywords for each review. The keywords are ranked by their importance
        scores and stored in a new 'keywords' column.
        
        Args:
            max_features (int, optional): Maximum number of features/keywords to extract.
                                        Defaults to 10.
            use_tfidf (bool, optional): If True, uses TF-IDF Vectorization; 
                                      otherwise uses Count Vectorization. 
                                      Defaults to True.
                                      
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.process_reviews().extract_keywords(max_features=15)
            >>> print(processor.df['keywords'].head())
        """
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
        """
        Generate and save a word cloud visualization from cleaned reviews.
        
        Creates a word cloud image showing the most frequent words in the reviews,
        with word size proportional to frequency. Uses Indonesian stop words
        to filter out common words.
        
        Args:
            output_dir (str): Directory where the word cloud image will be saved
            filename (str, optional): Name of the output image file. 
                                     Defaults to 'wordcloud.png'
                                     
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.process_reviews().generate_wordcloud(
            ...     "output/plots/", 
            ...     "review_wordcloud.png"
            ... )
        """
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
        """
        Save extracted keywords along with review metadata to a CSV file.
        
        Exports a subset of the DataFrame containing speaker, review, keywords,
        and sentiment columns to a CSV file for further analysis.
        
        Args:
            output_path (str): Directory where the CSV file will be saved
            filename (str, optional): Name of the output CSV file. 
                                     Defaults to "keywords.csv"
                                     
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.extract_keywords().save_keywords_to_csv("output/data/")
        """
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        keywords_df = self.df[['speaker', 'review', 'keywords', 'sentiment']]
        keywords_df.to_csv(full_path, index=False)
        print(f"Keywords saved to {full_path}")
        return self
    
    def save_processed_reviews(self, output_path, filename="processed_reviews.csv"):
        """
        Save the complete processed DataFrame to a CSV file.
        
        Exports the entire DataFrame including all original columns plus
        any new columns created during processing (cleaned_review, sentiment, keywords).
        
        Args:
            output_path (str): Directory where the CSV file will be saved
            filename (str, optional): Name of the output CSV file. 
                                     Defaults to "processed_reviews.csv"
                                     
        Returns:
            str: Full path to the saved CSV file
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> path = processor.process_reviews().save_processed_reviews("output/data/")
            >>> print(f"Data saved to: {path}")
        """
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
        self.df.to_csv(full_path, index=False)
        print(f"Processed reviews saved to {full_path}")
        return full_path
    
    def plot_sentiment_distribution(self, output_dir, filename='sentiment_distribution.png'):
        """
        Create and save a bar chart showing the distribution of sentiments.
        
        Generates a bar chart visualization showing the count of positive, negative,
        and neutral sentiments in the reviews. Each bar is color-coded and includes
        count labels for clarity.
        
        Args:
            output_dir (str): Directory where the plot image will be saved
            filename (str, optional): Name of the output image file. 
                                     Defaults to 'sentiment_distribution.png'
                                     
        Returns:
            ReviewProcessor: Self for method chaining
            
        Example:
            >>> processor = ReviewProcessor(df)
            >>> processor = processor.label_sentiment().plot_sentiment_distribution(
            ...     "output/plots/", 
            ...     "sentiment_chart.png"
            ... )
        """
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
    """
    Main execution block for audio transcription and text analysis pipeline.
    
    This script demonstrates the complete workflow:
    1. Set up directory structure for organizing outputs
    2. Transcribe audio file using Whisper
    3. Process and analyze the transcribed text
    4. Generate visualizations and save results
    
    The pipeline processes an interview audio file and performs comprehensive
    text analysis including sentiment analysis, keyword extraction, and visualization.
    """
    # Set up directory structure
    audio_dir = "media/audio"
    data_dir = "media/data"
    text_dir = "media/text"
    plots_dir = "media/plots"
    
    # Create directories if they don't exist
    for dir_path in [audio_dir, data_dir, text_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize transcriber with medium-sized Whisper model
    transcriber = AudioTranscriber("medium")
    audio_file = os.path.join(audio_dir, "wawancara.ogg")
    
    # Transcribe audio file if it exists
    if os.path.exists(audio_file):
        teks_wawancara = transcriber.transkrip_audio(audio_file)
        transcriber.save_transcription(audio_file, data_dir, "transkrip.txt")
    else:
        print(f"File audio tidak ditemukan: {audio_file}")
        teks_wawancara = "Wawancara tidak tersedia"
    
    # Create DataFrame with transcribed text
    df = pd.DataFrame({
        'speaker': ['Responden 1'],
        'review': [teks_wawancara],
        'timestamp': [pd.Timestamp.now()]
    })
    
    # Process and analyze the text using method chaining
    processor = ReviewProcessor(df)
    
    processor.process_reviews()\
             .label_sentiment()\
             .extract_keywords(max_features=15, use_tfidf=True)\
             .generate_wordcloud(plots_dir, "wordcloud.png")\
             .plot_sentiment_distribution(plots_dir, "sentiment_distribution.png")\
             .save_keywords_to_csv(data_dir, "keywords.csv")\
             .save_processed_reviews(data_dir, "processed_reviews.csv")
    
    print("Processing complete.")