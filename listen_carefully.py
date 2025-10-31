import streamlit as st
import numpy as np
import urllib.request
import os
import pandas as pd
import plotly.express as px
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


# Abstract Base Class for Data Processing
class DataProcessor(ABC):
    """Abstract base class for data processing strategies"""
    
    @abstractmethod
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data from source"""
        pass
    
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data"""
        pass
    
    @abstractmethod
    def find_similar(self, query: str, top_n: int) -> pd.DataFrame:
        """Find similar words based on query"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get processor name"""
        pass


# SQLite Database Manager
class DatabaseManager:
    """Handles SQLite database operations"""
    
    def __init__(self, db_name: str = "dictionary.db"):
        self.db_name = db_name
        self.conn = None
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            
    def create_table(self):
        """Create dictionary table if not exists"""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dictionary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    definition_clean TEXT
                )
            """)
            
    def insert_data(self, df: pd.DataFrame):
        """Insert dataframe into database"""
        df.to_sql('dictionary', self.conn, if_exists='replace', index=False)
        
    def load_data(self) -> pd.DataFrame:
        """Load data from database"""
        return pd.read_sql_query("SELECT * FROM dictionary", self.conn)
    
    def search_words(self, pattern: str) -> pd.DataFrame:
        """Search words by pattern"""
        query = "SELECT * FROM dictionary WHERE word LIKE ?"
        return pd.read_sql_query(query, self.conn, params=(f"%{pattern}%",))


# Scikit-learn Based Processor
class SklearnProcessor(DataProcessor):
    """TF-IDF processing using scikit-learn"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.df = None
        self.db_manager = DatabaseManager()
        
    def get_name(self) -> str:
        return "Scikit-learn"
        
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data from CSV file"""
        df = pd.read_csv(source, encoding='utf-8', on_bad_lines='skip')
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        # Filter null definitions
        df = df[df['definition'].notna()].copy()
        
        # Clean definitions
        df['definition_clean'] = df['definition'].apply(self._clean_text)
        
        # Remove empty definitions
        df = df[df['definition_clean'].str.len() > 0].reset_index(drop=True)
        
        # Store in SQLite
        with self.db_manager as db:
            db.create_table()
            db.insert_data(df)
        
        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(df['definition_clean'])
        self.df = df
        
        return df
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing special characters"""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def find_similar(self, query: str, top_n: int = 15) -> pd.DataFrame:
        """Find similar words based on query"""
        # Clean query
        query_clean = self._clean_text(query)
        
        # Transform query
        query_vector = self.vectorizer.transform([query_clean])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'Rank': range(1, len(top_indices) + 1),
            'Word': self.df.iloc[top_indices]['word'].values,
            'Similarity': similarities[top_indices],
            'Definition': self.df.iloc[top_indices]['definition'].values
        })
        
        return results


# PySpark Processor (for environments with Java)
class PySparkProcessor(DataProcessor):
    """TF-IDF processing using PySpark (requires Java)"""
    
    def __init__(self):
        self.spark = None
        self.df_tfidf = None
        self.tokenizer = None
        self.remover = None
        self.hashingTF = None
        self.idf_model = None
        self.available = False
        self.error_message = None
        
        try:
            from pyspark.sql import SparkSession
            from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
            from pyspark.sql.functions import col, lower, regexp_replace
            from pyspark.sql.types import FloatType
            
            # Store imports for later use
            self.SparkSession = SparkSession
            self.HashingTF = HashingTF
            self.IDF = IDF
            self.Tokenizer = Tokenizer
            self.StopWordsRemover = StopWordsRemover
            self.col = col
            self.lower = lower
            self.regexp_replace = regexp_replace
            self.FloatType = FloatType
            self.udf = __import__('pyspark.sql.functions', fromlist=['udf']).udf
            
            # Try to initialize Spark
            self.spark = SparkSession.builder \
                .appName("WordDefinitionRecommender") \
                .config("spark.driver.memory", "4g") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            
            self.available = True
            
        except Exception as e:
            self.available = False
            self.error_message = str(e)
    
    def get_name(self) -> str:
        return "PySpark"
    
    def is_available(self) -> bool:
        """Check if PySpark is available"""
        return self.available
    
    def load_data(self, source: str) -> pd.DataFrame:
        """Load data using Spark"""
        if not self.available:
            raise RuntimeError(f"PySpark not available: {self.error_message}")
        
        df = self.spark.read.csv(source, header=True, inferSchema=True, escape='"')
        df = df.filter(self.col("definition").isNotNull())
        
        return df.toPandas()
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data using Spark ML"""
        if not self.available:
            raise RuntimeError(f"PySpark not available: {self.error_message}")
        
        # Convert pandas to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        # Clean and prepare the data
        spark_df = spark_df.filter(self.col("definition").isNotNull()) \
            .withColumn("definition_clean",
                        self.lower(self.regexp_replace(self.col("definition"), "[^a-zA-Z\\s]", "")))
        
        # Tokenize the definitions
        self.tokenizer = self.Tokenizer(inputCol="definition_clean", outputCol="words")
        df_tokenized = self.tokenizer.transform(spark_df)
        
        # Remove stop words
        self.remover = self.StopWordsRemover(inputCol="words", outputCol="filtered_words")
        df_filtered = self.remover.transform(df_tokenized)
        
        # Apply TF-IDF
        self.hashingTF = self.HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
        df_tf = self.hashingTF.transform(df_filtered)
        
        idf = self.IDF(inputCol="raw_features", outputCol="features")
        self.idf_model = idf.fit(df_tf)
        self.df_tfidf = self.idf_model.transform(df_tf)
        
        # Cache the result
        self.df_tfidf.cache()
        
        return df
    
    def find_similar(self, query: str, top_n: int = 15) -> pd.DataFrame:
        """Find similar words using Spark"""
        if not self.available:
            raise RuntimeError(f"PySpark not available: {self.error_message}")
        
        # Process the query definition
        query_df = self.spark.createDataFrame([(query,)], ["definition"])
        query_df = query_df.withColumn("definition_clean",
                                         self.lower(self.regexp_replace(self.col("definition"), "[^a-zA-Z\\s]", "")))
        
        # Apply the same transformations
        query_tokenized = self.tokenizer.transform(query_df)
        query_filtered = self.remover.transform(query_tokenized)
        query_tf = self.hashingTF.transform(query_filtered)
        query_tfidf = self.idf_model.transform(query_tf)
        
        # Get the query vector
        query_vector = query_tfidf.select("features").first()[0]
        
        # Define UDF for similarity calculation
        def cosine_similarity_spark(vec1, vec2):
            """Calculate cosine similarity between two sparse vectors"""
            dot_product = float(vec1.dot(vec2))
            norm1 = float(np.sqrt(vec1.dot(vec1)))
            norm2 = float(np.sqrt(vec2.dot(vec2)))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        similarity_udf = self.udf(lambda x: cosine_similarity_spark(x, query_vector), self.FloatType())
        
        # Calculate similarities
        results = self.df_tfidf.withColumn("similarity", similarity_udf(self.col("features"))) \
            .select("word", "definition", "similarity") \
            .orderBy(self.col("similarity").desc()) \
            .limit(top_n)
        
        # Convert to pandas
        results_pd = results.toPandas()
        results_pd['Rank'] = range(1, len(results_pd) + 1)
        results_pd.columns = ['Word', 'Definition', 'Similarity', 'Rank']
        results_pd = results_pd[['Rank', 'Word', 'Similarity', 'Definition']]
        
        return results_pd


# Processor Factory
class ProcessorFactory:
    """Factory for creating processor instances"""
    
    @staticmethod
    def create_processor(prefer_pyspark: bool = True) -> Tuple[DataProcessor, str]:
        """
        Create a processor instance, preferring PySpark if available
        
        Returns:
            Tuple of (processor instance, processor type string)
        """
        if prefer_pyspark:
            # Try PySpark first
            try:
                pyspark_processor = PySparkProcessor()
                if pyspark_processor.is_available():
                    return pyspark_processor, "pyspark"
                else:
                    # PySpark not available, fall back to sklearn
                    st.info("ℹ️ PySpark not available (Java not found). Using Scikit-learn instead.")
                    return SklearnProcessor(), "sklearn"
            except Exception as e:
                st.warning(f"⚠️ Failed to initialize PySpark: {str(e)}. Using Scikit-learn instead.")
                return SklearnProcessor(), "sklearn"
        else:
            return SklearnProcessor(), "sklearn"


# Main Application Class
class WordRecommenderApp:
    """Main application controller"""
    
    def __init__(self):
        self.processor: Optional[DataProcessor] = None
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.processor = None
            st.session_state.processor_type = None
            st.session_state.data_loaded = False
    
    def setup(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="Word Definition Recommender",
            layout="wide",
            page_icon="📚"
        )
        
        st.title("📚 Word Definition Recommender")
        st.markdown("### Find similar words using TF-IDF and Machine Learning")
    
    @staticmethod
    @st.cache_data
    def download_dataset() -> str:
        """Download the CSV file if it doesn't exist"""
        csv_url = "https://raw.githubusercontent.com/benjihillard/English-Dictionary-Database/main/english%20Dictionary.csv"
        csv_filename = "english_dictionary.csv"
        
        if not os.path.exists(csv_filename):
            urllib.request.urlretrieve(csv_url, csv_filename)
        return csv_filename
    
    def load_and_prepare_data(self, processor: DataProcessor, csv_file: str):
        """Load and prepare data"""
        with st.spinner("📥 Loading data..."):
            df = processor.load_data(csv_file)
            st.info(f"Loaded {len(df)} words from dictionary")
        
        with st.spinner(f"🔄 Preparing data and building TF-IDF model using {processor.get_name()}..."):
            processor.prepare_data(df)
            st.success(f"✅ Data preparation complete using {processor.get_name()}!")
        
        return processor
    
    def render_sidebar(self) -> Tuple[str, int, bool]:
        """Render sidebar and return settings"""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Show current processor
            if st.session_state.processor_type:
                processor_name = "PySpark" if st.session_state.processor_type == "pyspark" else "Scikit-learn"
                st.info(f"🔧 Active Engine: **{processor_name}**")
            
            # Manual processor selection
            with st.expander("🔧 Advanced: Change Processor"):
                force_sklearn = st.checkbox(
                    "Force use Scikit-learn",
                    help="Override automatic detection and use Scikit-learn instead of PySpark"
                )
                
                if st.button("🔄 Reinitialize Processor"):
                    st.session_state.initialized = False
                    st.session_state.processor = None
                    st.session_state.processor_type = None
                    st.rerun()
            
            top_n = st.slider(
                "Number of results",
                min_value=5,
                max_value=50,
                value=15,
                step=5
            )
            
            st.markdown("---")
            st.header("🔍 Query Input")
            
            query_option = st.radio(
                "Select query type:",
                ["Custom Query", "Ongoing Event", "Teacher Definition", "Fast Movement"],
                index=0
            )
            
            query_examples = {
                "Ongoing Event": "occurring or being done at the present moment; happening now; current event",
                "Teacher Definition": "a person who teaches students in a school or educational institution",
                "Fast Movement": "the act of moving quickly on foot from one place to another"
            }
            
            if query_option in query_examples:
                query = st.text_area(
                    "Query Preview",
                    query_examples[query_option],
                    height=150,
                    disabled=True
                )
            else:
                query = st.text_area(
                    "Enter your query definition",
                    height=200,
                    placeholder="Enter a definition or text to find similar words...\n\nExamples:\n- 'Something happening right now'\n- 'A person who teaches students'\n- 'The act of moving quickly on foot'"
                )
            
            submit_button = st.button(
                "🔍 Find Similar Words",
                type="primary",
                use_container_width=True
            )
            
            # Return force_sklearn as well
            return query, top_n, submit_button, force_sklearn if 'force_sklearn' in locals() else False
    
    def display_results(self, results: pd.DataFrame):
        """Display search results with visualizations"""
        st.success(f"✅ Found {len(results)} similar words!")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "📋 Data Table"])
        
        with tab1:
            st.subheader("Similarity Scores - Bar Chart")
            fig_bar = px.bar(
                results,
                x='Word',
                y='Similarity',
                color='Similarity',
                color_continuous_scale='Viridis',
                hover_data=['Definition'],
                text='Similarity'
            )
            fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_bar.update_layout(
                height=500,
                xaxis_title="Word",
                yaxis_title="Similarity Score",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            st.subheader("Similarity Scores - Line Chart")
            fig_line = px.line(
                results,
                x='Rank',
                y='Similarity',
                markers=True,
                hover_data=['Word', 'Definition']
            )
            fig_line.update_traces(
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=10)
            )
            fig_line.update_layout(
                height=500,
                xaxis_title="Rank",
                yaxis_title="Similarity Score"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with tab3:
            st.subheader("Detailed Results Table")
            styled_df = results.style.background_gradient(
                subset=['Similarity'],
                cmap='YlGnBu'
            ).format({'Similarity': '{:.4f}'})
            
            st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Statistics
        self._display_statistics(results)
        
        # Download option
        self._display_download_button(results)
    
    @staticmethod
    def _display_statistics(results: pd.DataFrame):
        """Display statistics metrics"""
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Top Word", results.iloc[0]['Word'])
        with col2:
            st.metric("Max Similarity", f"{results['Similarity'].max():.4f}")
        with col3:
            st.metric("Avg Similarity", f"{results['Similarity'].mean():.4f}")
        with col4:
            st.metric("Min Similarity", f"{results['Similarity'].min():.4f}")
    
    @staticmethod
    def _display_download_button(results: pd.DataFrame):
        """Display download button for results"""
        st.markdown("---")
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="similar_words_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def run(self):
        """Main application loop"""
        self.setup()
        
        # Render sidebar and get settings
        que
