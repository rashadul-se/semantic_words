import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf, lower, regexp_replace
from pyspark.sql.types import FloatType
import numpy as np
import urllib.request
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Word Definition Recommender", layout="wide", page_icon="📚")

# Title and description
st.title("📚 Word Definition Recommender")
st.markdown("### Find similar words using PySpark ML and TF-IDF")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df_tfidf = None
    st.session_state.spark = None
    st.session_state.models = None

@st.cache_resource
def initialize_spark():
    """Initialize Spark Session"""
    spark = SparkSession.builder \
        .appName("WordDefinitionRecommender") \
        .config("spark.driver.memory", "4g") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    return spark

@st.cache_data
def download_dataset():
    """Download the CSV file if it doesn't exist"""
    csv_url = "https://raw.githubusercontent.com/benjihillard/English-Dictionary-Database/main/english%20Dictionary.csv"
    csv_filename = "english_dictionary.csv"
    
    if not os.path.exists(csv_filename):
        urllib.request.urlretrieve(csv_url, csv_filename)
    return csv_filename

def prepare_data(spark, csv_filename):
    """Load and prepare the data"""
    # Load the dictionary CSV
    df = spark.read.csv(
        csv_filename,
        header=True,
        inferSchema=True,
        escape='"'
    )
    
    # Clean and prepare the data
    df = df.filter(col("definition").isNotNull()) \
        .withColumn("definition_clean",
                    lower(regexp_replace(col("definition"), "[^a-zA-Z\\s]", "")))
    
    # Tokenize the definitions
    tokenizer = Tokenizer(inputCol="definition_clean", outputCol="words")
    df_tokenized = tokenizer.transform(df)
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = remover.transform(df_tokenized)
    
    # Apply TF-IDF
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    df_tf = hashingTF.transform(df_filtered)
    
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_model = idf.fit(df_tf)
    df_tfidf = idf_model.transform(df_tf)
    
    # Cache the result
    df_tfidf.cache()
    
    return df_tfidf, tokenizer, remover, hashingTF, idf_model

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors"""
    dot_product = float(vec1.dot(vec2))
    norm1 = float(np.sqrt(vec1.dot(vec1)))
    norm2 = float(np.sqrt(vec2.dot(vec2)))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def find_similar_words(query_definition, df_tfidf, tokenizer, remover, hashingTF, idf_model, spark, top_n=15):
    """Find similar words based on a query definition"""
    # Process the query definition
    query_df = spark.createDataFrame([(query_definition,)], ["definition"])
    query_df = query_df.withColumn("definition_clean",
                                     lower(regexp_replace(col("definition"), "[^a-zA-Z\\s]", "")))
    
    # Apply the same transformations
    query_tokenized = tokenizer.transform(query_df)
    query_filtered = remover.transform(query_tokenized)
    query_tf = hashingTF.transform(query_filtered)
    query_tfidf = idf_model.transform(query_tf)
    
    # Get the query vector
    query_vector = query_tfidf.select("features").first()[0]
    
    # Define UDF for similarity calculation
    similarity_udf = udf(lambda x: cosine_similarity(x, query_vector), FloatType())
    
    # Calculate similarities
    results = df_tfidf.withColumn("similarity", similarity_udf(col("features"))) \
        .select("word", "definition", "similarity") \
        .orderBy(col("similarity").desc()) \
        .limit(top_n)
    
    return results.collect()

# Main app logic
with st.spinner("🔄 Initializing Spark and loading data..."):
    if not st.session_state.initialized:
        spark = initialize_spark()
        csv_filename = download_dataset()
        df_tfidf, tokenizer, remover, hashingTF, idf_model = prepare_data(spark, csv_filename)
        
        st.session_state.initialized = True
        st.session_state.df_tfidf = df_tfidf
        st.session_state.spark = spark
        st.session_state.models = {
            'tokenizer': tokenizer,
            'remover': remover,
            'hashingTF': hashingTF,
            'idf_model': idf_model
        }
        st.success("✅ Initialization complete!")

# Sidebar for input
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_n = st.slider("Number of results", min_value=5, max_value=50, value=15, step=5)
    
    st.markdown("---")
    st.header("📝 Query Input")
    
    query_option = st.radio(
        "Select query type:",
        ["Custom Query", "Meeting Notes Example", "Simple Definition"],
        index=0
    )
    
    if query_option == "Meeting Notes Example":
        query = """
Hello worlf
        """
        st.text_area("Query Preview", query, height=200, disabled=True)
    elif query_option == "Simple Definition":
        query = "occurring or being done at the present moment; happening now; current event"
        st.text_area("Query Preview", query, height=100, disabled=True)
    else:
        query = st.text_area(
            "Enter your query definition", 
            height=200, 
            placeholder="Enter a definition or text to find similar words...\n\nExamples:\n- 'Something happening right now'\n- 'A person who teaches students'\n- 'The act of moving quickly on foot'"
        )
    
    # Submit button in sidebar
    submit_button = st.button("🔍 Find Similar Words", type="primary", use_container_width=True)

# Main content area
if submit_button:
    if query:
        with st.spinner("🔄 Processing query and finding similar words..."):
            results = find_similar_words(
                query, 
                st.session_state.df_tfidf,
                st.session_state.models['tokenizer'],
                st.session_state.models['remover'],
                st.session_state.models['hashingTF'],
                st.session_state.models['idf_model'],
                st.session_state.spark,
                top_n=top_n
            )
            
            # Convert to pandas DataFrame for easier manipulation
            df_results = pd.DataFrame([
                {
                    'Rank': idx + 1,
                    'Word': row['word'],
                    'Similarity': row['similarity'],
                    'Definition': row['definition']
                }
                for idx, row in enumerate(results)
            ])
            
            # Display results
            st.success(f"✅ Found {len(df_results)} similar words!")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "📋 Data Table"])
            
            with tab1:
                st.subheader("Similarity Scores - Bar Chart")
                fig_bar = px.bar(
                    df_results, 
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
                    df_results, 
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
                # Style the dataframe
                styled_df = df_results.style.background_gradient(
                    subset=['Similarity'], 
                    cmap='YlGnBu'
                ).format({'Similarity': '{:.4f}'})
                
                st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Additional statistics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Top Word", df_results.iloc[0]['Word'])
            with col2:
                st.metric("Max Similarity", f"{df_results['Similarity'].max():.4f}")
            with col3:
                st.metric("Avg Similarity", f"{df_results['Similarity'].mean():.4f}")
            with col4:
                st.metric("Min Similarity", f"{df_results['Similarity'].min():.4f}")
            
            # Download option
            st.markdown("---")
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="similar_words_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("⚠️ Please enter a query definition!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Built with PySpark ML, Streamlit, and Plotly | TF-IDF Based Word Similarity</p>
    </div>
    """,
    unsafe_allow_html=True
)
