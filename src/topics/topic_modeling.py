"""
Topic Modeling Exploration with BERTopic

This script explores topic distributions in tweets using BERTopic 
to understand underlying themes and patterns in the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from wordcloud import WordCloud
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def clean_text(text):
    """Minimal preprocessing - remove URLs and extra whitespace"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():
    print("="*80)
    print("TOPIC MODELING EXPLORATION WITH BERTOPIC")
    print("="*80)
    
    # 1. Load the training data
    print("\n1. Loading data...")
    data_path = os.path.join('data', 'train.csv')
    df = pd.read_csv(data_path)
    
    # Create label mapping
    label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
    df['sentiment'] = df['label'].map(label_map)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # 2. Minimal preprocessing
    print("\n2. Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    documents = df['cleaned_text'].tolist()
    print(f"Total documents: {len(documents)}")
    
    # 3. Train BERTopic model
    print("\n3. Training BERTopic model...")
    from umap import UMAP
    from hdbscan import HDBSCAN
    
    # Configure UMAP for less granular dimensionality reduction
    umap_model = UMAP(
        n_neighbors=30,      # Increased from default 15 (more neighbors = broader topics)
        n_components=5,      # Default is 5, keep it
        min_dist=0.1,        # Increased from 0.0 (allows more spread)
        metric='cosine'
    )
    
    # Configure HDBSCAN for less granular clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,     # Reduced to capture more documents (was 50)
        min_samples=5,           # Reduced for better coverage (was 10)
        metric='euclidean',
        cluster_selection_method='eom',  # Reverted to 'eom' (better results)
        prediction_data=True
    )
    
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=20         # Reverted to 20 for best balance
    )
    
    topics, probs = topic_model.fit_transform(documents)
    
    print(f"\nNumber of topics discovered: {len(set(topics)) - 1}")  # -1 to exclude outlier topic
    print(f"Number of outliers: {list(topics).count(-1)}")
    
    # 4. Get and save topic information
    print("\n4. Saving topic information...")
    topic_info = topic_model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info)
    
    os.makedirs('topics', exist_ok=True)
    topic_info_path = os.path.join('topics', 'topic_info.csv')
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Topic info saved to {topic_info_path}")
    
    # 5. Create output directory
    output_dir = os.path.join('outputs', 'figures', 'topics')
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Generate visualizations
    print("\n5. Generating visualizations...")
    
    # A. Topic distribution bar chart
    print("  - Topic bar chart...")
    fig = topic_model.visualize_barchart(top_n_topics=10, n_words=10, height=400)
    fig.write_html(os.path.join(output_dir, 'topic_barchart.html'))
    
    # B. Intertopic distance map
    print("  - Intertopic distance map...")
    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(output_dir, 'intertopic_distance.html'))
    
    # C. Topic hierarchy
    print("  - Topic hierarchy...")
    fig = topic_model.visualize_hierarchy()
    fig.write_html(os.path.join(output_dir, 'topic_hierarchy.html'))
    
    # D. Topic heatmap
    print("  - Topic similarity heatmap...")
    fig = topic_model.visualize_heatmap()
    fig.write_html(os.path.join(output_dir, 'topic_heatmap.html'))
    
    # E. Word clouds per topic
    print("  - Word clouds for top topics...")
    valid_topics = [t for t in set(topics) if t != -1]
    top_topics = sorted(valid_topics, key=lambda x: list(topics).count(x), reverse=True)[:5]
    
    for topic_id in top_topics:
        topic_words = topic_model.get_topic(topic_id)
        
        if topic_words:
            word_freq = {word: weight for word, weight in topic_words}
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_id}: {", ".join([w for w, _ in topic_words[:5]])}', 
                     fontsize=14, fontweight='bold')
            
            save_path = os.path.join(output_dir, f'wordcloud_topic_{topic_id}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  ✓ Word clouds saved!")
    
    # 7. Topic-Sentiment Analysis
    print("\n6. Analyzing topic-sentiment relationships...")
    df['topic'] = topics
    df['topic_prob'] = [probs[i][topics[i]] if topics[i] != -1 else 0 for i in range(len(topics))]
    
    # Create topic-sentiment heatmap
    topic_sentiment = df[df['topic'] != -1].groupby(['topic', 'sentiment']).size().reset_index(name='count')
    pivot_table = topic_sentiment.pivot(index='topic', columns='sentiment', values='count').fillna(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.title('Topic Distribution by Sentiment', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_sentiment_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Topic distribution across sentiments (normalized)
    pivot_normalized = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 6))
    pivot_normalized.plot(kind='bar', stacked=True, colormap='Set2', width=0.8, ax=plt.gca())
    plt.title('Topic Distribution by Sentiment (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Topic', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_sentiment_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Topic-sentiment visualizations saved!")
    
    # 8. Show representative documents
    print("\n7. Representative documents for top topics:")
    for topic_id in top_topics[:5]:
        print(f"\n{'='*80}")
        print(f"TOPIC {topic_id}")
        print(f"Keywords: {', '.join([w for w, _ in topic_model.get_topic(topic_id)[:10]])}")
        print(f"{'='*80}")
        
        topic_docs = df[df['topic'] == topic_id].nlargest(3, 'topic_prob')
        
        for idx, row in topic_docs.iterrows():
            print(f"\n[Sentiment: {row['sentiment']} | Probability: {row['topic_prob']:.3f}]")
            print(f"Text: {row['text']}")
    
    # 9. Save model and artifacts
    print("\n8. Saving model and artifacts...")
    model_path = os.path.join('topics', 'bertopic_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(topic_model, f)
    print(f"  ✓ Model saved to {model_path}")
    
    topic_dist_path = os.path.join('topics', 'topic_distributions.npy')
    np.save(topic_dist_path, probs)
    print(f"  ✓ Topic distributions saved to {topic_dist_path}")
    
    topics_path = os.path.join('topics', 'topics.npy')
    np.save(topics_path, np.array(topics))
    print(f"  ✓ Topics saved to {topics_path}")
    
    print("\n" + "="*80)
    print("✓ TOPIC MODELING COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"All models and artifacts saved to: topics/")

if __name__ == "__main__":
    main()
