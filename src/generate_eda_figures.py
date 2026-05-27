import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk

nltk.download('stopwords', quiet=True)

# Setup
sns.set_style('whitegrid')
OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv('data/train.csv')
label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
df['sentiment'] = df['label'].map(label_map)

# Cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return words

# 1. Word clouds per sentiment
for label, name in label_map.items():
    text = ' '.join(df[df['label'] == label]['text'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud - {name}', fontsize=16)
    fig.savefig(os.path.join(OUTPUT_DIR, f'wordcloud_{name.lower()}.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved wordcloud_{name.lower()}.png')

# Precompute cleaned words per class
class_words = {}
for label, name in label_map.items():
    all_words = []
    for text in df[df['label'] == label]['text']:
        all_words.extend(clean_text(text))
    class_words[name] = all_words

# 2. Top 20 words bar chart per class
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, words) in zip(axes, class_words.items()):
    top = Counter(words).most_common(20)
    w, c = zip(*top)
    ax.barh(range(len(w)), c, color=sns.color_palette('viridis', 3)[list(label_map.values()).index(name)])
    ax.set_yticks(range(len(w)))
    ax.set_yticklabels(w)
    ax.invert_yaxis()
    ax.set_title(f'Top 20 Words - {name}')
    ax.set_xlabel('Frequency')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'top_words_per_class.png'), dpi=100, bbox_inches='tight')
plt.close(fig)
print('Saved top_words_per_class.png')

# 3. Tweet length distribution
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('Set2', 3)
for (label, name), color in zip(label_map.items(), colors):
    lengths = df[df['label'] == label]['text'].str.len()
    sns.histplot(lengths, kde=True, label=name, color=color, alpha=0.4, ax=ax)
ax.set_xlabel('Character Count')
ax.set_ylabel('Frequency')
ax.set_title('Tweet Length Distribution by Sentiment')
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, 'tweet_length_distribution.png'), dpi=100, bbox_inches='tight')
plt.close(fig)
print('Saved tweet_length_distribution.png')

# 4. Top bigrams per class
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, words) in zip(axes, class_words.items()):
    bigrams = list(ngrams(words, 2))
    top = Counter(bigrams).most_common(15)
    labels = [' '.join(b) for b, _ in top]
    counts = [c for _, c in top]
    ax.barh(range(len(labels)), counts, color=sns.color_palette('viridis', 3)[list(label_map.values()).index(name)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f'Top 15 Bigrams - {name}')
    ax.set_xlabel('Frequency')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'top_bigrams_per_class.png'), dpi=100, bbox_inches='tight')
plt.close(fig)
print('Saved top_bigrams_per_class.png')

print('\nAll figures generated successfully!')
