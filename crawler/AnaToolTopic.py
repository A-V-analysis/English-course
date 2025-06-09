# LDA_Topic_Analysis_Visualization.py
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# Configure matplotlib to use a font that supports English characters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# File paths (hardcoded)
CSV_FILES = [
    "./result/data_discuss1.csv",
    "./result/data_discuss2.csv",
    "./result/data_news.csv",
    "./result/data_Frag1.csv",
    "./result/data_Frag2.csv",
    "./result/data_meme1.csv",
    "./result/data_meme2.csv"
]

# Custom stopwords (English explanations added)
STOPWORDS = {
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '在', '要',
    '我', '你', '他', '她', '它', '我们', '他们', '自己', '这', '那', '哪',
    '怎么', '什么', '为什么', '可以', '可能', '会', '不会', '没有', '有'
}

def load_texts(file_path):
    """Load CSV file and extract text content"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # Auto-detect text column
        for col in df.columns:
            if '评论' in col or '内容' in col or 'text' in col.lower():
                return [str(t) for t in df[col] if pd.notna(t)]
        # Fallback to first column
        return [str(t) for t in df.iloc[:, 0] if pd.notna(t)]
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def preprocess_text(texts):
    """Text preprocessing: tokenization + stopword removal"""
    processed = []
    for text in texts:
        words = jieba.lcut(text)
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 1 and not w.isdigit()]
        processed.append(' '.join(filtered))
    return processed

def analyze_topics(texts, n_topics=5, top_words=10):
    """Perform LDA analysis"""
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=5000)
    dtm = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        random_state=42,
        max_iter=20
    )
    lda.fit(dtm)
    
    # Extract topic keywords
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-top_words:][::-1]
        top_words_list = [feature_names[i] for i in top_indices]
        topics.append((f"Topic {idx+1}", top_words_list))
    
    # Calculate topic distribution diversity
    doc_topic = lda.transform(dtm)
    topic_dist = doc_topic.mean(axis=0)
    diversity = -np.sum(topic_dist * np.log(topic_dist + 1e-10)) / np.log(n_topics)
    
    return topics, topic_dist, diversity

def create_pie_chart(topic_dist, topics, filename):
    """Create pie chart for a single file's topic distribution"""
    plt.figure(figsize=(8, 6))
    
    # 只显示Topic编号，不显示主题内容
    labels = [f"{topic_name}" for topic_name, _ in topics]
    
    # Create pie chart
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(topic_dist)))
    wedges, texts, autotexts = plt.pie(
        topic_dist,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 9}
    )
    
    plt.axis('equal')
    plt.title(f"Topic Distribution\n{os.path.basename(filename)}", pad=20)
    
    # Save individual plot
    output_file = f"Topic_Pie_{os.path.splitext(os.path.basename(filename))[0]}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def create_diversity_bar_chart(results):
    """Create bar chart comparing diversity across all files"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    filenames = [os.path.splitext(os.path.basename(f))[0].replace('data_', '') 
                for f, _, _, _ in results]
    diversities = [d for _, _, _, d in results]
    
    # Create bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    bars = plt.bar(filenames, diversities, color=colors)
    
    # Customize chart
    plt.title('Topic Diversity Comparison Across Files', pad=20)
    plt.ylabel('Diversity Index (0-1)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.95, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # Save comparison plot
    plt.tight_layout()
    plt.savefig('Topic_Diversity_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: Topic_Diversity_Comparison.png")

def main():
    all_results = []
    all_file_topics = []  

    for file in CSV_FILES:
        print(f"\nAnalyzing: {file}")
        
        # 1. Load data
        raw_texts = load_texts(file)
        if not raw_texts:
            print(f"Skipping empty file: {file}")
            continue
        
        # 2. Preprocess
        processed_texts = preprocess_text(raw_texts)
        print(f"Processed texts: {len(processed_texts)}")
        
        # 3. LDA analysis
        topics, topic_dist, diversity = analyze_topics(processed_texts)
        all_results.append((file, topics, topic_dist, diversity))
        all_file_topics.append((os.path.basename(file), topics))  # 收集主题词
        
        # 4. Create individual pie chart
        create_pie_chart(topic_dist, topics, file)
        
        # 只输出必要信息
        print(f"Diversity: {diversity:.3f}")

    # 5. Create comparison chart if we have results
    if all_results:
        create_diversity_bar_chart(all_results)
    else:
        print("No valid data to analyze")

    # 6. 程序结束时集中输出所有文件主题词
    print("\n各文件主题词汇：")
    for fname, topics in all_file_topics:
        print(f"{fname}:")
        for topic_name, words in topics:
            print(f"  {topic_name}: {', '.join(words)}")

if __name__ == "__main__":
    # Add domain-specific words to dictionary
    jieba.add_word('虐猫', freq=2000)
    jieba.add_word('心理变态', freq=2000)
    
    main()
    input("Press Enter to exit...")