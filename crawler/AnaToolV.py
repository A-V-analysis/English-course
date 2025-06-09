import pandas as pd
import matplotlib.pyplot as plt
import os
import jieba
import numpy as np

# 文件路径列表（中文标题）
csv_files = [
    "./result/data_discuss1.csv",
    "./result/data_discuss2.csv",
    "./result/data_news.csv",
    "./result/data_Frag1.csv",
    "./result/data_Frag2.csv",
    "./result/data_meme1.csv",
    "./result/data_meme2.csv"
]

# 自定义攻击性词典（可扩展）
aggressive_words = {
    "变态": 1.0,
    "虐杀": 1.0,
    "心理变态": 1.0,
    "虐猫": 1.0,
    "犯法": 0.8,
    "虐待": 0.9,
    "犯罪": 0.9,
    "杀人": 1.0,
    "杀": 0.7,
    "恶心": 0.8,
    "可怕": 0.7,
    "人渣": 1.0,
    "伤害": 0.7,
    "去死": 1.0,
    "该死": 0.9
}

# 加载自定义词典
for word in aggressive_words:
    jieba.add_word(word)

# 存储结果
results = []
file_labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

for file in csv_files:
    try:
        # 读取数据
        df = pd.read_csv(file, encoding='utf-8-sig')
        
        # 列名清洗
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('　', '')
        
        # 自动识别评论列和点赞列
        comment_col = next((col for col in df.columns if '评论内容' in col), None)
        like_col = next((col for col in df.columns if '赞' in col), None)
        
        if not comment_col or not like_col:
            print(f"跳过文件（缺少必要列）: {file}")
            results.append(0)
            continue
        
        # 初始化统计量
        total_score = 0
        total_comments = len(df)
        
        # 分析每条评论
        for _, row in df.iterrows():
            content = str(row[comment_col])
            likes = int(row[like_col]) if pd.notna(row[like_col]) else 0
            
            # 分词处理
            words = jieba.lcut(content)
            
            # 计算攻击性得分
            score = 0
            for word in words:
                if word in aggressive_words:
                    score += aggressive_words[word] * (1 + np.log1p(likes))  # 点赞数加权
            
            total_score += score
        
        # 归一化（每100条评论的攻击性得分）
        normalized_score = (total_score / total_comments) * 100 if total_comments > 0 else 0
        results.append(normalized_score)
        
        print(f"分析完成：{file} | 总评论数：{total_comments} | 攻击性指数：{normalized_score:.2f}")
    
    except Exception as e:
        print(f"处理文件出错：{file} | 错误：{str(e)}")
        results.append(0)

# 可视化
plt.figure(figsize=(12, 6))
bars = plt.bar(file_labels, results, color=['#ff6b6b' if x > max(results)*0.7 else '#74b9ff' for x in results])

plt.title("Analysis of the aggressiveness", fontsize=16, pad=20) #视频评论区攻击性指数分析
plt.ylabel("Aggression index", fontsize=12) #攻击性指数（每100条评论）
plt.xlabel("Vedio Theme", fontsize=12)
plt.xticks(rotation=45, ha='right')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("中文评论区攻击性分析.png", dpi=300, bbox_inches='tight')
plt.show()