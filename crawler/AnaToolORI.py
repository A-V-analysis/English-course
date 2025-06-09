import pandas as pd
import matplotlib.pyplot as plt
import os
import jieba

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

# 定义分级词典
primary_dict = ["反虐待", "mxz", "猫小子", "猫孝子", "猫学长", "猫猫", "爱猫人士", "反社会", "心理变态","入侵物种","流浪猫","生态"]
secondary_dict = ["心理", "法律", "社会", "虐待","生命","动物",]
tertiary_dict = ["猫", "帽"]

# 加载自定义词典
for word in primary_dict + secondary_dict + tertiary_dict:
    jieba.add_word(word)

# 存储结果
native_scores = []
file_labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

for file in csv_files:
    try:
        # 读取数据
        df = pd.read_csv(file, encoding='utf-8-sig')
        
        # 列名清洗
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('　', '')
        
        # 自动识别评论列
        comment_col = next((col for col in df.columns if '评论内容' in col), None)
        
        if not comment_col:
            print(f"跳过文件（缺少评论列）: {file}")
            native_scores.append(0)
            continue
        
        # 初始化统计量
        total_score = 0
        total_comments = len(df)
        
        # 分析每条评论
        for _, row in df.iterrows():
            content = str(row[comment_col])
            
            # 分词处理
            words = jieba.lcut(content)
            
            # 计算原生指数得分
            score = 0
            for word in words:
                if word in primary_dict:
                    score += 4
                elif word in secondary_dict:
                    score += 2
                elif word in tertiary_dict:
                    score += 1
            
            total_score += score
        
        # 计算平均原生指数
        avg_score = total_score / total_comments if total_comments > 0 else 0
        native_scores.append(avg_score)
        
        print(f"分析完成：{file} | 总评论数：{total_comments} | 平均原生指数：{avg_score:.2f}")
    
    except Exception as e:
        print(f"处理文件出错：{file} | 错误：{str(e)}")
        native_scores.append(0)

# 可视化
plt.figure(figsize=(12, 6))
bars = plt.bar(file_labels, native_scores, color=['#66c2a5' if x > max(native_scores)*0.7 else '#8da0cb' for x in native_scores])

plt.title("Analysis of the original context shift of the terrier", fontsize=16, pad=20)
plt.ylabel("Average native index", fontsize=12) #平均原生指数
plt.xlabel("Video Theme", fontsize=12) # 视频主题
plt.xticks(rotation=45, ha='right')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# 添加参考线
plt.axhline(y=sum(native_scores)/len(native_scores), color='r', linestyle='--', alpha=0.3)
plt.text(len(native_scores)-0.5, sum(native_scores)/len(native_scores)+0.1,
         f'average: {sum(native_scores)/len(native_scores):.2f}',
         color='r')

plt.tight_layout()
plt.savefig("耄耋梗原生语境偏移分析.png", dpi=300, bbox_inches='tight')
plt.show()