import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = [
    "./result/data_discuss1.csv",
    "./result/data_discuss2.csv",
    "./result/data_news.csv",
    "./result/data_Frag1.csv",
    "./result/data_Frag2.csv",
    "./result/data_meme1.csv",
    "./result/data_meme2.csv"
]

primary_keywords = ["猫爹", "猫跌", "耄耋", "帽跌"]
secondary_keywords = ["猫", "帽", "耄", "耋", "爹", "跌", "蝶", "爱猫"]

meme_scores = []
total_comments = []
file_labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
meme_counts = []  # 统计primary和secondary都命中的评论数

for file in csv_files:
    meme_score = 0
    comment_count = 0
    meme_match_count = 0
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.replace('　', '').str.replace(' ', '')

        comment_col = next((col for col in df.columns if '评论内容' in col), None)
        like_col = next((col for col in df.columns if '赞' in col), None)

        if not comment_col or not like_col:
            print(f"❌ Missing required columns in {file}. Available: {df.columns.tolist()}")
            meme_scores.append(0)
            total_comments.append(1)
            meme_counts.append(0)
            continue

        df[comment_col] = df[comment_col].astype(str)
        df[like_col] = pd.to_numeric(df[like_col], errors='coerce').fillna(0).astype(int)

        comment_count = len(df)
        total_comments.append(comment_count)

        secondary_match_count = 0
        primary_match_count = 0
        for _, row in df.iterrows():
            content = row[comment_col]
            likes = row[like_col]
            is_meme = False

            if any(kw in content for kw in primary_keywords):
                if likes == 0:
                    likes = 1
                meme_score += likes
                primary_match_count += 1
                is_meme = True
            elif any(kw in content for kw in secondary_keywords):
                meme_score += 1
                secondary_match_count += 1
                is_meme = True

            if is_meme:
                meme_match_count += 1

        print(f"✅ {file}: matched {primary_match_count} primary and {secondary_match_count} secondary comments, raw score: {meme_score}, total comments: {comment_count}")
        meme_scores.append(meme_score)
        meme_counts.append(meme_match_count)

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")
        meme_scores.append(0)
        total_comments.append(1)
        meme_counts.append(0)

normalized_scores = [score / comments if comments > 0 else 0 for score, comments in zip(meme_scores, total_comments)]

plt.figure(figsize=(10, 6))
plt.bar(file_labels, normalized_scores, color="skyblue")
plt.title("Normalized Meme Influence Score (Score per Comment)", fontsize=16)
plt.ylabel("Meme Score per Comment", fontsize=12)
plt.xlabel("CSV File", fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("normalized_meme_influence_analysis.png", dpi=300)
plt.show()

print("\n各文件有效评论总数与含梗评论（primary/secondary_keywords）数：")
for label, total, meme in zip(file_labels, total_comments, meme_counts):
    print(f"{label}: 有效评论总数 = {total}，含梗评论数 = {meme}")