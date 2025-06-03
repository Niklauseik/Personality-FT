from datasets import load_dataset
import pandas as pd
import os

# 创建保存路径
os.makedirs("datasets/movie", exist_ok=True)

# === 1. 下载 IMDb 的 test split ===
imdb_test = load_dataset("stanfordnlp/imdb", split="test")
imdb_df = pd.DataFrame(imdb_test)
imdb_df.to_csv("datasets/movie/imdb_test.csv", index=False)
print("✅ IMDb test split 已保存为 datasets/movie/imdb_test.csv")

# === 2. 下载 wiki-movie-plots-with-summaries 的前 3000 条，仅保留两列 ===
wiki_subset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train[:3000]")
wiki_df = pd.DataFrame(wiki_subset)[["Plot", "PlotSummary"]]
wiki_df.to_csv("datasets/movie/wiki_movie_summ_3k.csv", index=False)
print("✅ Wiki Plot摘要前3000条已保存为 datasets/movie/wiki_movie_summ_3k.csv")
