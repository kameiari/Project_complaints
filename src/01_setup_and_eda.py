#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Corpus Setup & Exploratory Data Analysis (CFPB Complaints)

# In[1]:


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

ROOT = Path("..").resolve()
df = pd.read_csv(ROOT / "results" / "complaints_clean.csv").dropna(subset=["clean_text", "tokens"])
df["raw_words"] = df["clean_text"].str.split().str.len()
df["clean_tokens"] = df["tokens"].str.split().str.len()
print(f"Valid cleaned complaints: {len(df):,}")
display(df[["raw_words", "clean_tokens"]].describe())


# In[2]:


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df["raw_words"].clip(upper=600), bins=40, ax=axes[0], color="#4C72B0")
axes[0].set_title("Raw Cleaned Word Count (Clipped at 600)")
sns.histplot(df["clean_tokens"].clip(upper=300), bins=40, ax=axes[1], color="#55A868")
axes[1].set_title("Lemmatized Token Count (Clipped at 300)")
plt.tight_layout()
plt.show()


# In[3]:


vec = CountVectorizer(ngram_range=(1, 2), max_features=20)
X = vec.fit_transform(df["tokens"])
ngrams = pd.Series(X.sum(axis=0).A1, index=vec.get_feature_names_out()).sort_values(ascending=False)
display(ngrams.to_frame("frequency"))

