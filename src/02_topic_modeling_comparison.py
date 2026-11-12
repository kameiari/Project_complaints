#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Comparative Topic Modeling & Ground-Truth Alignment

# In[1]:


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

ROOT = Path("..").resolve()
RESULTS_DIR = ROOT / "results"

summary = pd.read_csv(RESULTS_DIR / "topic_comparison_summary.csv")
metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
display(metrics)
display(summary)


# In[2]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
nmf_df = summary[summary["model"] == "TF-IDF + NMF"].copy()
nmf_df["label"] = nmf_df.apply(lambda r: f"T{r['topic_id']}: {', '.join(r['top_words'].split(', ')[:3])}", axis=1)

bert_df = summary[summary["model"] == "BERTopic"].copy()
bert_df["label"] = bert_df.apply(lambda r: f"T{r['topic_id']}: {', '.join(r['top_words'].split(', ')[:3])}", axis=1)

sns.barplot(data=nmf_df, x="share_pct", y="label", ax=axes[0], color="#4C72B0")
axes[0].set_title(f"TF-IDF + NMF Topic Share (NPMI = {metrics.loc[0, 'npmi_coherence']:.4f})")
axes[0].set_xlabel("Share of Documents (%)")
axes[0].set_ylabel("")

sns.barplot(data=bert_df, x="share_pct", y="label", ax=axes[1], color="#55A868")
axes[1].set_title(f"BERTopic Top-10 Share (NPMI = {metrics.loc[1, 'npmi_coherence']:.4f} | 43.6% Outliers)")
axes[1].set_xlabel("Share of Documents (%)")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import normalized_mutual_info_score

df = pd.read_csv(RESULTS_DIR / "complaints_clean.csv").dropna(subset=["tokens", "clean_text"])
df = df.sample(n=25000, random_state=42).reset_index(drop=True)

extra_stopwords = ["not", "no", "never", "tell", "say", "receive", "send", "time", "information", "know", "want", "ask", "go", "get"]
vec = TfidfVectorizer(max_features=8000, min_df=10, max_df=0.85, ngram_range=(1, 2), stop_words=extra_stopwords)
X = vec.fit_transform(df["tokens"])
nmf = NMF(n_components=10, random_state=42, init="nndsvda", max_iter=400)
df["nmf_topic"] = nmf.fit_transform(X).argmax(axis=1)

top_prods = df["product"].value_counts().head(6).index
sub = df[df["product"].isin(top_prods)]
nmi = normalized_mutual_info_score(sub["product"], sub["nmf_topic"])

ct = pd.crosstab(sub["nmf_topic"], sub["product"], normalize="index") * 100
ct.index = [f"T{i}: {', '.join(nmf_df.loc[nmf_df['topic_id']==i, 'top_words'].values[0].split(', ')[:3])}" for i in ct.index]

plt.figure(figsize=(11, 5.5))
sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Row Share (%)"})
plt.title(f"NMF Topic Alignment with Ground-Truth CFPB Products (NMI = {nmi:.3f})")
plt.xlabel("CFPB Product Category")
plt.ylabel("Discovered NMF Topic")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

