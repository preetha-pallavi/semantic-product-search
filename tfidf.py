import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

df_clean = pd.read_parquet('amazon_reviews_clean.parquet')

# combine all review text per product into one document
product_docs = df_clean.groupby('ProductId')['Text'].apply(lambda x: " ".join(x)).reset_index()
product_ids_list = product_docs['ProductId'].tolist()
documents = product_docs['Text'].tolist()

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 1),
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

def get_top_keywords(row_idx, n=4):
    row = tfidf_matrix[row_idx].toarray().flatten()
    top_indices = row.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_indices if row[i] > 0]

product_keywords = {}
for idx, pid in enumerate(product_ids_list):
    keywords = get_top_keywords(idx, n=4)
    product_keywords[pid] = ", ".join(keywords)

with open('product_keywords.pkl', 'wb') as f:
    pickle.dump(product_keywords, f)

print("Done —", len(product_keywords), "products")
# quick preview
for pid, kw in list(product_keywords.items())[:10]:
    print(pid, "->", kw)