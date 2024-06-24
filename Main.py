import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('All The Universities of Pakistan.xls')

#Initializing Bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#Get bert embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

#Generate embeddings for the "Popular For" column
df['bert_embedding'] = df['Popular For'].apply(get_bert_embeddings)

#Apply PCA for dimensionality reduction dynamically
def apply_pca(embeddings, n_components=50):
    pca = PCA(n_components=min(n_components, embeddings.shape[0], embeddings.shape[1]))
    transformed_embeddings = pca.fit_transform(embeddings)
    return transformed_embeddings, pca

#Normalize the embeddings and apply PCA
bert_embeddings = np.vstack(df['bert_embedding'].values)
normalized_embeddings = bert_embeddings / np.linalg.norm(bert_embeddings, axis=1, keepdims=True)
pca_embeddings, pca = apply_pca(normalized_embeddings)

#Update the DataFrame with PCA embeddings
df['pca_embedding'] = list(pca_embeddings)

#Function to recommend universities
def recommend_universities(user_interest, tuition_fee_budget, preferred_location, num_recommendations=5):
    #Filter based on user criteria
    filtered_df = df[
        (df['tuition_fees'] <= tuition_fee_budget) &
        (df['Location'].str.contains(preferred_location, case=False, na=False))
    ]

    if filtered_df.empty:
        return pd.DataFrame(columns=['University', 'Popular For', 'World Rank', 'Location', 'tuition_fees', 'similarity_score', 'explanation'])

    #Get user interest
    user_embedding = get_bert_embeddings(user_interest)
    #Normalize the user embedding
    user_embedding = user_embedding / np.linalg.norm(user_embedding)  
    #Use the same PCA transformation
    user_embedding_pca = pca.transform(user_embedding)  

    #Calculating the cosine similarities
    filtered_pca_embeddings = np.vstack(filtered_df['pca_embedding'].values)
    similarities = cosine_similarity(user_embedding_pca, filtered_pca_embeddings).flatten()

    #Get the top N similar universities
    similar_indices = similarities.argsort()[::-1][:num_recommendations]

    #Retrieve recommendations
    recommendations = filtered_df.iloc[similar_indices].copy()
    recommendations['explanation'] = recommendations['Popular For'].apply(
        lambda x: f'Matches your interest in {x}'
    )

    return recommendations[['University', 'Popular For', 'World Rank', 'Location', 'tuition_fees', 'explanation']]


user_interest = input("Enter your field of interest: ")
tuition_fee_budget = float(input("Enter your tuition fee budget: "))
preferred_location = input("Enter your preferred location: ")
num_recommendations = int(input("Enter the number of recommendations you want: "))


recommendations = recommend_universities(user_interest, tuition_fee_budget, preferred_location, num_recommendations)

recommendations
