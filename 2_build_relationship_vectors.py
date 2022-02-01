import os
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Corpus tsv path
corpus_tsv_path = "corpora/Hamlet.tsv"

# --- Preprocess dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Make the list of important characters
sent_char_count = corpus_df.groupby(["char_from"]).size()
char_list = sent_char_count[sent_char_count > 20].index

# Reduce the corpus on the list of chars
reduced_corpus_df = corpus_df[corpus_df["char_from"].isin(char_list)]

# Preprocess sentences function
def process_sentence(sent):
    # Punctuation list
    punct_list = string.punctuation + "”’—“–"
    # Lower char
    sent_pp = sent.lower()
    # Remove numbers
    sent_pp = re.sub(r"[0-9]", " ", sent_pp)
    # Remove punctuation
    sent_pp = sent_pp.translate(str.maketrans(punct_list, " " * len(punct_list)))
    # Return the sentence
    return sent_pp


# Apply the function
corpus_sent_list = [process_sentence(sent) for sent in reduced_corpus_df["sentence"]]

# Build the vocabulary and tf-idf measure
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_sent_list)
corpus_voc = vectorizer.get_feature_names_out()

# --- Make vectors

# Loading wordvector models
home = os.path.expanduser("~")
wv_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")
wv_voc = wv_model.index_to_key
wv_dim = wv_model.vector_size

# Save common voc
common_voc = list(set(corpus_voc) & set(wv_voc))

# Reduce the tfidf_matrix
index_ok = [word in common_voc for word in corpus_voc]
tfidf_matrix = tfidf_matrix[:, index_ok]
corpus_voc = corpus_voc[index_ok]

# Making vectors for words
voc_vectors = np.zeros((len(corpus_voc), wv_dim))
for i, word in enumerate(corpus_voc):
    voc_vectors[i, :] = wv_model.get_vector(word)

# Making vectors for sentences
sent_vectors = (tfidf_matrix / np.sum(tfidf_matrix, axis=1)) @ voc_vectors
sent_weights = np.sum(tfidf_matrix, axis=1).reshape(-1, 1)

# Making vectors for characters and interactions
char_weight_list, interact_list, interact_weight_list = [], [], []
char_vectors, interact_vectors = np.empty((0, wv_dim)), np.empty((0, wv_dim))
for char in char_list:
    # Finding the index of char
    index_char = np.array(reduced_corpus_df["char_from"] == char).reshape(-1, 1)
    # Finding the weights of all sentences of char
    char_sent_weights = np.multiply(sent_weights, index_char)
    # Computing the total weight of the char
    char_weight = sum(char_sent_weights)
    # Computing the vector of char
    char_vec = (char_sent_weights / char_weight).T @ sent_vectors
    # Saving results
    char_vectors = np.append(char_vectors, char_vec, axis=0)
    char_weight_list.append(char_weight)

    for char_2 in char_list:
        # Finding the index of interaction, if it exists
        index_interact = \
            np.array((reduced_corpus_df["char_from"] == char) & (reduced_corpus_df["char_to"] == char_2)).reshape(-1, 1)
        if np.sum(index_interact) > 0:
            # Finding the weights of all sentences of interaction
            interact_sent_weights = np.multiply(sent_weights, index_interact)
            # Computing total weight of interaction
            interact_weight = sum(interact_sent_weights)
            # Computing the vector of interaction
            interact_vec = (interact_sent_weights / interact_weight).T @ sent_vectors
            # Saving results
            interact_list.append(f"{char} -> {char_2}")
            interact_vectors = np.append(interact_vectors, interact_vec, axis=0)
            interact_weight_list.append(interact_weight)

# ---- TSNE Projection

# Make the whole points table
all_vectors = np.concatenate([voc_vectors, sent_vectors, char_vectors, interact_vectors])