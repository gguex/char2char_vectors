import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from local_functions import display_char_network

# Corpus tsv path
corpus_tsv_path = "corpora/Hamlet.tsv"

# Global or per_act
global_view = True

# --- Preprocess dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Make the list of important characters
min_char_count = 10
sent_char_count = corpus_df.groupby(["char_from"]).size()
char_list = sent_char_count[sent_char_count > min_char_count].index

# Reduce the corpus on the list of chars
reduced_corpus_df = corpus_df[corpus_df["char_from"].isin(char_list) & corpus_df["char_to"].isin(char_list)]

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

# Get all interactions per act
if global_view:
    count_act_interact_df = reduced_corpus_df.groupby(["char_from", "char_to"]).count()
    act_interact_list = list(count_act_interact_df.index)
else:
    count_act_interact_df = reduced_corpus_df.groupby(["act", "char_from", "char_to"]).count()
    act_interact_list = list(count_act_interact_df.index)

# Make the corpus for act_interact
act_interact_corpus = []
act_interact_name_list = []
for act_interact in act_interact_list:
    # get the
    if global_view:
        index_act_interact = np.where((reduced_corpus_df["char_from"] == act_interact[0]) &
                                      (reduced_corpus_df["char_to"] == act_interact[1]))[0]
        act_interact_name_list.append(f"{act_interact[0]}_{act_interact[1]}")
    else:
        index_act_interact = np.where((reduced_corpus_df["act"] == act_interact[0]) &
                                      (reduced_corpus_df["char_from"] == act_interact[1]) &
                                      (reduced_corpus_df["char_to"] == act_interact[2]))[0]
        act_interact_name_list.append(f"{act_interact[0]}_{act_interact[1]}_{act_interact[2]}")
    all_interact_sent_list = np.array(corpus_sent_list)[index_act_interact].tolist()
    act_interact_corpus.append(" ".join(all_interact_sent_list))

# Build the document-term matrix
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
dt_matrix = vectorizer.fit_transform(act_interact_corpus)
corpus_voc = vectorizer.get_feature_names_out()

# Make a threshold for the minimum vocabulary
voc_threshold = 20
occ_threshold = 20
index_interact_ok = np.where(np.sum(dt_matrix, axis=1) >= voc_threshold)[0]
dt_thr_matrix = dt_matrix[index_interact_ok, :]
index_voc_ok = np.where(np.sum(dt_thr_matrix, axis=0) > occ_threshold)[1]
dt_thr_matrix = dt_thr_matrix[:, index_voc_ok]
corpus_thr_voc = corpus_voc[index_voc_ok]
act_interact_thr_name_list = np.array(act_interact_name_list)[index_interact_ok]
act_interact_thr_list = np.array(act_interact_list)[index_interact_ok]

# ---- Make the embedding

# Loading wordvector models
home = os.path.expanduser("~")
wv_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")
wv_voc = wv_model.index_to_key
wv_dim = wv_model.vector_size

# Save common voc
common_voc = list(set(corpus_thr_voc) & set(wv_voc))

# Reduce the dt_thr_matrix with common voc
index_ok = [word in common_voc for word in corpus_thr_voc]
dt_thr_matrix = dt_thr_matrix[:, index_ok]
corpus_thr_voc = corpus_thr_voc[index_ok]
index_interact_ok = np.where(np.sum(dt_thr_matrix, axis=1) >= 0)[0]
dt_thr_matrix = dt_thr_matrix[index_interact_ok, :]

# Making vectors for words
voc_vectors = np.zeros((len(corpus_thr_voc), wv_dim))
for i, word in enumerate(corpus_thr_voc):
    voc_vectors[i, :] = wv_model.get_vector(word)

# Making vectors for interaction
interact_vectors = (dt_thr_matrix.todense() / np.sum(dt_thr_matrix, axis=1)) @ voc_vectors

# --- PLotting interaction

# # Make the TSNE projection
# proj2D_all_vectors = TSNE(perplexity=30).fit_transform(np.asarray(interact_vectors))
#
# # ---- Plot
#
# fig, ax = plt.subplots()
# ax.scatter(proj2D_all_vectors[:, 0], proj2D_all_vectors[:, 1], alpha=0, color="white")
#
# for i, txt in enumerate(act_interact_thr_name_list):
#     ax.annotate(txt, (proj2D_all_vectors[i, 0],
#                       proj2D_all_vectors[i, 1]))
#
# ax.grid()
# plt.show()

# --- Getting similarity

direction_vector = wv_model.get_vector("love")
interaction_polarity_score = cosine_similarity(np.asarray(direction_vector.reshape(1, -1)),
                                               np.asarray(interact_vectors))[0]

interact_polarity_df = pd.DataFrame({"polarity": interaction_polarity_score}, index=act_interact_thr_name_list)

# second choice

# direction_vector_2 = wv_model.get_vector("hatred")
# interaction_polarity_score_2 = cosine_similarity(np.asarray((direction_vector - direction_vector_2).reshape(1, -1)),
#                                                  np.asarray(interact_vectors))[0]
#
# interact_polarity_df_2 = pd.DataFrame({"polarity": interaction_polarity_score_2}, index=act_interact_thr_name_list)

# --- Recreate interactions

edge_weights = np.sum(dt_thr_matrix, axis=1).T.tolist()[0]

display_char_network(act_interact_thr_list, interaction_polarity_score, edge_weights)