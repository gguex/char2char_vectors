import os
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import colorsys

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/Hamlet/Hamlet.tsv"

# -------------------------------
#  Code
# -------------------------------

# --- Preprocess dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Make the list of important characters
min_char_count = 20
sent_char_count = corpus_df.groupby(["char_from"]).size()
char_list = sent_char_count[sent_char_count > min_char_count].index

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
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
tfidf_matrix = vectorizer.fit_transform(corpus_sent_list)
corpus_voc = vectorizer.get_feature_names_out()
index_notnull = np.where(np.sum(tfidf_matrix, axis=1) > 0)[0]
tfidf_matrix = tfidf_matrix[index_notnull, :]

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
voc_weights = np.sum(tfidf_matrix, axis=0).reshape(-1, 1)

# Making vectors for sentences
sent_weights = np.sum(tfidf_matrix, axis=1).reshape(-1, 1)
sent_vectors = (tfidf_matrix / sent_weights) @ voc_vectors
sent_actscene_list = np.array(reduced_corpus_df["act"].astype(str) + "."
                              + reduced_corpus_df["scene"].astype(str)).reshape(-1, 1)[index_notnull, :]

# Making vectors for characters and interactions
interact_list, char_color_list, interact_color_list = [], [], []
char_vectors, char_weights, interact_vectors, interact_weights = np.empty((0, wv_dim)), np.empty((0, 1)), \
                                                                 np.empty((0, wv_dim)), np.empty((0, 1))
for i, char in enumerate(char_list):
    # creating character color
    char_color = np.array(colorsys.hsv_to_rgb(i * 1 / len(char_list), 1, 0.75))
    char_color_list.append(char_color)
    # Finding the index of char
    index_char = np.array(reduced_corpus_df["char_from"] == char).reshape(-1, 1)
    index_char = index_char[index_notnull]
    # Finding the weights of all sentences of char
    char_sent_weights = np.multiply(sent_weights, index_char)
    # Computing the total weight of the char
    char_weight = sum(char_sent_weights)
    # Computing the vector of char
    char_vec = (char_sent_weights / char_weight).T @ sent_vectors
    # Saving results
    char_vectors = np.append(char_vectors, char_vec, axis=0)
    char_weights = np.append(char_weights, char_weight, axis=0)

    for char_2 in char_list:
        # Finding the index of interaction, if it exists
        index_interact = \
            np.array((reduced_corpus_df["char_from"] == char) & (reduced_corpus_df["char_to"] == char_2)).reshape(-1, 1)
        index_interact = index_interact[index_notnull]
        if np.sum(index_interact) > 0:
            # adding color
            interact_color_list.append(char_color)
            # Finding the weights of all sentences of interaction
            interact_sent_weights = np.multiply(sent_weights, index_interact)
            # Computing total weight of interaction
            interact_weight = sum(interact_sent_weights)
            # Computing the vector of interaction
            interact_vec = (interact_sent_weights / interact_weight).T @ sent_vectors
            # Saving results
            interact_list.append(f"{char} -> {char_2}")
            interact_vectors = np.append(interact_vectors, interact_vec, axis=0)
            interact_weights = np.append(interact_weights, interact_weight, axis=0)

# define scene color
sent_color_list = [np.array(char_color_list)[np.where(char_list == char)[0]][0]
                   for i, char in enumerate(reduced_corpus_df["char_from"]) if i in index_notnull]

# ---- TSNE Projection

# Minimum TFIDF for words
min_tfidf = 3
selected_word_index = np.where(np.sum(tfidf_matrix, axis=0) > min_tfidf)[1]
selected_word = corpus_voc[selected_word_index]
selected_word_vectors = voc_vectors[selected_word_index, :]
selected_word_weights = voc_weights[selected_word_index]

# Make the whole points table
all_vectors = np.concatenate([selected_word_vectors, char_vectors, interact_vectors])

# Make the TSNE projection
proj2D_all_vectors = TSNE(perplexity=30).fit_transform(np.asarray(all_vectors))


# ---- Plot
w_pow = 0.5
w_ch_pow = 0.3
w_int_pow = 0.3
magn = 2

fig, ax = plt.subplots()
ax.scatter(proj2D_all_vectors[:, 0], proj2D_all_vectors[:, 1], alpha=0, color="white")

for i, txt in enumerate(selected_word):
    ax.annotate(txt, (proj2D_all_vectors[i, 0], proj2D_all_vectors[i, 1]),
                size=magn*np.power(selected_word_weights[i], w_pow))

for i, txt in enumerate(char_list):
    ax.annotate(txt, (proj2D_all_vectors[i + len(selected_word_weights), 0],
                      proj2D_all_vectors[i + len(selected_word_weights), 1]),
                size=magn*np.power(char_weights[i], w_ch_pow), color="blue")

for i, txt in enumerate(interact_list):
    ax.annotate(txt, (proj2D_all_vectors[i + len(selected_word_weights) + len(char_weights), 0],
                      proj2D_all_vectors[i + len(selected_word_weights) + len(char_weights), 1]),
                size=magn*np.power(interact_weights[i], w_int_pow), color="red")

ax.grid()
plt.show()


# ---- Only char and interactions Projection

# Make the whole points table
charint_vectors = np.concatenate([char_vectors, interact_vectors])

# Make the TSNE projection
proj2D_charint_vectors = TSNE(perplexity=30).fit_transform(np.asarray(charint_vectors))


# ---- Plot
w2_ch_pow = 0.3
w2_int_pow = 0.3
magn2 = 2

fig, ax = plt.subplots()
ax.scatter(proj2D_charint_vectors[:, 0], proj2D_charint_vectors[:, 1], alpha=0, color="white")

for i, txt in enumerate(char_list):
    ax.annotate(txt, (proj2D_charint_vectors[i, 0],
                      proj2D_charint_vectors[i, 1]),
                size=magn2*np.power(char_weights[i], w2_ch_pow), color=char_color_list[i])

for i, txt in enumerate(interact_list):
    ax.annotate(txt, (proj2D_charint_vectors[i + len(char_weights), 0],
                      proj2D_charint_vectors[i + len(char_weights), 1]),
                size=magn2*np.power(interact_weights[i], w2_int_pow), color=interact_color_list[i])

ax.grid()
plt.show()


# ---- Only sent, char and interactions Projection

# Make the whole points table
charsent_vectors = np.concatenate([sent_vectors, char_vectors, interact_vectors])

# Make the TSNE projection
proj2D_charsent_vectors = TSNE(perplexity=30).fit_transform(np.asarray(charsent_vectors))


# ---- Plot
w3_sent_pow = 0.5
w3_ch_pow = 0.3
w3_int_pow = 0.3
magn3 = 2

fig, ax = plt.subplots()
ax.scatter(proj2D_charsent_vectors[:, 0], proj2D_charsent_vectors[:, 1], alpha=0, color="white")

for i, txt in enumerate(sent_actscene_list):
    ax.annotate(txt, (proj2D_charsent_vectors[i, 0],
                      proj2D_charsent_vectors[i, 1]),
                size=magn3*np.power(sent_weights[i], w3_sent_pow), color=sent_color_list[i])

for i, txt in enumerate(char_list):
    ax.annotate(txt, (proj2D_charsent_vectors[i + len(sent_weights), 0],
                      proj2D_charsent_vectors[i + len(sent_weights), 1]),
                size=magn3*np.power(char_weights[i], w3_ch_pow), color=char_color_list[i])

for i, txt in enumerate(interact_list):
    ax.annotate(txt, (proj2D_charsent_vectors[i + len(sent_weights) + len(char_weights), 0],
                      proj2D_charsent_vectors[i + len(sent_weights) + len(char_weights), 1]),
                size=magn3*np.power(interact_weights[i], w3_int_pow), color=interact_color_list[i])

ax.grid()
plt.show()