import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from prince import CA

# Corpus tsv path
corpus_tsv_path = "corpora/Hamlet.tsv"

# --- Preprocess dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Make the list of important characters
sent_char_count = corpus_df.groupby(["char_from"]).size()
char_list = sent_char_count[sent_char_count > 20].index

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
count_act_interact_df = reduced_corpus_df.groupby(["act", "char_from", "char_to"]).count()
act_interact_list = list(count_act_interact_df.index)

# Make the corpus for act_interact
act_interact_corpus = []
act_interact_name_list = []
for act_interact in act_interact_list:
    # get the
    index_act_interact = np.where((reduced_corpus_df["act"] == act_interact[0]) &
                                  (reduced_corpus_df["char_from"] == act_interact[1]) &
                                  (reduced_corpus_df["char_to"] == act_interact[2]))[0]
    all_interact_sent_list = np.array(corpus_sent_list)[index_act_interact].tolist()
    act_interact_corpus.append(" ".join(all_interact_sent_list))
    act_interact_name_list.append(f"{act_interact[0]}_{act_interact[1]}_{act_interact[2]}")


# Build the document-term matrix
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
dt_matrix = vectorizer.fit_transform(act_interact_corpus)
corpus_voc = vectorizer.get_feature_names_out()

# Make a threshold for the minimum vocabulary
voc_threshold = 10
occ_threshold = 10
index_interact_ok = np.where(np.sum(dt_matrix, axis=1) >= voc_threshold)[0]
dt_thr_matrix = dt_matrix[index_interact_ok, :]
index_voc_ok = np.where(np.sum(dt_thr_matrix, axis=0) > occ_threshold)[1]
dt_thr_matrix = dt_thr_matrix[:, index_voc_ok]
corpus_thr_voc = corpus_voc[index_voc_ok]
act_interact_thr_name_list = np.array(act_interact_name_list)[index_interact_ok]

# ---- Make the CA

# dataframe for ca
act_interact_df = pd.DataFrame(dt_thr_matrix.todense(), columns=corpus_thr_voc.tolist(),
                               index=act_interact_thr_name_list.tolist())

# Perform Ca
ca = CA(n_components=2)
ca.fit(act_interact_df)

# Get the coordinates
interact_coord = ca.row_coordinates(act_interact_df)
word_coord = ca.column_coordinates(act_interact_df)

# ---- Plot it

ca.plot_coordinates(act_interact_df)
