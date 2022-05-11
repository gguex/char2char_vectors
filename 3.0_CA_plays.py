import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from local_functions import display_char_network
import pickle

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/Hamlet/Hamlet.tsv"
# File for node position
corpus_node_pos = "corpora/Hamlet.pkl"
# Global or per_act
global_view = True

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
min_words_per_char = 20
min_occ_per_word = 10
index_interact_ok = np.where(np.sum(dt_matrix, axis=1) >= min_words_per_char)[0]
dt_thr_matrix = dt_matrix[index_interact_ok, :]
index_voc_ok = np.where(np.sum(dt_thr_matrix, axis=0) > min_occ_per_word)[1]
dt_thr_matrix = dt_thr_matrix[:, index_voc_ok]
corpus_thr_voc = corpus_voc[index_voc_ok]
act_interact_thr_name_list = np.array(act_interact_name_list)[index_interact_ok]
act_interact_thr_list = np.array(act_interact_list)[index_interact_ok]

# ---- Make the CA

# dataframe for ca
act_interact_df = pd.DataFrame(dt_thr_matrix.todense(), columns=corpus_thr_voc.tolist(),
                               index=act_interact_thr_name_list.tolist())
# Creating matrix
cross_mat = np.array(act_interact_df)

# Number of row and col
n_row, n_col = cross_mat.shape
min_rowcol = min(n_row, n_col)

# Constructing independence table
n = np.sum(cross_mat)
f = cross_mat.sum(axis=1)
f = f / sum(f)
fs = cross_mat.sum(axis=0)
fs = fs / sum(fs)
indep_mat = np.outer(f, fs) * n
quot_mat = cross_mat / indep_mat

# Transformation of quotients
beta = 1
quot_trans = 1 / beta * (quot_mat ** beta - 1)

# Scalar products matrix, eigen decomposition
B = (quot_trans * fs) @ quot_trans.T
K = np.outer(np.sqrt(f), np.sqrt(f)) * B
val_p, vec_p = np.linalg.eig(K)
idx = val_p.argsort()[::-1]
val_p = np.abs(val_p[idx])[:min_rowcol]
vec_p = vec_p[:, idx][:, :min_rowcol]

# Row coordinates
coord_row = np.real(np.outer(1 / np.sqrt(f), np.sqrt(val_p)) * vec_p)
# Col coordinates
coord_col = (quot_trans.T * f) @ coord_row / np.sqrt(val_p)

# Row contrib ( 1 for sum(axis=0) )
contrib_row = np.outer(f, 1 / val_p) * coord_row ** 2
# Col contrib ( 1 for sum(axis=0) )
contrib_col = np.outer(fs, 1 / val_p) * coord_col ** 2

# Row cos2 ( 1 for sum(axis=1) )
cos2_row = coord_row ** 2
cos2_row = np.outer(1 / cos2_row.sum(axis=1), np.repeat(1, min_rowcol)) * cos2_row
# Col cos2 (1 for sum(axis=1) )
cos2_col = coord_col ** 2
cos2_col = np.outer(1 / cos2_col.sum(axis=1), np.repeat(1, min_rowcol)) * cos2_col

# ---- Display

edge_weights = np.sum(dt_thr_matrix, axis=1).T.tolist()[0]

with open(corpus_node_pos, "rb") as pkl_file:
    node_pos = pickle.load(pkl_file)

axis = 1
display_char_network(act_interact_thr_list, coord_row[:, axis], cos2_row[:, axis], node_pos=node_pos,
                     edge_min_width=0.5, edge_max_width=8, node_min_width=200, node_max_width=2000)
relation_df = pd.DataFrame({"Coord": coord_row[:, axis], "Contrib": contrib_row[:, axis], "Cos²": cos2_row[:, axis]},
                           index=act_interact_thr_name_list)
voc_df = pd.DataFrame({"Coord": coord_col[:, axis], "Contrib": contrib_col[:, axis], "Cos²": cos2_col[:, axis]},
                      index=corpus_thr_voc.tolist())

voc_df.to_csv(f"results/Hamlet_voc_df_factor{axis + 1}.csv")