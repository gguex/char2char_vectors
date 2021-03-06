import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from local_functions import *
import scipy
import matplotlib.pyplot as plt

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "../corpora/LesMiserables_fr/LesMiserables_tokens.tsv"

# Aliases file path
aliases_path = "../corpora/LesMiserables_fr/LesMiserables_aliases.txt"

# Set aggregation level (None for each line)
aggregation_level = "chapitre"

# Choice of weighting ("count" or "tfidf")
weighting_scheme = "tfidf"

# Minimum occurrences for words
word_min_occurrences = 20
word_min_tfidf = 0

# The minimum occurrences for an object to be considered
min_occurrences = 3

# Max interactions
max_interaction_degree = 2

# Tome separation
tome_sep = False

# Objects to explore
object_names = ["Cosette", "Cosette-Marius", "Cosette-Valjean", "Marius", "Valjean", "Marius-Valjean", "Javert",
                "Javert-Valjean", "Myriel", "Myriel-Valjean"]
object_names_tome = ["1", "2", "3", "4", "5"]
for i in range(5):
    object_names_tome.extend([f"{obj}-{i + 1}" for obj in object_names])
object_names.extend(object_names_tome)

# -------------------------------
#  Processing
# -------------------------------

# --- Process dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)
# Get the columns name for separation and words
meta_columns = corpus_df.iloc[:, :(np.where(corpus_df.columns == "text")[0][0])].columns
character_columns = corpus_df.iloc[:, ((np.where(corpus_df.columns == "text")[0][0]) + 1):].columns

# Aggregate at the defined level and split the df
if aggregation_level is not None:
    meta_variables = corpus_df.groupby([aggregation_level])[meta_columns].max()
    texts = list(corpus_df.groupby([aggregation_level])["text"].apply(lambda x: " ".join(x)))
    character_occurrences = corpus_df.groupby([aggregation_level])[character_columns].sum()
else:
    meta_variables = corpus_df[meta_columns]
    texts = list(corpus_df["text"])
    character_occurrences = corpus_df[character_columns]

# Get char list
character_names = list(character_occurrences.columns)

# Read aliases
with open(aliases_path) as aliases_file:
    aliases = aliases_file.readlines()
aliases = {alias.split(",")[0].strip(): alias.split(",")[1].strip() for alias in aliases}

# --- Construct the document term matrix and remove

# Check which weighting scheme, count
if weighting_scheme == "count":
    # Build the document-term matrix
    vectorizer = CountVectorizer(stop_words=stopwords.words("french"))
    dt_matrix = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()

    # Make a threshold for the minimum vocabulary
    index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= word_min_occurrences)[0]
    dt_matrix = dt_matrix[:, index_voc_ok]
    vocabulary = vocabulary[index_voc_ok]
# Or tfidf
else:
    # Build the document-term matrix
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("french"))
    dt_matrix = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()

    # Make a threshold for the tfidf
    dt_matrix[dt_matrix < word_min_tfidf] = 0
    index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= 0)[0]
    index_unit_ok = np.where(np.sum(dt_matrix, axis=1) >= 0)[0]
    dt_matrix = dt_matrix[:, index_voc_ok]
    dt_matrix = dt_matrix[index_unit_ok, :]
    vocabulary = vocabulary[index_voc_ok]
    meta_variables = meta_variables.iloc[index_unit_ok, :]
    texts = list(np.array(texts)[index_unit_ok])
    character_occurrences = character_occurrences.iloc[index_unit_ok, :]

# Remove character names
not_a_character = [i for i, word in enumerate(vocabulary)
                   if word not in [process_text(character_name)
                                   for character_name in character_names + list(aliases.keys())]]
dt_matrix = dt_matrix[:, not_a_character]
vocabulary = vocabulary[not_a_character]

# Build interactions
interaction_occurrences = build_interactions(character_occurrences, max_interaction_degree)
interaction_names = list(interaction_occurrences.columns)

# ----------------------------------------
# ---- OCCURRENCES
# ----------------------------------------

# ---- Make the occurrences

# Concat
occurrences = np.concatenate([character_occurrences, interaction_occurrences], axis=1)
# Names
occurrence_names = character_names + interaction_names

# ---- Modification of occurrences

# Threshold
# occurrences[occurrences < min_occurrences] = 0

# Binary with threshold
occurrences = occurrences >= min_occurrences

# Make sure no columns are null
object_remaining = np.where(occurrences.sum(axis=0) > 0)[0]
occurrences = occurrences[:, object_remaining]
occurrence_names = list(np.array(occurrence_names)[object_remaining])

# ---- Occurrences with time

if tome_sep:
    # Get dummies for tomes
    tome_dummies = pd.get_dummies(meta_variables, columns=["tome"])
    tome_dummies = tome_dummies[["tome_1", "tome_2", "tome_3", "tome_4", "tome_5"]]
    tome_dummies.columns = [1, 2, 3, 4, 5]

    # Build occurences and names
    pre_occurrences = np.concatenate([character_occurrences, interaction_occurrences], axis=1)
    occurrences = tome_dummies.to_numpy()
    occurrence_names = [f"{col}" for col in tome_dummies.columns]
    for dummy in tome_dummies.columns:
        new_elements = pre_occurrences * np.outer(tome_dummies[dummy].to_numpy(), np.ones(pre_occurrences.shape[1]))
        non_zero_col = np.where(np.sum(new_elements, axis=0) > 0)[0]
        occurrences = np.concatenate([occurrences, new_elements[:, non_zero_col]], axis=1)
        occurrence_names = occurrence_names + [f"{reg_name}-{dummy}"
                                               for id_reg, reg_name in enumerate(character_names + interaction_names)
                                               if id_reg in non_zero_col]

# -------------------------------
#  Analysis
# -------------------------------

# ------------1

# Units-character matrix
dc_matrix = character_occurrences.to_numpy()
unit_with_char = np.where(dc_matrix.sum(axis=1) > 0)[0]
char_with_unit = np.where(dc_matrix.sum(axis=0) > 0)[0]
dc_matrix = dc_matrix[unit_with_char, :]
dc_matrix = dc_matrix[:, char_with_unit]
dc_transition = dc_matrix / dc_matrix.sum(axis=1).reshape(-1, 1)
cd_transition = dc_matrix.T / dc_matrix.T.sum(axis=1).reshape(-1, 1)

# Char-char transition matrix
cc_transition = cd_transition @ dc_transition

# Compute the stationary distribution
cc_transition_eig_val, cc_transition_eig_vec = sorted_eig(cc_transition.T, 1)
cc_stationary = np.abs(cc_transition_eig_vec[:, 0])
cc_stationary = cc_stationary / sum(cc_stationary)

# Compute the power
pow = 1
cc_pow_transition = cc_transition
for _ in range(pow - 1):
    cc_pow_transition = cc_pow_transition @ cc_transition

cc_kernel = cc_pow_transition @ np.diag(1/cc_stationary) @ cc_pow_transition.T

# Max dim
dim_max = cc_kernel.shape[0] - 1

# Perform the eigen-decomposition of character
cc_eig_val, cc_eig_vec = scipy.sparse.linalg.eigs(cc_kernel, dim_max)
# Reorder eigen-vector and eigen-values, and cut to the maximum of dimensions
cc_idx = cc_eig_val.argsort()[::-1]
cc_eig_val = np.abs(cc_eig_val[cc_idx])[:dim_max]
cc_eig_vec = np.real(cc_eig_vec[:, cc_idx])[:, :dim_max]

# Coordinates
char_coord = np.sqrt(cc_eig_val) * np.array(cc_eig_vec)

# Plot
fig, ax = plt.subplots()
ax.scatter(char_coord[:, 0], char_coord[:, 1], alpha=0, color="white")

for i in range(char_coord.shape[0]):
    ax.annotate(np.array(character_names)[char_with_unit][i], (char_coord[i, 0], char_coord[i, 1]), size=10)

ax.grid()
plt.show()

# ------------2

# Create binary variable for character occurrences
dc_binary = 1*(character_occurrences >= min_occurrences).to_numpy()
unit_with_char = np.where(dc_binary.sum(axis=1) > 0)[0]
char_with_unit = np.where(dc_binary.sum(axis=0) > 0)[0]
dc_binary = dc_binary[unit_with_char, :]
dc_binary = dc_binary[:, char_with_unit]

# Documents-characters and characters-documents transition matrices
dc_transition = dc_binary / dc_binary.sum(axis=1).reshape(-1, 1)
cd_transition = dc_binary.T / dc_binary.T.sum(axis=1).reshape(-1, 1)

# Units-terms and terms-units transition matrices
dt_matrix = dt_matrix[unit_with_char, :]
term_remaining = np.where(dt_matrix.sum(axis=0) > 0)[0]
dt_matrix = dt_matrix[:, term_remaining]
dt_transition = dt_matrix / dt_matrix.sum(axis=1).reshape(-1, 1)
td_transition = dt_matrix.T / dt_matrix.T.sum(axis=1).reshape(-1, 1)

# Characters-terms and terms-characters transition matrices
ct_transition = cd_transition @ dt_transition
tc_transition = td_transition @ dc_transition

# Char-char and term-term transition matrix
cc_transition = ct_transition @ tc_transition
tt_transition = tc_transition @ ct_transition

# Compute the stationary distribution
cc_transition_eig_val, cc_transition_eig_vec = sorted_eig(cc_transition.T, 1)
cc_stationary = np.abs(cc_transition_eig_vec[:, 0])
cc_stationary = cc_stationary / sum(cc_stationary)
tt_transition_eig_val, tt_transition_eig_vec = sorted_eig(tt_transition.T, 1)
tt_stationary = np.abs(tt_transition_eig_vec[:, 0])
tt_stationary = tt_stationary / sum(tt_stationary)

# Compute the kernels
pow = 4
cc_pow_transition = cc_transition
tt_pow_transition = tt_transition
for _ in range(pow - 1):
    cc_pow_transition = cc_pow_transition @ cc_transition
    tt_pow_transition = tt_pow_transition @ tt_transition


cc_kernel = cc_pow_transition @ np.diag(1/cc_stationary) @ cc_pow_transition.T
tt_kernel = tt_pow_transition @ np.diag(1/tt_stationary) @ tt_pow_transition.T

# Maximum of dim
dim_max = min(cc_kernel.shape[0], tt_kernel.shape[0]) - 1

# Perform the eigen-decomposition of character
cc_eig_val, cc_eig_vec = scipy.sparse.linalg.eigs(cc_kernel, dim_max)
# Reorder eigen-vector and eigen-values, and cut to the maximum of dimensions
cc_idx = cc_eig_val.argsort()[::-1]
cc_eig_val = np.abs(cc_eig_val[cc_idx])[:dim_max]
cc_eig_vec = np.real(cc_eig_vec[:, cc_idx])[:, :dim_max]

# Perform the eigen-decomposition of terms
tt_eig_val, tt_eig_vec = scipy.sparse.linalg.eigs(tt_kernel, dim_max)
# Reorder eigen-vector and eigen-values, and cut to the maximum of dimensions
tt_idx = tt_eig_val.argsort()[::-1]
tt_eig_val = np.abs(tt_eig_val[tt_idx])[:dim_max]
tt_eig_vec = np.real(tt_eig_vec[:, tt_idx])[:, :dim_max]

# Coordinates
char_coord = np.sqrt(cc_eig_val) * np.array(cc_eig_vec)
term_coord = np.sqrt(tt_eig_val) * np.array(tt_eig_vec)

# Plot
fig, ax = plt.subplots()
ax.scatter(char_coord[:, 0], char_coord[:, 1], alpha=0, color="white")

for i in range(char_coord.shape[0]):
    ax.annotate(np.array(character_names)[char_with_unit][i], (char_coord[i, 0], char_coord[i, 1]), size=10)

ax.grid()
plt.show()

# Compute the scalar product between occurrences_coord and word_coord
words_vs_occurrences = pd.DataFrame(term_coord @ char_coord.T,
                                    index=np.array(vocabulary)[term_remaining],
                                    columns=np.array(character_names)[char_with_unit])
# Reorder by occurrences name
words_vs_occurrences = words_vs_occurrences.reindex(sorted(words_vs_occurrences.columns), axis=1)
