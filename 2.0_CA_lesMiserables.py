import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from local_functions import *

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/LesMiserables_fr/LesMiserables_tokens.tsv"

# Aliases file
aliases_path = "corpora/LesMiserables_fr/LesMiserables_aliases.txt"

# Set aggregation level (None for each line)
aggregation_level = "chapitre"

# Minimum occurrences for words
word_min_occurrences = 20

# The minimum occurrences for an object to be considered
min_occurrences = 3
# Max interactions
max_interaction_degree = 2

# Tome separation
tome_sep = True

# Objects to explore
object_names = ["Cosette", "Cosette-Marius", "Cosette-Valjean", "Marius", "Valjean", "Marius-Valjean", "Javert",
                "Javert-Valjean", "Myriel", "Myriel-Valjean"]
object_names_tome = ["1", "2", "3", "4", "5"]
for i in range(5):
    object_names_tome.extend([f"{obj}-{i+1}" for obj in object_names])
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

# Choice of stopwords
#used_stop_words = stopwords.words('french')
with open("aux_files/frenchST.txt") as stopwords_file:
    used_stop_words = stopwords_file.readlines()
used_stop_words = [process_text(stop_word) for stop_word in used_stop_words]

# Build the document-term matrix
vectorizer = CountVectorizer(stop_words=used_stop_words)
dt_matrix = vectorizer.fit_transform(texts)
vocabulary = vectorizer.get_feature_names_out()

# Make a threshold for the minimum vocabulary
index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= word_min_occurrences)[1]
dt_matrix = dt_matrix[:, index_voc_ok]
vocabulary = vocabulary[index_voc_ok]

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
#  Occurrences
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
        occurrence_names = occurrence_names + \
                           [f"{reg_name}-{dummy}" for id, reg_name in enumerate(character_names + interaction_names)
                            if id in non_zero_col]

# -------------------------------
#  Analysis
# -------------------------------

# ---- Make the CA

# Perform the CA
dim_max, percentage_var, row_coord, col_coord, row_contrib, col_contrib, row_cos2, col_cos2 = \
    correspondence_analysis(dt_matrix.todense())

# Explore the CA
row_explore_df, row_cos2_explore_df, col_explore_df, col_cos2_explore_df = \
    explore_correspondence_analysis(list(meta_variables[aggregation_level]), vocabulary, dim_max, row_coord, col_coord,
                                    row_contrib, col_contrib, row_cos2, col_cos2)

# ----------------------------------------
# --- Occurrences vectors
# ----------------------------------------

# --- Simple model of character + interactions

# Compute their coordinates
occurrence_coord = build_occurrences_vectors(occurrences, row_coord)
# Compute the scalar product between occurrences_coord and word_coord
words_vs_occurrences = pd.DataFrame(col_coord @ occurrence_coord.T, index=vocabulary, columns=occurrence_names)
# Reorder by occurrences name
words_vs_occurrences = words_vs_occurrences.reindex(sorted(words_vs_occurrences.columns), axis=1)

# ---- Make the regression

# Get units weights
f_row = np.array(dt_matrix.sum(axis=1)).reshape(-1)
f_row = f_row / sum(f_row)
# Build regression vectors
regression_coord = build_regression_vectors(occurrences, row_coord, f_row, regularization_parameter=10)
# Compute the scalar product between regression_coord and word_coord
words_vs_regressions = pd.DataFrame(col_coord @ regression_coord.T, index=vocabulary,
                                    columns=["intercept"] + occurrence_names)
# Reorder by name
words_vs_regressions = words_vs_regressions.reindex(sorted(words_vs_regressions.columns), axis=1)

# ---- Explore the desired relationships

# The subset of object
present_object_names = []
for obj in object_names:
    if obj in words_vs_regressions.columns:
        present_object_names.append(obj)

A_occurrence = words_vs_occurrences[present_object_names]
A_regression = words_vs_regressions[present_object_names]

occurrences_vs_words = words_vs_occurrences.transpose()
regression_vs_words = words_vs_regressions.transpose()