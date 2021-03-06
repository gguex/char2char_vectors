import numpy as np

from local_functions import *

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "../corpora/LesMiserables_fr/LesMiserables_tokens.tsv"
# Aliases file
aliases_path = "../corpora/LesMiserables_fr/LesMiserables_aliases.txt"
# Set aggregation level (None for each line)
aggregation_level = "chapitre"
# Minimum occurrences for words
min_word_frequency = 20
# Max interactions
max_interaction_degree = 2
# The minimum occurrences for an object to be considered
min_occurrences = 3
# Use a meta variable to build occurrences (None for original)
meta_for_occurrences = "tome"
# Regularization parameter
regularization_parameter = 1

# -------------------------------
#  Loading
# -------------------------------

# Load stopwords
with open("../aux_files/frenchST.txt") as stopwords_file:
    used_stop_words = stopwords_file.readlines()
used_stop_words = [process_text(stop_word) for stop_word in used_stop_words]

# Read aliases
with open(aliases_path) as aliases_file:
    aliases = aliases_file.readlines()
aliases = {alias.split(",")[0].strip(): alias.split(",")[1].strip() for alias in aliases}

# Load dataframe
corpus = CharacterCorpus()
corpus.load_corpus(corpus_tsv_path)

# Get character names
character_names = [process_text(character_name)
                   for character_name in list(corpus.occurrences.columns) + list(aliases.keys())]

# -------------------------------
#  Processing
# -------------------------------

# Aggregate on the level
corpus.aggregate_on(aggregation_level)

# Construct the unit-term matrix and remove rare words
corpus.build_units_words(CountVectorizer(stop_words=used_stop_words))

# Make a threshold for the minimum vocabulary and remove units without words
corpus.remove_words_with_frequency(min_word_frequency)

# Remove characters from words
corpus.remove_words(character_names)

# Build interactions and add them to data
interaction_occurrences = build_interactions(corpus.occurrences, max_interaction_degree)
corpus.occurrences = pd.concat([corpus.occurrences, interaction_occurrences], axis=1)

# Make occurrences binary
corpus.occurrences = 1*(corpus.occurrences >= min_occurrences)

# Make the occurrences across a meta
corpus.update_occurrences_across_meta(meta_for_occurrences)

# Make sure no units are empty
corpus.remove_units_without_words()
corpus.remove_units_without_occurrences()
corpus.remove_words_with_frequency(1e-10)

# -------------------------------
#  Analyses
# -------------------------------

# --- CA

# Perform the CA
dim_max, eig_val, row_coord, col_coord, row_contrib, col_contrib, row_cos2, col_cos2 = \
    correspondence_analysis(corpus.units_words.to_numpy())

# Explore the CA
row_explore_df, row_cos2_explore_df, col_explore_df, col_cos2_explore_df = \
    explore_correspondence_analysis(corpus.meta_variables.index, corpus.units_words.columns, dim_max, row_coord,
                                    col_coord, row_contrib, col_contrib, row_cos2, col_cos2)

# --- Make the occurrences frequency vectors

# Compute occurrence_coord
occurrence_coord = build_occurrences_vectors(corpus.occurrences, row_coord)
# Compute the scalar product between occurrences_coord and word_coord
words_vs_occurrences = pd.DataFrame(col_coord @ occurrence_coord.T, index=corpus.units_words.columns,
                                    columns=corpus.occurrences.columns)
# Reorder by occurrences name
words_vs_occurrences = words_vs_occurrences.reindex(sorted(words_vs_occurrences.columns), axis=1)

# ---- Make the regression

# Get units weights
f_row = (corpus.units_words.sum(axis=1) / corpus.units_words.sum().sum()).to_numpy()
# Build regression vectors
regression_coord = build_regression_vectors(corpus.occurrences, row_coord, f_row,
                                            regularization_parameter=regularization_parameter)
# Compute the scalar product between regression_coord and word_coord
words_vs_regressions = pd.DataFrame(col_coord @ regression_coord.T, index=corpus.units_words.columns,
                                    columns=["intercept"] + list(corpus.occurrences.columns))
# Reorder by name
words_vs_regressions = words_vs_regressions.reindex(sorted(words_vs_regressions.columns), axis=1)

# ---- Examine relations with axes

regressions_vs_axes = pd.DataFrame(regression_coord, index=["intercept"] + list(corpus.occurrences.columns))
occurrences_vs_axes = pd.DataFrame(occurrence_coord, index=list(corpus.occurrences.columns))
axes_vs_regressions = regressions_vs_axes.transpose()
axes_vs_regressions = axes_vs_regressions.reindex(sorted(axes_vs_regressions.columns), axis=1)
axes_vs_occurrences = occurrences_vs_axes.transpose()
axes_vs_occurrences = axes_vs_occurrences.reindex(sorted(axes_vs_occurrences.columns), axis=1)

# ---- Explore the desired relationships

# Objects to explore
object_names = ["Cosette", "Cosette-Marius", "Cosette-Valjean", "Marius", "Valjean", "Marius-Valjean", "Javert",
                "Javert-Valjean", "Myriel", "Myriel-Valjean"]
object_names_tome = ["1", "2", "3", "4", "5"]
for i in range(5):
    object_names_tome.extend([f"{obj}_{i+1}" for obj in object_names])
object_names.extend(object_names_tome)

# The subset of object
present_object_names = []
for obj in object_names:
    if obj in words_vs_regressions.columns:
        present_object_names.append(obj)

A_occurrence = words_vs_occurrences[present_object_names]
A_regression = words_vs_regressions[present_object_names]

occurrences_vs_words = words_vs_occurrences.transpose()
regression_vs_words = words_vs_regressions.transpose()

# ----------------------------------------
# --- Topic clustering
# ----------------------------------------

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

n_groups = 50

clust = AgglomerativeClustering(n_groups)
w_col_coord = col_coord * np.sqrt(eig_val)
clust_res = clust.fit(w_col_coord)

res = pd.DataFrame()
for i in range(n_groups):
    res = pd.concat([res, pd.Series(np.array(corpus.units_words.columns)[np.where(clust_res.labels_ == i)])],
                    axis=1, ignore_index=True)
