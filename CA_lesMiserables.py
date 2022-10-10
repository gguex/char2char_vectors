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
min_word_frequency = 20
# Max interactions
max_interaction_degree = 2
# The minimum occurrences for an object to be considered
min_occurrences = 2
# Use a meta variable to build occurrences (None for original)
meta_for_occurrences = None
# Regularization parameter
regularization_parameter = 0.01

# -------------------------------
#  Loading
# -------------------------------

# Load stopwords
with open("aux_files/frenchST.txt") as stopwords_file:
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
corpus.remove_occurrences_with_frequency(1e-10)
corpus.remove_words_with_frequency(1e-10)

# Get units weights
f_row = corpus.n_tokens / corpus.n_tokens.sum()

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
occurrence_coord = build_occurrences_vectors(corpus.occurrences, row_coord, f_row)
# Compute the scalar product between occurrences_coord and word_coord
words_vs_occurrences = pd.DataFrame(col_coord @ occurrence_coord.T, columns=list(corpus.occurrences.columns))
words_vs_occurrences.index = corpus.units_words.columns
# Reorder by occurrences name
words_vs_occurrences = words_vs_occurrences.reindex(sorted(words_vs_occurrences.columns), axis=1)
occurrences_vs_words = words_vs_occurrences.transpose()

# ---- Make the regression

# Build regression vectors
regression_coord = build_regression_vectors(corpus.occurrences, row_coord, f_row,
                                            regularization_parameter=regularization_parameter)
# Compute the scalar product between regression_coord and word_coord
words_vs_regressions = pd.DataFrame(col_coord @ regression_coord.T, index=corpus.units_words.columns,
                                    columns=["intercept"] + list(corpus.occurrences.columns))
# Reorder by name
words_vs_regressions = words_vs_regressions.reindex(sorted(words_vs_regressions.columns), axis=1)
regressions_vs_words = words_vs_regressions.transpose()

# -------------------------------
#  RESULTS
# -------------------------------

# ---- Limited char and words

# Objects to explore
object_names = ["Cosette", "Cosette-Marius", "Cosette-Valjean", "Marius", "Valjean",
                "Marius-Valjean", "Javert", "Javert-Valjean", "Myriel", "Myriel-Valjean"]
word_names = ["aimer", "rue", "justice", "guerre"]

# The subset of object
present_object_names = []
for obj in object_names:
    if obj in words_vs_regressions.columns:
        present_object_names.append(obj)

lim_words_vs_occurrences = words_vs_occurrences[present_object_names]
lim_words_vs_regressions = words_vs_regressions[present_object_names]

form_lim_words_vs_occurrences = pd.DataFrame(index=range(10))
for object_name in present_object_names:
    object_largest = lim_words_vs_occurrences[object_name].nlargest()
    object_smallest = lim_words_vs_occurrences[object_name].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({object_name: object_col})
    form_lim_words_vs_occurrences = pd.concat([form_lim_words_vs_occurrences, object_df], axis=1)

form_lim_words_vs_regressions = pd.DataFrame(index=range(10))
for object_name in present_object_names:
    object_largest = lim_words_vs_regressions[object_name].nlargest()
    object_smallest = lim_words_vs_regressions[object_name].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({object_name: object_col})
    form_lim_words_vs_regressions = pd.concat([form_lim_words_vs_regressions, object_df], axis=1)

form_lim_occurrences_vs_words = pd.DataFrame(index=range(10))
for to_explore_word in word_names:
    object_largest = occurrences_vs_words[to_explore_word].nlargest()
    object_smallest = occurrences_vs_words[to_explore_word].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({to_explore_word: object_col})
    form_lim_occurrences_vs_words = pd.concat([form_lim_occurrences_vs_words, object_df], axis=1)

form_lim_regressions_vs_words = pd.DataFrame(index=range(10))
for to_explore_word in word_names:
    object_largest = regressions_vs_words[to_explore_word].nlargest()
    object_smallest = regressions_vs_words[to_explore_word].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({to_explore_word: object_col})
    form_lim_regressions_vs_words = pd.concat([form_lim_regressions_vs_words, object_df], axis=1)

# ---- Save results

# Words vs Objects - centroids
words_vs_occurrences.to_csv("results/CA_CENT_words_vs_objects.csv")
# Words vs Objects - regression
words_vs_regressions.to_csv("results/CA_REG_words_vs_objects.csv")
# Objects vs Words - centroids
occurrences_vs_words.to_csv("results/CA_CENT_objects_vs_words.csv")
# Objects vs Words - regression
regressions_vs_words.to_csv("results/CA_REG_objects_vs_words.csv")

# -- For article

form_lim_words_vs_occurrences.iloc[:, :5].to_csv("results/for_article/CA_CENT_words_vs_objects_LIMITED_1.csv",
                                                 index=False)
form_lim_words_vs_occurrences.iloc[:, 5:].to_csv("results/for_article/CA_CENT_words_vs_objects_LIMITED_2.csv",
                                                 index=False)
form_lim_words_vs_regressions.iloc[:, :5].to_csv("results/for_article/CA_REG_words_vs_objects_LIMITED_1.csv",
                                                 index=False)
form_lim_words_vs_regressions.iloc[:, 5:].to_csv("results/for_article/CA_REG_words_vs_objects_LIMITED_2.csv",
                                                 index=False)
