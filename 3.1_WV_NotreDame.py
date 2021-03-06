from local_functions import *
from gensim.models import KeyedVectors

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/NotreDame_fr/NotreDame.tsv"
# Aliases file
aliases_path = "corpora/NotreDame_fr/NotreDame_aliases.txt"
# Word vectors to use
word_vectors_path = "/home/gguex/Documents/data/pretrained_word_vectors/fr_fasttext.model"
# Set aggregation level (None for each line)
aggregation_level = "chapitre"
# Minimum occurrences for words
min_word_frequency = 5
# Max interactions
max_interaction_degree = 2
# The minimum occurrences for an object to be considered
min_occurrences = 1
# Use a meta variable to build occurrences (None for original)
meta_for_occurrences = None
# Regularization parameter
regularization_parameter = 1

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
corpus.remove_units_without_occurrences()
corpus.remove_occurrences_with_frequency(1e-10)
corpus.remove_words_with_frequency(1e-10)

# Get units weights
f_row = corpus.n_tokens / corpus.n_tokens.sum()

# -------------------------------
#  Analysis
# -------------------------------

# --- Word vector model

# Loading word vectors model
wv_model = KeyedVectors.load(word_vectors_path)

# Getting wv model data
wv_vocabulary = wv_model.index_to_key
wv_dim = wv_model.vector_size

# Save common voc
common_vocabulary = list(set(corpus.units_words.columns) & set(wv_vocabulary))
# Get the index of existing voc
existing_word_index = [word in common_vocabulary for word in corpus.units_words.columns]
# Reodrer common_vocabulary in the same order found in vocabulary
common_vocabulary = list(np.array(corpus.units_words.columns)[existing_word_index])
# Words to suppress
absent_vocabulary = [word for word in corpus.units_words.columns if word not in common_vocabulary]

# --- Build vectors for words and units

# Making vectors for words
word_coord = np.zeros((len(common_vocabulary), wv_dim))
for i, word in enumerate(common_vocabulary):
    word_coord[i, :] = wv_model.get_vector(word)

# Reduce dt matrix to the common voc
corpus.remove_words(absent_vocabulary)
# Remove empty units
corpus.remove_units_without_words()
corpus.remove_occurrences_with_frequency(1e-10)

# Compute the unit vectors
# choice 1
#unit_coord = build_occurrences_vectors(dt_matrix.T, word_coord)

# choice 2
weighting_param = 0.001
word_weights = corpus.units_words.to_numpy().sum(axis=0)
word_weights = word_weights / sum(word_weights)
smoothed_word_weights = weighting_param / (weighting_param + word_weights)
uw_probability = corpus.units_words.to_numpy() / corpus.units_words.to_numpy().sum(axis=1).reshape(-1, 1)
weighted_dt_matrix = uw_probability * smoothed_word_weights
unit_coord = uw_probability @ word_coord

# Remove first singular vector residuals
first_eigen_vec = scipy.sparse.linalg.svds(unit_coord.T, 1)[0]
first_eigen_residual = first_eigen_vec @ first_eigen_vec.T @ unit_coord.T
unit_coord = unit_coord - first_eigen_residual.T

# ----------------------------------------
# ---- Occurrences vectors
# ----------------------------------------

# --- Simple model of character + interactions

# Compute their coordinates
occurrence_coord = build_occurrences_vectors(corpus.occurrences, unit_coord)
# Compute the cosine sim between occurrences_coord and word_coord
words_vs_occurrences = pd.DataFrame(compute_cosine(word_coord, occurrence_coord), index=common_vocabulary,
                                    columns=corpus.occurrences.columns)
# Reorder by occurrences name
words_vs_occurrences = words_vs_occurrences.reindex(sorted(words_vs_occurrences.columns), axis=1)

# ---- Make the regression

# Build regression vectors
regression_coord = build_regression_vectors(corpus.occurrences, unit_coord, f_row,
                                            regularization_parameter=regularization_parameter)
# Compute the cosine between regression_coord and word_coord
norm_regression_coord = (regression_coord.T / np.sqrt(np.sum(regression_coord ** 2, axis=1))).T
words_vs_regressions = pd.DataFrame(compute_cosine(word_coord, regression_coord), index=common_vocabulary,
                                    columns=["intercept"] + list(corpus.occurrences.columns))
# Reorder by name
words_vs_regressions = words_vs_regressions.reindex(sorted(words_vs_regressions.columns), axis=1)

# ---- Explore the desired relationships

# Objects to explore
object_names = ["Esmeralda", "Claude Frollo-Esmeralda", "Esmeralda-Quasimodo", "Claude Frollo", "Quasimodo",
                "Gringoire", "Esmeralda-Gringoire", "Ch??teaupers-Emeralada", "Ch??teaupers-Claude Frollo",
                "Ch??teaupers-Quasimodo", "Claude Frollo-Quasimodo"]
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

to_explore_words = ["aimer", "rue", "justice", "guerre"]
word_vs_occ = pd.DataFrame(index=range(10))
for to_explore_word in to_explore_words:
    object_largest = occurrences_vs_words[to_explore_word].nlargest()
    object_smallest = occurrences_vs_words[to_explore_word].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({to_explore_word: object_col})
    word_vs_occ = pd.concat([word_vs_occ, object_df], axis=1)
word_vs_occ.to_csv("word_vs_occ_wv_ND.csv")