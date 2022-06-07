from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
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
min_occurrences = 3
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
corpus.build_units_words(TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=used_stop_words))

# Make a threshold for the minimum vocabulary and remove units without words
#corpus.remove_words_with_frequency(min_word_frequency)

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
tfidf_sums = corpus.units_words.sum(axis=1).to_numpy()

# -------------------------------
#  Analysis
# -------------------------------

# --- Make the NMF model
n_groups = 10

nmf_model = NMF(n_components=n_groups, init="nndsvd", max_iter=2000)
nmf_model.fit(corpus.units_words.to_numpy())

# Getting components for units and words
unit_prob = nmf_model.transform(corpus.units_words.to_numpy())
word_prob = nmf_model.components_

theme_vs_word = pd.DataFrame(word_prob.T, index=corpus.units_words.columns)
theme_vs_unit = pd.DataFrame(unit_prob)

# Computing the character theme
char_vs_theme = build_occurrences_vectors(corpus.occurrences, unit_prob)

# Make the regression vectors
unit_sizes = corpus.units_words.sum(axis=1).to_numpy()
reg_vs_theme = build_logistic_regression_vectors(corpus.occurrences, unit_prob, tfidf_sums*10)
reg_vs_theme = pd.DataFrame(reg_vs_theme, list(corpus.occurrences.columns))
reg_vs_theme_2 = build_logistic_regression_vectors(corpus.occurrences, unit_prob, corpus.n_tokens)
reg_vs_theme_2 = pd.DataFrame(reg_vs_theme, list(corpus.occurrences.columns))