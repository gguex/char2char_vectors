from local_functions import *
import pickle

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
# Regularization parameter (0.1)
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

# Get the main characters
main_char_idx = np.where((corpus.occurrences >= min_occurrences).to_numpy().sum(axis=0) > 10)
main_char_names = np.array(corpus.occurrences.columns)[main_char_idx]
main_char_occ = (corpus.occurrences >= min_occurrences).to_numpy().sum(axis=0)[main_char_idx]

# Build interactions and add them to data
interaction_occurrences = build_interactions(corpus.occurrences, max_interaction_degree)
corpus.occurrences = pd.concat([corpus.occurrences, interaction_occurrences], axis=1)

# Make occurrences binary
corpus.occurrences = 1*(corpus.occurrences >= min_occurrences)

# Make the occurrences across a meta
corpus.update_occurrences_across_meta(meta_for_occurrences)

# Make sure no units are empty
corpus.remove_units_without_words()
#corpus.remove_units_without_occurrences()
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
regression_vs_words = words_vs_regressions.transpose()

# ---- Examine relations with axes

regressions_vs_axes = pd.DataFrame(regression_coord, index=["intercept"] + list(corpus.occurrences.columns))
occurrences_vs_axes = pd.DataFrame(occurrence_coord, index=list(corpus.occurrences.columns))
axes_vs_regressions = regressions_vs_axes.transpose()
axes_vs_regressions = axes_vs_regressions.reindex(sorted(axes_vs_regressions.columns), axis=1)
axes_vs_occurrences = occurrences_vs_axes.transpose()
axes_vs_occurrences = axes_vs_occurrences.reindex(sorted(axes_vs_occurrences.columns), axis=1)

# ---- Explore the desired relationships

# Objects to explore
object_names = ["T1", "T2", "T3", "T4", "T5", "Cosette", "Cosette-Marius", "Cosette-Valjean", "Marius", "Valjean",
                "Marius-Valjean", "Javert", "Javert-Valjean", "Myriel", "Myriel-Valjean"]
if meta_for_occurrences is not None:
    separation_name = list(set(corpus.meta_variables[meta_for_occurrences]))
    for i in range(len(separation_name)):
        separation_name.extend([f"{obj}_{i+1}" for obj in object_names])
    object_names.extend(separation_name)

# The subset of object
present_object_names = []
for obj in object_names:
    if obj in words_vs_regressions.columns:
        present_object_names.append(obj)

A_occurrence = words_vs_occurrences[present_object_names]
A_regression = words_vs_regressions[present_object_names]

# -------------------------------
#  Network
# -------------------------------

studied_word = "justice"
polarity = regression_vs_words[studied_word]
min_occ = 20

char_1 = main_char_names[0]

interact_list = []
interaction_polarity = []
edge_weights = []
for char_1 in main_char_names:
    char_1_found = np.array([char_1 in ind_name for ind_name in polarity.index])
    for char_2 in main_char_names:
        if char_2 != char_1:
            char_2_found = np.array([char_2 in ind_name for ind_name in polarity.index])
            index_found = np.where(char_1_found & char_2_found)[0]
            if len(index_found) > 0:
                index_unique = index_found[0]
                interact_name = polarity.index[index_unique]
                interact_weight = corpus.occurrences[interact_name].sum()
                if interact_weight > min_occ:
                    interact_list.append([char_1, char_2])
                    interaction_polarity.append(polarity.iloc[index_unique])
                    edge_weights.append(interact_weight)

with open("corpora/LesMiserables_fr/node_pos.pkl", "rb") as pkl_file:
    node_pos = pickle.load(pkl_file)

display_char_network(interact_list, interaction_polarity, edge_weights, edge_min_width=0.1, edge_max_width=10,
                     node_pos=node_pos, node_min_width=10, node_max_width=200, font_size=15, plt_title=studied_word)

# with open("corpora/LesMiserables_fr/node_pos.pkl", "wb") as pkl_file:
#     pickle.dump(node_pos, pkl_file)

