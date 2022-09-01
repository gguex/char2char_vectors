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
min_occurrences = 1
# Use a meta variable to build occurrences (None for original)
meta_for_occurrences = None
# Regularization parameter (0.1)
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
#corpus.occurrences = 1*(corpus.occurrences >= min_occurrences)

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

occ_v_word = pd.DataFrame(index=range(10))
for object_name in present_object_names:
    object_largest = A_occurrence[object_name].nlargest()
    object_smallest = A_occurrence[object_name].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({object_name: object_col})
    occ_v_word = pd.concat([occ_v_word, object_df], axis=1)
occ_v_word.to_csv("occ_v_word.csv")

reg_v_word = pd.DataFrame(index=range(10))
for object_name in present_object_names:
    object_largest = A_regression[object_name].nlargest()
    object_smallest = A_regression[object_name].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({object_name: object_col})
    reg_v_word = pd.concat([reg_v_word, object_df], axis=1)
reg_v_word.to_csv("reg_v_word_long2.csv")

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
word_vs_occ.to_csv("word_vs_occ.csv")

to_explore_words = ["aimer", "rue", "justice", "guerre"]
word_vs_reg = pd.DataFrame(index=range(10))
for to_explore_word in to_explore_words:
    object_largest = regression_vs_words[to_explore_word].nlargest()
    object_smallest = regression_vs_words[to_explore_word].nsmallest()
    object_col = []
    for i in range(object_largest.shape[0]):
        object_col.append(f"{object_largest.index[i]} ({np.round(object_largest.iloc[i], 2)})")
    for i in range(object_smallest.shape[0]):
        object_col.append(f"{object_smallest.index[i]} ({np.round(object_smallest.iloc[i], 2)})")
    object_df = pd.DataFrame({to_explore_word: object_col})
    word_vs_reg = pd.concat([word_vs_reg, object_df], axis=1)
word_vs_reg.to_csv("word_vs_reg.csv")


# -------------------------------
#  Plots
# -------------------------------

# Selection of axes
axes = [0, 1]

# Filter top words
top_word_index = np.where(corpus.units_words.to_numpy().sum(axis=0) > 100)[0]
sel_col_coord = col_coord[top_word_index, :]
top_words = np.array(corpus.units_words.columns)[top_word_index]

# Compute all coord
all_coord = np.concatenate([row_coord, sel_col_coord])

# --- Plot 1

fig, ax = plt.subplots()

ax.scatter(all_coord[:, axes[0]], all_coord[:, axes[1]], alpha=0, color="white")

for i, txt in enumerate(list(corpus.units_words.index)):
    ax.annotate(txt, (row_coord[i, axes[0]], row_coord[i, axes[1]]), size=10, color="red")

for i, txt in enumerate(top_words):
    ax.annotate(txt, (sel_col_coord[i, axes[0]], sel_col_coord[i, axes[1]]), size=12, color="blue", alpha=0.8)

ax.grid()
plt.show()

# --- Plot 2

plotted_characters = ["Cosette", "Javert", "Marius"]
character_colors = ["green", "purple", "red"]
other_color = "grey"
shift = 0.005

plotted_character = plotted_characters[0]

fig, ax = plt.subplots()

ax.scatter(all_coord[:, axes[0]], all_coord[:, axes[1]], alpha=0, color="white")

for i, txt in enumerate(top_words):
    ax.annotate(txt, (sel_col_coord[i, 0], sel_col_coord[i, 1]), size=10, color="blue", alpha=0.2)

all_char_index = []
for id_char, plotted_character in enumerate(plotted_characters):
    character_indices = np.where(corpus.occurrences[plotted_character].to_numpy())[0]
    char_unit_names = corpus.units_words.index.to_numpy()[character_indices]
    character_coords = row_coord[character_indices, :]
    mean_coords = character_coords.mean(axis=0)
    all_char_index.extend(char_unit_names)
    for i, index in enumerate(char_unit_names):
        ax.annotate(index,
                    (character_coords[i, axes[0]] + shift*id_char,
                     character_coords[i, axes[1]] + shift*id_char),
                    size=12, color=character_colors[id_char], alpha=0.8)
    ax.annotate(plotted_characters[id_char], (mean_coords[axes[0]], mean_coords[axes[1]]), size=20,
                color=character_colors[id_char])

for i, txt in enumerate(list(corpus.units_words.index)):
    if txt not in set(all_char_index):
        ax.annotate(txt, (row_coord[i, axes[0]], row_coord[i, axes[1]]), size=8, color=other_color, alpha=0.2)

ax.grid()
plt.show()

# --- Plot 3

plotted_characters = ["Cosette", "Javert", "Cosette-Javert"]
character_colors = ["green", "red", "orange"]
other_color = "grey"

plotted_character = plotted_characters[0]

fig, ax = plt.subplots()

ax.scatter(all_coord[:, axes[0]], all_coord[:, axes[1]], alpha=0, color="white")

for i, txt in enumerate(top_words):
    ax.annotate(txt, (sel_col_coord[i, 0], sel_col_coord[i, 1]), size=10, color="blue", alpha=0.2)

all_char_index = []
interaction_indices = np.where(corpus.occurrences[plotted_characters[2]].to_numpy())[0]
for id_char, plotted_character in enumerate(plotted_characters):
    character_indices = np.where(corpus.occurrences[plotted_character].to_numpy())[0]
    if id_char < 2:
        character_indices = np.array([ind for ind in character_indices if ind not in interaction_indices])
    char_unit_names = corpus.units_words.index.to_numpy()[character_indices]
    character_coords = row_coord[character_indices, :]
    mean_coords = character_coords.mean(axis=0)
    all_char_index.extend(char_unit_names)
    for i, index in enumerate(char_unit_names):
        ax.annotate(index,
                    (character_coords[i, axes[0]],
                     character_coords[i, axes[1]]),
                    size=12, color=character_colors[id_char], alpha=0.8)
    ax.annotate(plotted_characters[id_char], (mean_coords[axes[0]], mean_coords[axes[1]]), size=16,
                color=character_colors[id_char])

for i, txt in enumerate(list(corpus.units_words.index)):
    if txt not in set(all_char_index):
        ax.annotate(txt, (row_coord[i, axes[0]], row_coord[i, axes[1]]), size=8, color=other_color, alpha=0.2)

ax.grid()
plt.show()


# --- Plot 4

plotted_characters = ["Cosette", "Javert", "Cosette-Javert"]
character_colors = ["green", "red", "orange"]
other_color = "grey"
scale = 50

plotted_character = plotted_characters[0]

fig, ax = plt.subplots()

ax.scatter(all_coord[:, axes[0]], all_coord[:, axes[1]], alpha=0, color="white")

for i, txt in enumerate(top_words):
    ax.annotate(txt, (sel_col_coord[i, 0], sel_col_coord[i, 1]), size=10, color="blue", alpha=0.2)

all_char_index = []
interaction_indices = np.where(corpus.occurrences[plotted_characters[2]].to_numpy())[0]
for id_char, plotted_character in enumerate(plotted_characters):
    character_indices = np.where(corpus.occurrences[plotted_character].to_numpy())[0]
    if id_char < 2:
        character_indices = np.array([ind for ind in character_indices if ind not in interaction_indices])
    char_unit_names = corpus.units_words.index.to_numpy()[character_indices]
    character_coords = row_coord[character_indices, :]
    mean_coords_idx = np.where(corpus.occurrences.columns.to_numpy() == plotted_character)[0][0]
    mean_coords = regression_coord[mean_coords_idx, :].reshape(-1) * scale
    all_char_index.extend(char_unit_names)
    for i, index in enumerate(char_unit_names):
        ax.annotate(index,
                    (character_coords[i, axes[0]],
                     character_coords[i, axes[1]]),
                    size=12, color=character_colors[id_char], alpha=0.8)
    ax.annotate(plotted_characters[id_char], (mean_coords[axes[0]], mean_coords[axes[1]]), size=16,
                color=character_colors[id_char])

for i, txt in enumerate(list(corpus.units_words.index)):
    if txt not in set(all_char_index):
        ax.annotate(txt, (row_coord[i, axes[0]], row_coord[i, axes[1]]), size=8, color=other_color, alpha=0.2)

ax.grid()
plt.show()

