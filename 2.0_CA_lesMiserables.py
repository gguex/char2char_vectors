import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from local_functions import *

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/LesMiserables_fr/LesMiserables_tokens.tsv"

# Set aggregation level (None for each line)
aggregation_level = "chapitre"

# Minimum occurrences for words
word_min_occurrences = 20

# The minimum occurrences for an object to be considered
min_occurrences = 5
# Max interactions
max_interaction_degree = 2

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

# --- Construct the document term matrix and remove

# Build the document-term matrix
vectorizer = CountVectorizer(stop_words=stopwords.words('french'))
dt_matrix = vectorizer.fit_transform(texts)
vocabulary = vectorizer.get_feature_names_out()

# Make a threshold for the minimum vocabulary
index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= word_min_occurrences)[1]
dt_matrix = dt_matrix[:, index_voc_ok]
vocabulary = vocabulary[index_voc_ok]

# Remove character names
not_a_character = [i for i, word in enumerate(vocabulary)
                   if word not in [process_text(character_name) for character_name in character_names]]
dt_matrix = dt_matrix[:, not_a_character]
vocabulary = vocabulary[not_a_character]

# Build interactions
interaction_occurrences = build_interactions(character_occurrences, max_interaction_degree)
interaction_names = list(interaction_occurrences.columns)

# -------------------------------
#  Analysis
# -------------------------------

# ---- Make the CA

dim_max, percentage_var, row_coord, col_coord, row_contrib, col_contrib, row_cos2, col_cos2 = \
    correspondence_analysis(dt_matrix.todense())

# ---- Make the occurrences

# Concat
occurrences = np.concatenate([character_occurrences, interaction_occurrences], axis=1)
# Names
occurrence_names = character_names + interaction_names

# Threshold
occurrences[occurrences < min_occurrences] = 0
object_remaining = np.where(occurrences.sum(axis=0) > 0)[0]
occurrences = occurrences[:, object_remaining]
occurrence_names = list(np.array(occurrence_names)[object_remaining])

# ---- Simple model of character + interactions

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
regression_coord = build_regression_vectors(occurrences, row_coord, f_row, regularization_parameter=1)
# Compute the scalar product between regression_coord and word_coord
words_vs_regressions = pd.DataFrame(col_coord @ regression_coord.T, index=vocabulary,
                                    columns=["intercept"] + occurrence_names)
# Reorder by name
words_vs_regressions = words_vs_regressions.reindex(sorted(words_vs_regressions.columns), axis=1)


# ---- Make weid means of characters and relationships

# Character coordinates
character_weights = character_occurrences.to_numpy() / sum(character_occurrences.to_numpy())
character_coord = character_weights.T @ coord_row

# Make relationships and weights
relationships = []
relationship_presences = []
for i in range(len(characters) - 1):
    for j in range(i + 1, len(characters)):
        relationships.append([characters[i], characters[j]])
        relationship_presences.append(list((character_occurrences[characters[i]] > 0) *
                                           (character_occurrences[characters[j]] > 0)))
relationship_presences = np.array(relationship_presences).T

# Reduce existing relationships
relationships = np.array(relationships)[relationship_presences.sum(axis=0) > relationship_threshold]
relationship_presences = relationship_presences[:, relationship_presences.sum(axis=0) > relationship_threshold]

# Compute coord
relationship_weights = relationship_presences / sum(relationship_presences)
relationship_coord = relationship_weights.T @ coord_row

# ---- Result dataframes

relationship_df = pd.DataFrame(relationship_coord, index=["-".join(relationship) for relationship in relationships])
columns_whl_n = [(len(f"{dim_max}") - len(f"{dim}")) * "0" + f"{dim}" for dim in range(dim_max)]
word_coord_df = pd.DataFrame(coord_col, index=vocabulary, columns=[whl_n + "_coord" for whl_n in columns_whl_n])
word_contrib_df = pd.DataFrame(contrib_col, index=vocabulary, columns=[whl_n + "_contrib" for whl_n in columns_whl_n])
word_cos_df = pd.DataFrame(cos2_col, index=vocabulary, columns=[whl_n + "_cos2" for whl_n in columns_whl_n])
word_df = pd.concat([word_coord_df, word_contrib_df, word_cos_df], axis=1)
word_df = word_df.reindex(sorted(word_df.columns), axis=1)

# ---- Plots

fig, ax = plt.subplots()
ax.scatter(relationship_coord[:, displayed_axes[0]], relationship_coord[:, displayed_axes[1]], alpha=0, color="white")

for i in range(relationships.shape[0]):
    ax.annotate("-".join(relationships[i]),
                (relationship_coord[i, displayed_axes[0]], relationship_coord[i, displayed_axes[1]]), size=10)

ax.grid()
plt.show()

axis = 0
display_char_network(relationships,
                     relationship_coord[:, axis], relationship_coord[:, axis],
                     edge_min_width=0.5, edge_max_width=8, node_min_width=200, node_max_width=2000)
