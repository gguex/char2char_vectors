import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from nltk.corpus import stopwords
from local_functions import *


# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/LesMiserables_fr/LesMiserables.tsv"
# Set aggregation level (None for each line)
aggregation_level = "chapitre"
# Axes displayed
displayed_axes = (0, 1)
# Word threshold
word_threshold = 20
# Row information threshold
row_threshold = 5
# Relationship threshold
relationship_threshold = 10
# Character occurence threshold
character_occ_threshold = 3
# Max interactions
max_interaction_degree = 2

# -------------------------------
#  Code
# -------------------------------

# --- Preprocess dataframe

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)
# Get the columns name for separation and words
separation_columns = corpus_df.iloc[:, :(np.where(corpus_df.columns == "text")[0][0])].columns
word_columns = corpus_df.iloc[:, ((np.where(corpus_df.columns == "text")[0][0]) + 1):].columns

# Aggregate at the defined level and split the df
if aggregation_level is not None:
    separations = corpus_df.groupby(["tome", aggregation_level])[separation_columns].max()
    texts = list(corpus_df.groupby(["tome", aggregation_level])["text"].apply(lambda x: "\n".join(x)))
    character_occurrences = corpus_df.groupby(["tome", aggregation_level])[word_columns].sum()
else:
    separations = corpus_df[separation_columns]
    texts = list(corpus_df["text"])
    character_occurrences = corpus_df[word_columns]

# Get char list
characters = list(character_occurrences.columns)

# Process text function
def process_text(text):
    # Punctuation list
    enhanced_punctuation = string.punctuation + "”’—“–\n"
    # Lower char
    processed_text = text.lower()
    # Remove numbers
    processed_text = re.sub(r"[0-9]", " ", processed_text)
    # Remove punctuation
    processed_text = processed_text.translate(str.maketrans(enhanced_punctuation, " " * len(enhanced_punctuation)))
    # Return the sentence
    return processed_text


# Apply the function on texts
processed_texts = [process_text(text) for text in texts]

# Build the document-term matrix
vectorizer = CountVectorizer(stop_words=stopwords.words('french'))
dt_matrix = vectorizer.fit_transform(processed_texts)
vocabulary = vectorizer.get_feature_names_out()

# Make a threshold for the minimum vocabulary
index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= word_threshold)[1]
dt_matrix = dt_matrix[:, index_voc_ok]
vocabulary = vocabulary[index_voc_ok]

# Remove character name
not_a_character = [i for i, word in enumerate(vocabulary)
                   if word not in [process_text(character) for character in characters]]
dt_matrix = dt_matrix[:, not_a_character]
vocabulary = vocabulary[not_a_character]

# Remove row with not enough occurrences
kept_row_indices = np.where(dt_matrix.sum(axis=1) >= row_threshold)[0]
dt_matrix = dt_matrix[kept_row_indices, :]
character_occurrences = character_occurrences.iloc[kept_row_indices, :]
kept_col_indices = np.where(np.sum(dt_matrix, axis=0) >= word_threshold)[1]
dt_matrix = dt_matrix[:, kept_col_indices]
vocabulary = vocabulary[kept_col_indices]


# ---- Make the CA

dim_max, percentage_var, coord_row, coord_col, contrib_row, contrib_col, cos2_row, cos2_col = \
    correspondence_analysis(dt_matrix.todense())

# ---- Build interactions

# Get the presence (with a minimum of occurrences)
character_presences = (character_occurrences.to_numpy() > character_occ_threshold) * 1
# The reduced list of char
reduced_characters = np.array(characters)[character_presences.sum(axis=0) > 0]
character_presences = character_presences[:, character_presences.sum(axis=0) > 0]
# Build interactions
interaction_presences = build_interactions(pd.DataFrame(character_presences, columns=reduced_characters),
                                           max_interaction_degree)
interactions = list(interaction_presences.columns)

# ---- Make the regression

# Get sample weights
f_row = np.array(dt_matrix.sum(axis=1)).reshape(-1)
f_row = f_row / sum(f_row)

# Linear models
reg_coefs = []
for i in range(dim_max):
    predictors = np.concatenate([character_presences, interaction_presences.to_numpy()], axis=1)
    num_results = coord_row[:, i]
    lin_reg = linear_model.Ridge(0.1)
    lin_reg.fit(predictors, num_results, sample_weight=f_row)
    reg_coef = [lin_reg.intercept_]
    reg_coef.extend(lin_reg.coef_)
    reg_coefs.append(reg_coef)
reg_coefs = np.array(reg_coefs)

# Diplay the results
regression_elements = ["intercept"] + list(reduced_characters) + interactions
regression_df = pd.DataFrame(reg_coefs, columns=regression_elements)

# Compute the cosine between reg_coefs and coord_col
norm_coord_col = (coord_col.T / np.sqrt(np.sum(coord_col ** 2, axis=1))).T
norm_regression = regression_df.to_numpy().T
norm_regression = (norm_regression.T / np.sqrt(np.sum(norm_regression ** 2, axis=1))).T

# Make the cosine between regression df and words
wordsVSreg = pd.DataFrame(norm_coord_col @ norm_regression.T, index=vocabulary, columns=regression_elements)

# ---- Make weid means of characters and relationships

# Character coordinates
character_weights = character_occurrences.to_numpy() / sum(character_occurrences.to_numpy())
character_coord = character_weights.T @ coord_row

# Make relationships and weights
relationships = []
relationship_presences = []
for i in range(len(characters) - 1):
    for j in range(i+1, len(characters)):
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
word_df = pd.DataFrame(contrib_col, index=vocabulary)

# ---- Plot

fig, ax = plt.subplots()
ax.scatter(relationship_coord[:, displayed_axes[0]], relationship_coord[:, displayed_axes[1]], alpha=0, color="white")

for i in range(relationships.shape[0]):
    ax.annotate("-".join(relationships[i]),
                (relationship_coord[i, displayed_axes[0]], relationship_coord[i, displayed_axes[1]]), size=10)

ax.grid()
plt.show()

