import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from local_functions import *

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/LesMiserables_fr/LesMiserables.tsv"
# Set aggregation level (None for each line)
aggregation_level = "chapitre"

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
    character_occurences = corpus_df.groupby(["tome", aggregation_level])[word_columns].sum()
else:
    separations = corpus_df[separation_columns]
    texts = list(corpus_df["text"])
    character_occurences = corpus_df[word_columns]


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
min_occurences_per_word = 10
index_voc_ok = np.where(np.sum(dt_matrix, axis=0) >= min_occurences_per_word)[1]
dt_matrix = dt_matrix[:, index_voc_ok]
vocabulary = vocabulary[index_voc_ok]

# ---- Make the CA

dim_max, percentage_var, coord_row, coord_col, contrib_row, contrib_col, cos2_row, cos2_col = \
    correspondence_analysis(dt_matrix.todense())

# ---- Display
