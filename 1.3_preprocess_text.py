import pandas as pd
import spacy
from local_functions import process_text

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/LesMiserables_fr/LesMiserables.tsv"
# Outfile
corpus_tsv_out_path = "corpora/LesMiserables_fr/LesMiserables_tokens.tsv"

# -------------------------------
#  Code
# -------------------------------

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Get the texts
texts = corpus_df["text"]

# Make
nlp = spacy.load("fr_core_news_lg")
processed_texts = []
for text in texts:
    text_pp = nlp(text)
    processed_texts.append(" ".join([word.lemma_ for word in text_pp if process_text(word.lemma_) != ""]))

