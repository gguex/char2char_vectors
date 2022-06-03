import pandas as pd
import spacy
from local_functions import process_text
from tqdm import tqdm

# -------------------------------
#  Parameters
# -------------------------------

# Corpus tsv path
corpus_tsv_path = "corpora/Romeo&Juliet/Romeo&Juliet.tsv"
# Outfile
output_tsv_path = "corpora/Romeo&Juliet/Romeo&Juliet_tokens.tsv"
# Language ("fr" or "en")
language = "en"

# -------------------------------
#  Code
# -------------------------------

# Load the dataframe
corpus_df = pd.read_csv(corpus_tsv_path, sep="\t", index_col=0)

# Get the texts
texts = corpus_df["text"]

# Load the spacy model depending on language
if language == "fr":
    nlp = spacy.load("fr_core_news_lg")
else:
    nlp = spacy.load("en_core_web_lg")

# Process the text
processed_texts = []
for text in tqdm(texts):
    text_pp = nlp(text)
    lemmas_pp = " ".join([word.lemma_ for word in text_pp if word.lemma_.strip() != ""])
    processed_texts.append(process_text(lemmas_pp))

# Change the text to the processed text
corpus_df["text"] = processed_texts

# Remove empty lines
corpus_df = corpus_df[corpus_df["text"] != ""]

# Save the new corpus
corpus_df.to_csv(output_tsv_path, sep="\t")
