import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import networkx as nx
from collections import Counter

# -------------------------------
# ---- PARAMETERS

# Input/output paths
input_corpus_path = "corpora/LesMiserables_fr/LesMiserables1_fr.txt"  # french
#input_corpus_path = "corpora/LesMiserables_mini_en.txt" # english
output_tsv_path = "corpora/LesMiserables_pp.txt"

# Target character list
target_characters = ["Valjean", "Cosette", "Fantine", "Marius", "Gavroche", "Javert", "Monsieur Thénardier",
                     "Madame Thénardier", "Babet", "Claquesous", "Montparnasse", "Gueulemer", "Brujon", "Bamatabois",
                     "Madame Victurnien", "Enjolras", "Combeferre", "Courfeyrac", "Prouvaire", "Feuilly", "Bahorel",
                     "Lesgle", "Joly", "Grantaire", "Favourite", "Dahlia", "Zéphine", "Tholomyès", "Listolier",
                     "Blachevelle", "Fauchelevent", "Mabeuf", "Toussaint", "Gillenormand", "Pontmercy", "Myriel",
                     "Baptistine", "Magloire", "Gervais", "Éponine", "Magnon", "Fameuil", "Azelma", "Champmathieu",
                     "Brevet", "Simplice", "Chenildieu", "Cochepaille", "Innocente", "Mademoiselle Gillenormand",
                     "Bougon"]
# Minimum signs for a character
minimum_sign = 3
# Minimum occurences for a character
minimum_occ = 5

# -------------------------------
# ---- CODE

# ---- Character, text and divisions extraction

# Loading the corpus
with open(input_corpus_path) as corpus_file:
    lines = corpus_file.readlines()

# Setting tagger and divisions keywords (finer - coarser)
# For french
flair_tagger = SequenceTagger.load("fr-ner")
division_keywords = ["Chapitre", "Livre"]
# For english
# flair_tagger = SequenceTagger.load("ner")
# division_keywords = ["CHAPTER ", "BOOK "]


# To store values in loop
tokens = []
characters = []
paragraph = ""
paragraphs = []
divisions_counters = np.repeat(0, len(division_keywords))
divisions_l = [[] for _ in division_keywords]
# Loop on lines
for line in tqdm(lines):

    # If the line is empty
    if line.strip() == "":
        # If there is nothing before
        if paragraph == "":
            continue
        # Else store the paragraph
        else:
            # Append divisions
            paragraphs.append(paragraph)
            for divisions_id, divisions in enumerate(divisions_l):
                divisions.append(divisions_counters[divisions_id])
            # Get flair information and save it
            flair_sentence = Sentence(paragraph, use_tokenizer=True)
            flair_tagger.predict(flair_sentence)
            tokens.append([token.text for token in flair_sentence.tokens])
            characters.append([entity.text for entity in flair_sentence.get_spans("ner") if entity.tag == "PER"])
            # Reset paragraph
            paragraph = ""
    else:
        # Division count
        divisions_presence = [division_keyword in line for division_keyword in division_keywords]
        # Update the counter for divisions
        divisions_counters += divisions_presence
        # Add the paragraph if there are no divisions counter
        if not any(divisions_presence):
            paragraph += " " + line

# ---- Characters processing

# Construct the full list of characters
all_characters = sum(characters, [])
# Construct the unique list of characters
unique_characters = set(all_characters)
# Remove characters with not enough signs
unique_characters = [unique_character for unique_character in unique_characters
                     if len(unique_character) >= minimum_sign]

# Create a graph of character
graph = nx.DiGraph()
# Add characters
graph.add_nodes_from(unique_characters)
links = []
# Add links
for i, unique_character in enumerate(unique_characters):
    links.extend([(unique_character, distination_character) for distination_character in unique_characters
                  if (distination_character != unique_character) and unique_character in distination_character])
graph.add_edges_from(links)
