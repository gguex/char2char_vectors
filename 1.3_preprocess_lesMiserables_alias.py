import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
from collections import Counter

# -------------------------------
# ---- PARAMETERS

# Input/output paths
input_corpus_path = "corpora/LesMiserables1_fr.txt"  # french
# input_corpus_path = "corpora/LesMiserables_mini_en.txt" # english
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
# Alias dic
aliases = {"Jean-le-cric": "Valjean",
           "Madeleine": "Valjean",
           "Ultime": "Valjean",
           "Leblanc": "Valjean",
           "Euphrasie": "Cosette",
           "l'Alouette": "Cosette",
           "Montmercy": "Marius",
           "Petit Oiseau": "Gavroche",
           "Gamin de Paris": "Gavroche",
           "Inspecteur": "Javert",
           "Mr. Jondrette": "Monsieur Thénardier",
           "Thénardier": "Monsieur Thénardier",
           "Fabantou": "Monsieur Thénardier",
           "Thénard": "Monsieur Thénardier",
           "Madame Jondrette": "Madame Thénardier",
           "Jehan": "Prouvaire",
           "L'Aigle de Meaux": "Lesgle",
           "Laigle": "Lesgle",
           "Lèsgle": "Lesgle",
           "Bossuet": "Lesgle",
           "Jolllly": "Joly",
           "R": "Grantaire",
           "Fauvent": "Fauchelevent",
           "Zelma": "Azelma",
           "Bienvenu": "Myriel",
           "Monseigneur": "Myriel",
           "L'évêque": "Myriel"}
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

# Function for character treatment

def recognise_character(tested_characters, ref_characters, ref_aliases):
    # For the output
    output_characters = []
    # Loop on characters
    for tested_character in tested_characters:
        # Check if target_character is in character
        target_characters_present = [ref_character for ref_character in ref_characters
                                     if ref_character in tested_character]
        if len(target_characters_present) > 0:
            output_characters.extend(target_characters_present)
        else:
            # Check if an alias key is in character
            aliases_present = [ref_alias for ref_alias in ref_aliases.keys() if ref_alias in tested_character]
            if len(aliases_present) > 0:
                output_characters.append(ref_aliases[aliases_present[0]])

    # Return result
    return output_characters


# To store values in loop
tokens_l = []
characters_l = []
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
            tokens_l.append([token.text for token in flair_sentence.tokens])
            potential_characters = [entity.text for entity in flair_sentence.get_spans("ner") if entity.tag == "PER"]
            characters_l.append(recognise_character(potential_characters, target_characters, aliases))
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
all_characters = sum(characters_l, [])
# Construct the unique list of characters
unique_characters = set(all_characters)
# Remove characters with not enough signs
unique_characters = [unique_character for unique_character in unique_characters
                     if len(unique_character) >= minimum_sign]
