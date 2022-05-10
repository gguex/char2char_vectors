import os
import pandas as pd
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm

# -------------------------------
# ---- PARAMETERS

# Input folder path
input_corpus_path = "corpora/LesMiserables_fr"

# Ouput folder path
output_tsv_path = "corpora/LesMiserables_fr/LesMiserables.tsv"

# Keywords for division
division_keywords = ["Chapitre", "Livre"]

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

# -------------------------------
# ---- CODE

# Get tomes
tomes = os.listdir(input_corpus_path)
tomes.sort()

# Create dataframe for results
target_characters.sort()
columns_name = ["tome", *[kwd.lower() for kwd in division_keywords], "text", *target_characters]
final_df = pd.DataFrame(columns=columns_name)

# Setting tagger and divisions keywords (finer - coarser)
flair_tagger = SequenceTagger.load("fr-ner")

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
            output_characters.append(max(target_characters_present))
        else:
            # Check if an alias key is in character
            aliases_present = [ref_alias for ref_alias in ref_aliases.keys() if ref_alias in tested_character]
            if len(aliases_present) > 0:
                output_characters.append(ref_aliases[max(aliases_present)])

    # Return result
    return output_characters


# Loop on tomes
for tome_id, tome in enumerate(tomes):

    # Loading the corpus
    with open(f"{input_corpus_path}/{tome}") as corpus_file:
        lines = corpus_file.readlines()

    # To store values in loop
    characters_l = []
    paragraph = ""
    paragraphs = []
    divisions_counters = np.repeat(0, len(division_keywords))
    # Loop on lines
    for line in tqdm(lines):

        # If the line is empty
        if line.strip() == "":
            # If there is nothing before
            if paragraph == "":
                continue
            # Else store the paragraph
            else:
                # Get characters with flair and references
                flair_sentence = Sentence(paragraph, use_tokenizer=True)
                flair_tagger.predict(flair_sentence)
                potential_characters = [entity.text for entity in flair_sentence.get_spans("ner") if
                                        entity.tag == "PER"]
                recognised_characters = recognise_character(potential_characters, target_characters, aliases)
                character_counter = [recognised_characters.count(target_character)
                                     for target_character in target_characters]
                # Add information into df
                row_df = pd.DataFrame([[tome_id + 1, *divisions_counters, paragraph, *character_counter]],
                                      columns=columns_name)
                final_df = pd.concat([final_df, row_df], ignore_index=True)
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

# Save dataframe
final_df.to_csv(output_tsv_path, sep="\t")

final_df.sum()