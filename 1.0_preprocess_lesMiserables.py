import os
import pandas as pd
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import re

# -------------------------------
#  Parameters
# -------------------------------

# Input folder path
input_corpus_path = "corpora/LesMiserables_fr/text"

# Ouput folder path
output_tsv_path = "corpora/LesMiserables_fr/LesMiserables.tsv"

# Keywords for division
division_keywords = ["Chapitre", "Livre"]

# Target character file
target_characters_path = "corpora/LesMiserables_fr/LesMiserables_characters.txt"

# Aliases file
aliases_path = "corpora/LesMiserables_fr/LesMiserables_aliases.txt"

# -------------------------------
#   Code
# -------------------------------

# Read characters
with open(target_characters_path) as target_character_file:
    target_characters = target_character_file.readlines()
target_characters = [target_character.strip() for target_character in target_characters]

# Read aliases
with open(aliases_path) as aliases_file:
    aliases = aliases_file.readlines()
aliases = {alias.split(",")[0].strip(): alias.split(",")[1].strip() for alias in aliases}

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
divisions_counters = np.repeat(0, len(division_keywords))
for tome_id, tome in enumerate(tomes):

    # Loading the corpus
    with open(f"{input_corpus_path}/{tome}") as corpus_file:
        lines = corpus_file.readlines()

    # To store values in loop
    characters_l = []
    paragraph = ""
    paragraphs = []
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
                paragraph_treated = re.sub("['_]", " ", paragraph)
                flair_sentence = Sentence(paragraph_treated, use_tokenizer=True)
                flair_tagger.predict(flair_sentence)
                potential_characters = [entity.text for entity in flair_sentence.get_spans("ner")]
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
