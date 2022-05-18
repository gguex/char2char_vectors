import re
import pandas as pd

# -------------------------------
#  Parameters
# -------------------------------

# Input/output paths
input_corpus_path = "corpora/Hamlet/Hamlet.txt"
output_tsv_path = "corpora/Hamlet/Hamlet.tsv"

# -------------------------------
#  Code
# -------------------------------

# Loading the corpus
with open(input_corpus_path) as corpus_file:
    corpus_lines = corpus_file.readlines()

# Detect character
act_presences = []
scene_presences = []
speaking_characters = []
speaking_lines = []
entrances = []
actions = []
for id_line, line in enumerate(corpus_lines):

    # Detect act or scene
    act_bool = ("ACT" in line)
    scene_bool = ("SCENE" in line)
    # Detect entrance
    entrance_detect = re.search("^Enter ", line)
    # Detect action
    action_detect = re.search("^\[\_", line)
    # Detect character
    char_detect = re.search("[A-Z `'’]{3,}\.", line)

    # Potential adding
    speaking_character = ""
    speaking_line = ""
    entrance = ""
    action = ""

    # Check if entrance
    if entrance_detect is not None:
        entrance = line[entrance_detect.span()[1]:]
    # Check if action
    elif action_detect is not None:
        action = line[action_detect.span()[1]:-3]
    elif (not act_bool) and (not scene_bool):
        if char_detect is not None:
            speaking_character = char_detect.group()[:-1]
            speaking_line = line[char_detect.span()[1]:]
        else:
            speaking_line = line

    # Save results
    act_presences.append(act_bool)
    scene_presences.append(scene_bool)
    speaking_characters.append(speaking_character)
    speaking_lines.append(speaking_line)
    entrances.append(entrance)
    actions.append(action)