import re

import numpy as np
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
    entrance_detect = re.search("^(Enter|Re-enter) ", line)
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

# Get the characters
unique_character = [char.lower().strip() for char in set(speaking_characters) if char != ""]
unique_character = list(set(unique_character))
unique_character.remove("all")
unique_character.remove("first player")
unique_character.append("player")

# Treat entrances
list_of_entrances = []
for entrance in entrances:
    if "a King and a Queen" in entrance:
        char_detected = ["player king", "player queen"]
    else:
        char_detected = [char for char in unique_character if char in entrance.lower()]
    list_of_entrances.append(char_detected)

# Build df
act = 0
acts = []
scene = 0
scenes = []
speaking_char = ""
speaking_chars = []
sentence = ""
sentences = []
char_presence = []
char_presences = []
char_presence_table = []

def add_line():
    if sentence != "":
        acts.append(act)
        scenes.append(scene)
        speaking_chars.append(speaking_char)
        sentences.append(sentence)
        char_presences.append(char_presence)
        char_presence_line = np.zeros(len(unique_character))
        char_presence_line[np.where([char in char_presence for char in unique_character])[0]] = 1
        char_presence_line[np.where(np.array(unique_character) == speaking_char)[0]]
        char_presence_table.append(char_presence_line)

for i, line in enumerate(speaking_lines):
    # Update act or scene
    if act_presences[i]:
        act += 1
        if sentence != "":
            add_line()
            sentence = ""
    elif scene_presences[i]:
        scene += 1
        if sentence != "":
            add_line()
            sentence = ""
    # Update speaking char
    elif speaking_characters[i] != "":
        add_line()
        sentence = ""
        speaking_char = speaking_characters[i].lower().strip()

    # Update sentence
    sentence += speaking_lines[i]

    # Enter chararcters
    char_presence += list_of_entrances[i]

    # Outgoing char
    if actions[i] != "":
        if actions[i] == "Exit." or actions[i] == "Dies.":
            add_line()
            char_presence.remove(speaking_char)
            sentence = ""
            speaking_char = ""
        elif actions[i] == "Exeunt.":
            add_line()
            sentence = ""
            speaking_char = ""
            char_presence = []
        elif "Exeunt all but" in actions[i]:
            add_line()
            sentence = ""
            char_presence = [char for char in unique_character if char in actions[i].lower()]
        elif "Exit " in actions[i] or "Exeunt " in actions[i]:
            add_line()
            sentence = ""
            for char in unique_character:
                if char in actions[i].lower() and char in char_presence:
                    char_presence.remove(char)
        elif "dies" in actions[i]:
            add_line()
            sentence = ""
            for char in unique_character:
                if char in actions[i].lower() and char in char_presence:
                    char_presence.remove(char)

results_df = pd.DataFrame(np.array(char_presence_table).astype(int), columns=unique_character)
results_df["act"] = acts
results_df["scene"] = scenes
results_df["speaking"] = speaking_chars
results_df["text"] = sentences

results_df = results_df.reindex(columns=["act", "scene", "speaking", "text"] + unique_character)

results_df.to_csv(output_tsv_path, sep="\t")
