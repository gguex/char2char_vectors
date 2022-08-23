import pandas as pd
import re

# -------------------------------
#  Parameters
# -------------------------------

min_occurrences_threshold = 2

# -------------------------------
#   Code
# -------------------------------

# Load entities and token list
entities_df = pd.read_csv("corpora/relationship_corpora_characters/candide.entities", sep="\t")
token_df = pd.read_csv("corpora/relationship_corpora_characters/candide.tokens", sep="\t")

# Remove -1 entities and keep characters
entities_df = entities_df.loc[(entities_df.COREF > 0) & (entities_df.cat == "PER")]

# Build a dictionary for unique entities
id_to_name = {}
for entity_id in set(entities_df.COREF):
    entity_df = entities_df.loc[entities_df.COREF == entity_id]
    if len(entity_df) >= min_occurrences_threshold:
        id_shortest_name = entity_df.text.str.len().idxmin()
        entity_short_name = entity_df.text.loc[id_shortest_name]
        id_to_name[entity_id] = entity_short_name.lower().strip()

# Characters list
characters = list(set(id_to_name.values()))

# To create a dataframe
output_data = []
# Loop on sentence id
for sentence_id in token_df.sentence_ID.unique():
    # Select the sentence id
    sentence_df = token_df[token_df.sentence_ID == sentence_id]
    # Get paragraph id
    paragraph_id = sentence_df.paragraph_ID.min()
    # Get sentence
    text = " ".join([re.sub("['_]", " ", word) for word in sentence_df.word if word != "\ufeff"])
    # Get characters ids
    char_ids = [int(char_id) for char_id
                in list(entities_df.join(sentence_df, on="start_token", how="right").COREF.dropna())]

    # Build row and append to dataframe
    row_dict = {char: 0 for char in characters}
    for char_id in char_ids:
        if char_id in id_to_name:
            row_dict[id_to_name[char_id]] += 1
    row_dict["paragraph_id"] = paragraph_id
    row_dict["text"] = text
    output_data.append(row_dict)

results_df = pd.DataFrame(output_data, columns=["paragraph_id", "text"] + characters)
