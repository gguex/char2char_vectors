import pandas as pd

entities_df = pd.read_csv("corpora/relationship_corpora_characters/candide.entities", sep="\t")
token_df = pd.read_csv("corpora/relationship_corpora_characters/candide.tokens", sep="\t")

entities_df = entities_df[entities_df.COREF > 0]

with open("corpora/relationship_corpora/candide.txt") as corpus_file:
    corpus_txt = corpus_file.read()

corpus_txt[47110:47113]