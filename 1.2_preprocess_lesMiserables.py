import re
import pandas as pd

# Input/output paths
input_corpus_path = "corpora/LesMiserables1_fr.txt"
output_tsv_path = "corpora/LesMiserables_pp.txt"

# Loading the corpus
with open(input_corpus_path) as corpus_file:
    corpus_lines = corpus_file.readlines()

