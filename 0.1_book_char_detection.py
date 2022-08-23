from booknlp.booknlp import BookNLP
import torch

# Empty cache for memory
torch.cuda.empty_cache()

# The model pipeline
model_params = {"pipeline": "entity", "model": "big"}

# The model
booknlp = BookNLP("en", model_params)

# Input file to process
input_file = "corpora/booknlp_corpora/candide.txt"

# Output directory to store resulting files in
output_directory = "corpora/booknlp_corpora_characters"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id = "candide"

booknlp.process(input_file, output_directory, book_id)