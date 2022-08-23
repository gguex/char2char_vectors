from booknlp.booknlp import BookNLP
import torch

torch.cuda.empty_cache()

model_params={
		"pipeline":"entity", 
		"model":"big"
	}
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file="corpora/relationship_corpora/candide.txt"

# Output directory to store resulting files in
output_directory="corpora/relationship_corpora_characters"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="candide"

booknlp.process(input_file, output_directory, book_id)
