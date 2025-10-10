# Explaining design choices:

- data_io.py- Reading, downloading, and splitting datasets
- preprocessing.py- Tokenization, normalisation, stopword filtering
- embeddings.py- Vocabulary and pretrained embedding loading
- training.py- Model training 
- evaluation.py- Metrics and evaluation 
- plotting.py- Visualisation of results
- dataloaders.py- Turns tokenised data into padded PyTorch batches
- dataset_pipeline.py- Glue between config, IO, and preprocessing
- config.py- Typed loaders for YAML configs (data + embeddings)
- reports.py- Vocabulary and OOV summaries to keep notebooks tidy
- notebooks/02_rnn_baseline.py- Scaffold for Part 2 RNN experiments
- configs/experiments/part2_rnn.yaml- Baseline hyperparameter template
- notebooks/02_rnn_baseline.py- Scaffold for Part 2 RNN experiments
- configs/experiments/part2_rnn.yaml- Baseline hyperparameter template

I.e. line 22 data_io.py: Alright, using @dataclass(frozen=True) might be overkill. Chat suggested it to make things "read only" (to prevent config file mutations) and to make auto __init__ etc. Thought it was a nice addition.

Usage of spaCy: spaCy is more "aware" than normal whitespace tokenisation (e.g. splits “don’t” → “do” + “n’t”). Also ** NOTE ** Run this for downloading the correct language tokenisation package: python3.11 -m spacy download en_core_web_sm

Line 61 and 122 in embeddings.py:
instead of returning all elements separately, I thought that it would be better to create "organised objects" so that we ensure datatypes instead of doing {matrix:...}. Thought it had a clean look but feel free to change. 

Added typehints everywhere because it's easier to debug! 

Used GloVe mainly because it's smaller and simpler to use than Word2Vec

OOV mitigation: assign the global mean embedding to each OOV token before training so they at least start from a deterministic vector.
