# ------------------------------------------
# Hyperparameters
# ------------------------------------------

# Data preprocessing
MIN_PASSAGE_LENGTH = 1000
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Default batch parameters
DEFAULT_CONTEXT_LENGTH = 8
DEFAULT_BATCH_SIZE = 4

# Model parameters
MAX_EPOCH = 20000
LEARNING_RATE = 3e-4
CONTEXT_LENGTH = 8
BATCH_SIZE = 32
REPORT_INTERVAL = MAX_EPOCH / 5
EMBEDDING_DIM = 32
HEAD_SIZE = 16
