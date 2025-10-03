from typing import List
import torch


# ------------------------------------------
# Hyperparameters and flags
# ------------------------------------------

# Hyperparameters
from hyperparams import (
    min_passage_length,
    train_split_ratio,
    random_seed,
    default_context_length,
    default_batch_size,
)

# Feature flags
enable_file_info = False
enable_vocab_info = False
enable_dataset_info = False
enable_tokenization_demo = False
enable_batch_demo = False


# ------------------------------------------
# Device
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------
# File input
# ------------------------------------------

# Load the complete Shakespeare text corpus
with open("../data/tiny_shakespeare.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

if enable_file_info:
    print(f"Total characters = {len(input_text)}")
    print()


# ------------------------------------------
# Vocabulary
# ------------------------------------------

# Extract unique characters and create sorted vocabulary
vocab_set = set(input_text)
vocab_list = sorted(list(vocab_set))
vocab_size = len(vocab_list)
chars = "".join(vocab_list)

if enable_vocab_info:
    print(f"Vocabulary size (distinct characters): {vocab_size}")
    print(f"Vocabulary: {ascii(chars)}")
    print()

# Split text into passages of at least 1000 characters, breaking at double newlines
dataset = []
left, right = 0, 0
while right < len(input_text):
    if right - left >= min_passage_length and input_text[right : right + 2] == "\n\n":
        dataset.append(input_text[left:right])
        left = right + 2
    right += 1

# Create train/test split
split_index = int(train_split_ratio * len(dataset))
train_list = dataset[:split_index]
test_list = dataset[split_index:]

if enable_dataset_info:
    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size ({int(train_split_ratio * 100)}%): {len(train_list)}")
    print(
        f"Test dataset size ({int((1 - train_split_ratio) * 100)}%): {len(test_list)}"
    )
    print()


# ------------------------------------------
# Tokenization
# ------------------------------------------

# Create character-level tokenization mappings
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> List[int]:
    """Convert string to list of character token IDs"""
    encoding = []
    for char in s:
        encoding.append(str_to_int[char])

    return encoding


def decode(nums: List[int]) -> str:
    """Convert list of token IDs back to string"""
    decoding = []
    for num in nums:
        decoding.append(int_to_str[num])

    return "".join(decoding)


if enable_tokenization_demo:
    print(f"Encoding of 'Shakespeare': {encode('Shakespeare')}")
    print(f"Decoding of 'Shakespeare': {decode(encode('Shakespeare'))}")
    print()


# ------------------------------------------
# Dataset
# ------------------------------------------

# Convert text passages to tokenized tensors and concatenate
train_dataset = torch.cat(
    [torch.tensor(encode(passage), dtype=torch.long) for passage in train_list]
).to(device)
test_dataset = torch.cat(
    [torch.tensor(encode(passage), dtype=torch.long) for passage in test_list]
).to(device)

if enable_dataset_info:
    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")
    print()


# ------------------------------------------
# Batch processing
# ------------------------------------------

torch.manual_seed(random_seed)


def get_batch(
    data, context_length=default_context_length, batch_size=default_batch_size
):
    """Generate random batches of input/target sequences for training"""
    input_sequences = []
    target_sequences = []
    random_start_indices = torch.randint(len(data) - context_length, size=(batch_size,))
    for start_idx in random_start_indices:
        input_sequences.append(data[start_idx : start_idx + context_length])
        target_sequences.append(data[start_idx + 1 : start_idx + context_length + 1])
    return torch.stack(input_sequences), torch.stack(target_sequences)


if enable_batch_demo:
    # Generate sample batches to demonstrate the data structure
    train_x, train_y = get_batch(train_dataset, batch_size=2)
    test_x, test_y = get_batch(test_dataset, batch_size=2)

    print(f"A test sequence with batch size {len(train_x)}:")

    # Show how input sequences map to target predictions
    for b in range(len(train_x)):
        print(f"Batch {b}:")
        for t in range(len(train_x[b])):
            xb, yb = train_x[b], train_y[b]
            context = [x.item() for x in xb[0 : t + 1]]
            target = yb[t]
            print(f"    Input: {context}, output: {target}")
    print()
