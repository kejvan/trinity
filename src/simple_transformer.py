import torch
import torch.nn as tnn
import torch.nn.functional as tnf
from data import vocab_size
from data import get_batch, train_dataset, test_dataset


# ------------------------------------------
# Hyperparameters and flags
# ------------------------------------------

# Hyperparameters
from hyperparams import (
    MAX_EPOCH,
    LEARNING_RATE,
    CONTEXT_LENGTH,
    BATCH_SIZE,
    REPORT_INTERVAL,
    EMBEDDING_DIM,
    HEAD_SIZE,
)

# Feature flags
enable_parameter_count = True
enable_loss_report = True
enable_initial_generation = True
enable_post_training_generation = True
enable_plotting = False


# ------------------------------------------
# Device
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------
# Attention head
# ------------------------------------------
# Size hints: B: batch, T: block, C: embedding, H: head, V: vocab


class Head(tnn.Module):
    def __init__(
        self,
        head_size=HEAD_SIZE,
        embedding_dim=EMBEDDING_DIM,
        block_size=CONTEXT_LENGTH,
    ):
        super().__init__()

        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size

        # (C, H)
        self.key = tnn.Linear(embedding_dim, head_size, bias=False)
        self.query = tnn.Linear(embedding_dim, head_size, bias=False)
        self.value = tnn.Linear(embedding_dim, head_size, bias=False)

        # (T_max, T_max)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # (B, T, C) * (C, H) -> (B, T, H)
        key, query, value = self.key(x), self.query(x), self.value(x)

        # (B, T, T)
        mask = self.tril[:T, :T] == 0

        # (B, T, H) * (B, H, T) -> (B, T, T)
        attention_weights = query @ key.transpose(-2, -1) / self.head_size**0.5
        attention_weights = attention_weights.masked_fill(mask, float("-inf"))
        attention_weights = tnf.softmax(attention_weights, dim=-1)

        # (B, T, T) * (B, T, H) -> (B, T, H)
        out = attention_weights @ value

        return out


class SimpleTransformer(tnn.Module):
    def __init__(
        self,
        vocab_size=vocab_size,
        head_size=HEAD_SIZE,
        embedding_dim=EMBEDDING_DIM,
        block_size=CONTEXT_LENGTH,
    ):
        super().__init__()

        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size

        # (V, C)
        self.token_embedding = tnn.Embedding(vocab_size, embedding_dim)

        # (T_max, C)
        self.position_embedding = tnn.Embedding(block_size, embedding_dim)

        # Takes (B, T, C), spits (B, T, H)
        self.sa_head = Head(
            head_size=head_size, embedding_dim=embedding_dim, block_size=block_size
        )

        # (H, V)
        self.lm_head = tnn.Linear(head_size, vocab_size)

    def forward(self, input_idx, targets=None):
        B, T = input_idx.shape
        positions = torch.arange(T, device=device)

        # (B, T, C)
        token_embedding = self.token_embedding(input_idx)

        # (T, C)
        position_embedding = self.position_embedding(positions)

        # (B, T, C) + (T, C) = (B, T, C)
        x = token_embedding + position_embedding

        # (B, T, H)
        x = self.sa_head(x)

        # (B, T, H) * (H, V) = (B, T, V)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = tnf.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            bs = self.block_size
            idx_cond = idx[:, -bs:] if idx.size(1) > self.block_size else idx
            logits, _ = self.forward(idx_cond)

            # Focus on last time step predictions
            logits = logits[:, -1, :]
            probs = tnf.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# ------------------------------------------
# Text generation
# ------------------------------------------


def generate_text(model, start_text, max_new_tokens):
    """Generate text starting from given input"""
    from data import encode, decode

    context = torch.tensor([encode(start_text)], device=device)
    generated = model.generate(context, max_new_tokens)
    return decode(generated[0].tolist())


# ------------------------------------------
# Model initialization
# ------------------------------------------

model = SimpleTransformer(vocab_size).to(device)

if enable_parameter_count:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    print()

if enable_initial_generation:
    input_text = "\n"
    max_new_tokens = 200
    output_init = generate_text(model, input_text, max_new_tokens)
    print(f"Model output before training: {output_init}")
    print()


# ------------------------------------------
# Training
# ------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

epochs = range(MAX_EPOCH)
loss_history = {"train": [], "validation": []}

for epoch in epochs:
    # Training step
    model.train()
    x_batch, y_batch = get_batch(train_dataset, CONTEXT_LENGTH, BATCH_SIZE)
    logits, loss = model.forward(x_batch, y_batch)

    if loss is not None:
        loss_history["train"].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        xv_batch, yv_batch = get_batch(test_dataset, CONTEXT_LENGTH, BATCH_SIZE)
        _, vloss = model.forward(xv_batch, yv_batch)

        if vloss is not None:
            loss_history["validation"].append(vloss.item())

    if enable_loss_report:
        if (epoch + 1) % REPORT_INTERVAL == 0:
            print(
                f"Epoch: {epoch + 1:5d}, loss: {loss:.4f}, validation loss: {vloss:.4f}"
            )

if enable_loss_report:
    print()

if enable_post_training_generation:
    input_text = "\n"
    max_new_tokens = 200
    output_final = generate_text(model, input_text, max_new_tokens)
    print(f"Model output after training: {output_final}")
    print()


# ------------------------------------------
# Loss visualization
# ------------------------------------------


def plot_loss(loss_history, epochs):
    """Plot training and validation loss curves"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 7
    plt.rcParams["figure.dpi"] = 300

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_history["train"], label="Training loss")
    plt.plot(epochs, loss_history["validation"], label="Validation loss")
    plt.legend()
    plt.show()


if enable_plotting:
    plot_loss(loss_history, epochs)
