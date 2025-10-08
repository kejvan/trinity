import torch
import torch.nn as tnn
import torch.nn.functional as tnf
from data import get_batch
from data import vocab_size, train_dataset, test_dataset


# ------------------------------------------
# Hyperparameters and flags
# ------------------------------------------

# Hyperparameters
from hyperparams import (
    max_epoch,
    learning_rate,
    context_length,
    batch_size,
    report_interval,
)

# Feature flags
enable_loss_report = True
enable_initial_generation = True
enable_post_training_generation = True
enable_plotting = False


# ------------------------------------------
# Device
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------
# Bigram language model
# ------------------------------------------


class BigramLanguageModel(tnn.Module):
    def __init__(self, vocab_size, embedding_dim=None):
        super().__init__()
        if embedding_dim is None:
            # Direct vocab_size x vocab_size embedding table
            self.token_embedding_table = tnn.Embedding(vocab_size, vocab_size)
            self.lm_head = None
        else:
            # Smaller embedding with linear projection
            self.token_embedding_table = tnn.Embedding(vocab_size, embedding_dim)
            self.lm_head = tnn.Linear(embedding_dim, vocab_size)

    def forward(self, input_idx, targets=None):
        embeddings = self.token_embedding_table(input_idx)
        if self.lm_head is None:
            logits = embeddings
        else:
            logits = self.lm_head(embeddings)

        if targets is None:
            loss = None
        else:
            # Flatten for cross entropy calculation (B: batch, T: context, C: vocab)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = tnf.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
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

model = BigramLanguageModel(vocab_size).to(device)

if enable_initial_generation:
    input_text = "\n"
    max_new_tokens = 200
    output_init = generate_text(model, input_text, max_new_tokens)
    print(f"Model output before training: {output_init}")
    print()


# ------------------------------------------
# Training
# ------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

epochs = range(max_epoch)
loss_history = {"train": [], "validation": []}

for epoch in epochs:
    # Training step
    model.train()
    x_batch, y_batch = get_batch(train_dataset, context_length, batch_size)
    logits, loss = model.forward(x_batch, y_batch)

    if loss is not None:
        loss_history["train"].append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        xv_batch, yv_batch = get_batch(test_dataset, context_length, batch_size)
        _, vloss = model.forward(xv_batch, yv_batch)

        if vloss is not None:
            loss_history["validation"].append(vloss.item())

    if enable_loss_report:
        if (epoch + 1) % report_interval == 0:
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
