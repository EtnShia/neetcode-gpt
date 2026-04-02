
import torch
from model import GPT
from train import Solution as Trainer
from generate import Solution as Generator

# ---------------------------
# Hyperparameters
# ---------------------------
vocab_size = 65       # number of unique tokens
context_length = 32   # sequence length
model_dim = 64        # embedding / hidden size
num_blocks = 2        # number of transformer blocks
num_heads = 4         # number of attention heads
batch_size = 8
epochs = 5
lr = 1e-3
new_chars = 100       # number of characters to generate

# ---------------------------
# Dummy dataset
# ---------------------------
# Generate a random sequence of token IDs
data_length = 1000
data = torch.randint(0, vocab_size, (data_length,))

# Simple int_to_char mapping (for generation)
# Maps 0 -> 'A', 1 -> 'B', etc.
int_to_char = {i: chr(65 + i) for i in range(vocab_size)}

# ---------------------------
# Create GPT model
# ---------------------------
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)

# ---------------------------
# Train the model
# ---------------------------
trainer = Trainer()
final_loss = trainer.train(
    model=model,
    data=data,
    epochs=epochs,
    context_length=context_length,
    batch_size=batch_size,
    lr=lr
)
print("Final training loss:", final_loss)

# ---------------------------
# Generate text
# ---------------------------
generator = Generator()

# Start context: first sequence from data
start_context = data[:context_length].unsqueeze(0)  # shape (1, context_length)
generated_text = generator.generate(
    model=model,
    new_chars=new_chars,
    context=start_context,
    context_length=context_length,
    int_to_char=int_to_char
)
print("Generated text:")
print(generated_text)
