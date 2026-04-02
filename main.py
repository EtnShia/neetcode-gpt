import torch
from model import GPT
from train import Solution as Trainer
from generate import Solution as Generator

# ---------------------------
# Hyperparameters
# ---------------------------
context_length = 32
model_dim = 64
num_blocks = 2
num_heads = 4
batch_size = 8
epochs = 50       # increase for better learning
lr = 1e-3
new_chars = 200   # number of characters to generate

# ---------------------------
# Load small text dataset
# ---------------------------
# Here we just use a small string. You can replace it with a file.
text = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,"
)

# with open("text.txt") as f:
#     text = f.read()

# ---------------------------
# Build character vocabulary
# ---------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for ch, i in char_to_int.items()}

# Convert text to tensor of token IDs
data = torch.tensor([char_to_int[ch] for ch in text], dtype=torch.long)

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
start_context = data[:context_length].unsqueeze(0)  # initial context
generated_text = generator.generate(
    model=model,
    new_chars=new_chars,
    context=start_context,
    context_length=context_length,
    int_to_char=int_to_char
)

print("\nGenerated text:\n")
print(generated_text)
