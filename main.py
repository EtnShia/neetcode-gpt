import torch
Trainer = TrainerSolution
Generator = GeneratorSolution

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Hyperparameters
# ---------------------------
context_length = 16
model_dim = 64
num_blocks = 2
num_heads = 4
batch_size = 8
epochs = 200
lr = 1e-3
new_chars = 20
temperature = 1.2 # Increased temperature for more diverse output

# ---------------------------
# Load small text dataset
# ---------------------------

text = (
    """In the heart of a hidden valley, where the mountains curved like gentle waves and rivers sang songs older than time, lived a young unicorn named Lira. Her coat was the softest silver, and her mane sparkled with tiny motes of starlight that shimmered even in the darkest night. Lira had always been curious, more curious than any of the older unicorns who preferred the comfort of familiar meadows and sunlit streams.
One evening, as the sky melted into shades of lavender and deep indigo, Lira noticed something unusual: a flicker of golden light far beyond the meadow she called home. It danced between the trees, shifting and weaving as if it had a life of its own. The other unicorns were resting, but Lira’s heart thumped with an irresistible pull. She knew she had to follow it.
Silently, she stepped through the tall grass, her hooves barely making a sound. Fireflies floated around her, drawn to the shimmer of her mane, and the night air smelled sweet, like honey and moonflowers. The golden light led her past the familiar stream, where she paused to drink, and into the shadowed forest that whispered secrets in the rustle of leaves.
The deeper she went, the stranger the forest became. Trees twisted into shapes that seemed almost alive, their branches reaching out like gentle hands. Tiny creatures peeked from behind mushrooms, watching her with glowing eyes, but none dared approach. Soon, Lira reached a clearing she had never seen before. The flowers here glowed with colors she had no names for, and in the center of the clearing lay a pool that shimmered like liquid crystal.
The golden light hovered just above the pool’s surface, moving as though it were beckoning her closer. Lira felt a thrill of both fear and wonder. She lowered her head to peer into the water, and for a moment, she thought she saw shapes moving beneath it—shapes that didn’t belong to any creature she had ever known. A ripple formed, spreading outward in concentric circles, and the golden light began to swirl, growing brighter, pulsing with a rhythm that matched her own heartbeat.
Then, just as she reached forward, her reflection in the water shifted. It wasn’t her face looking back at her, but something… else. Something magical, ancient, and alive. And then the pool began to shimmer more intensely, as if inviting her to step closer and"""
)

# with open("text.txt") as f:
#     text = f.read()

# ---------------------------
# Build character vocabulary
# ---------------------------

# Split text into words
words = text.split()  # simple word-level tokenization

# Build vocabulary
vocab = sorted(set(words))
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}

# Convert text to integers
data = torch.tensor([word_to_int[w] for w in words], dtype=torch.long)

# Move the data to the same device as the model
data = data.to(device)

# Update vocab_size for GPT
vocab_size = len(vocab)

# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# char_to_int = {ch: i for i, ch in enumerate(chars)}
# int_to_char = {i: ch for ch, i in char_to_int.items()}

# # Convert text to tensor of token IDs
# data = torch.tensor([char_to_int[ch] for ch in text], dtype=torch.long)

# ---------------------------
# Create GPT model
# ---------------------------
model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)

# Move model to GPU
model = model.to(device)

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

# Move context to the same device
start_context = start_context.to(device)

generated_text = generator.generate(
    model=model,
    new_chars=new_chars,
    context=start_context,
    context_length=context_length,
    int_to_char=int_to_word,
    temperature=temperature # Pass the temperature parameter
)

print("\nGenerated text:\n")
print(generated_text)
