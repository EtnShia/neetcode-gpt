import torch
import torch.nn as nn
from torchtyping import TensorType

class GeneratorSolution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict, temperature: float = 0.8) -> str:
        torch.manual_seed(0)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        result = []
        for _ in range(new_chars):
            # Crop context to max length the model can handle
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            # Forward pass -> logits for every position
            logits = model(context)                          # (1, T, vocab_size)
            last_logits = logits[:, -1, :]                   # (1, vocab_size)

            # Apply temperature before softmax
            probs = nn.functional.softmax(last_logits / temperature, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1)

            # Append token to context and decode
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        return ''.join(result)
