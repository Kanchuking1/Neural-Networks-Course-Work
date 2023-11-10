# Dev: Wahib Sabir Kapdi

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# print("Dataset Size : " + str(len(text)))

# Creating our token

chars = sorted(list(set(text)))
vocab_size = len(chars)

# print ("Vocab : " + ' '.join(chars))
# print ("Vocab Size : " + str(vocab_size))

# Building our encoders and decoders

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print (encode('Wahib'))
# print (decode(encode('Wahib')))

# Now that we are done encoding the text, Let's start building the model
import torch 
import torch.nn as nn
from torch.nn import functional as F

data = torch.tensor(encode(text), dtype=torch.long)
# print (data.shape, data.dtype)
# print (data[0:1000])

# Let's split our dataset into training and validation dataset.
n = int (0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Time to start training
block_size = 8
batch_size = 4

# Use block_size + 1 to generate 8 contexts
# x = train_data[:block_size]
# y = train_data[1:block_size + 1]

# for t in range(block_size):
#     context = train_data[:t + 1]
#     target = y[t]
#     print (f"when input is {context} the target: {target}")

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack ([data[i: i + block_size] for i in ix])
    y = torch.stack ([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

xb , yb = get_batch("train")
# print ("inputs : ")
# print (xb.shape)
# print (xb)

# print ("targets : ")
# print (yb.shape)
# print (yb)  

# Building a simple Bigram Model 
class BigramLanguageModel (nn.Module):
    def __init__ (self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward (self, idx, targets):
        logits = self.token_embedding_table (idx)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate (self, idx, max_new_tokens):

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)

print (logits.shape)
print (loss)