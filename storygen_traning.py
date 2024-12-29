import torch
import torch.nn as nn
import numpy as np

# Define the model architecture
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

# Load the character mappings (use the same character set as used during training)
chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n") + [
    chr(i) for i in range(128, 143)
] + ["€", "‚", "ƒ", "„", "…", "†", "‡", "ˆ", "–", "—"]

# Ensure there are 110 unique characters
if len(chars) != 110:
    raise ValueError(f"The character set does not have 110 unique characters. It has {len(chars)} characters.")
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Load the model
model = CharRNN(vocab_size, hidden_size=256, num_layers=2)
checkpoint_path = '/mnt/data/storygen.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval()

# Function to generate text
def generate(model, start_str, predict_len=100, temperature=0.8):
    model.eval()
    hidden = model.init_hidden(1)
    start_input = torch.tensor([char_to_idx[ch] for ch in start_str], dtype=torch.long).unsqueeze(0)
    predicted = start_str

    with torch.no_grad():
        for p in range(len(start_str) - 1):
            _, hidden = model(start_input[:, p].unsqueeze(0), hidden)
        inp = start_input[:, -1]

        for _ in range(predict_len):
            output, hidden = model(inp.unsqueeze(0), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            predicted_char = idx_to_char[top_i.item()]
            predicted += predicted_char
            inp = torch.tensor([top_i], dtype=torch.long)

    return predicted

# User input prompt
prompt = input("Enter a prompt to generate a story: ")

# Generate and print the story
generated_story = generate(model, prompt, predict_len=200)
print("\nGenerated Story:\n")
print(generated_story)
