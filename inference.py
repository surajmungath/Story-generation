import torch
import torch.nn as nn
import numpy as np
import pickle
import pyttsx3


# Define the CharRNN model
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

# Load the vocabulary and mappings
with open("char_to_idx.pkl1", "rb") as f:
    char_to_idx = pickle.load(f)
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(char_to_idx)

# Model configuration
hidden_size = 256  # Must match the trained model
num_layers = 2     # Must match the trained model

# Load the trained model
model = CharRNN(vocab_size, hidden_size, num_layers)
model.load_state_dict(torch.load("storygen.pth",weights_only=True))
model.eval()

# Function to generate text
def generate(model, start_str, predict_len=100, temperature=0.7):
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

# Generate a story
start_string = input("Enter the start of your story: ")
generated_story = generate(model, start_string, predict_len=500)
print("Generated Story:")
print(generated_story)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
     # Set the voice property 0 for male voice 1for female voice
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 160)  # Speed of speech
engine.setProperty('volume', 1.2)  # Volume (0.0 to 1.0)

# Convert text to speech
engine.say(generated_story)

# Play the speech
engine.runAndWait()
