import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# ---------------------------------------------------------------------------
# Logging the training process: writing to both the console and training.log
# ---------------------------------------------------------------------------
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_file_handler = logging.FileHandler("training.log", encoding="utf-8", mode="a")
_console_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler.setFormatter(_formatter)
_console_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Data Loading and Preparation
# ---------------------------------------------------------------------------
logger.info("Loading intents.json")
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!", ",", ":", ";"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

logger.info(f"Dataset: {len(xy)} examples, {len(tags)} intents: {tags}")
logger.info(f"Vocabulary size (unique stems): {len(all_words)}")

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
log_every = 50  # how often to write loss to log

logger.info(
    "Hyperparameters: "
    f"epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, "
    f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}"
)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device for training: {device}")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
logger.info("Starting training")
best_loss = float("inf")

for epoch in range(num_epochs):
    epoch_losses = []
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    if avg_loss < best_loss:
        best_loss = avg_loss

    if (epoch + 1) % log_every == 0 or epoch == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {avg_loss:.4f}, best_loss: {best_loss:.4f}"
        )

logger.info(f"Training completed. Final loss: {avg_loss:.4f}")

# ---------------------------------------------------------------------------
# Model Saving
# ---------------------------------------------------------------------------
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)
logger.info(f"Model saved to file {FILE}")
