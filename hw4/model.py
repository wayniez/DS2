import torch.nn as nn


class NeuralNet(nn.Module):
    """
    A fully connected (feedforward) neural network for classifying intentions
    based on a bag-of-words vector.

    Why feedforward, not RNN/LSTM/Transformer:
    - The input is a bag-of-words of fixed dimension (word order is irrelevant),
      so recurrent and transformer architectures offer no advantage;
    - the dataset is small (tens to hundreds of examples per class)—complex models
      will simply overfit and learn more slowly;
    - a simple MLP with 2 hidden layers trains in seconds on a CPU,
      which is convenient for a local Telegram bot.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out
