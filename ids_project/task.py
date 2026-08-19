"""Model definition, training/evaluation loops and data loading for the
federated Intrusion Detection System.

Note: raw partitions are expected as pre-processed PyTorch tensors saved
under data/train_partition_<id>.pt and data/test_partition_<id>.pt.
The preprocessing / Dirichlet-partitioning script that produces these files
is being finalized separately (see README > Known limitations).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Lightweight MLP for binary traffic classification (benign vs. attack).

    Designed to stay small enough to be plausible on resource-constrained
    (embedded) clients: two hidden layers (64 -> 32) with dropout.
    """

    def __init__(self, input_size: int = 80):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)


def load_data(partition_id: int, batch_size: int):
    """Loads the pre-partitioned train/test tensors for a given client."""
    try:
        train_data = torch.load(f"data/train_partition_{partition_id}.pt")
        test_data = torch.load(f"data/test_partition_{partition_id}.pt")
    except FileNotFoundError:
        print(
            f"Error: Data for partition {partition_id} not found. "
            "Please run the data preparation script first."
        )
        return None, None

    trainloader = DataLoader(train_data, batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size)
    return trainloader, testloader


def train(net, trainloader, num_epochs, learning_rate, verbose=False):
    """Local training loop (binary cross-entropy, Adam)."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(X_batch)
            # NOTE: labels are binarized here (>0 -> attack). This assumes
            # the upstream OrdinalEncoder maps "BENIGN" to 0 -- true given
            # alphabetical ordering of the current label set, but not
            # asserted anywhere. Flagged as a known fragility (see README).
            y_batch = (y_batch > 0).float()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(trainloader)} Loss: {loss.item():.4f}")

        if verbose:
            print(f"Epoch {epoch+1} - Loss medio: {epoch_loss/len(trainloader):.4f}")


def evaluate(net, testloader):
    """Evaluation loop. Returns (loss, accuracy).

    NOTE: only loss/accuracy are computed today. Precision/recall/F1 on the
    attack class are on the roadmap (see README > Known limitations) since
    accuracy alone is a weak signal on an imbalanced binary task.
    """
    net.eval()
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            outputs = net(X_batch)
            y_batch = (y_batch > 0).float()
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy


def get_parameters(net):
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict)


def get_frac_attack(trainloader):
    """Fraction of attack samples in a client's local training data.

    Used by FedCustom to weight each client's contribution by how balanced
    its local class distribution is (see FedCustom.py).
    """
    total = 0
    attacks = 0
    for _, y_batch in trainloader:
        y_bin = (y_batch > 0).float()
        attacks += y_bin.sum().item()
        total += y_bin.size(0)
    return attacks / total if total > 0 else 0.0
