import os
import torch
from torch import nn
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision import transforms


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare FakeData dataset
    dataset = FakeData(size=15000, image_size=(3, 28, 28), num_classes=2, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Prepare targets
            targets = targets \
                .type(torch.FloatTensor) \
                .reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')