import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from logzero import logger

epochs = 1
batch_size = 64


class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024) # Input vector would be flattened image of size 28 x 28
        self.fc2 = nn.Linear(1024, 10) # output is 10 as we are classifying hand written digits

    def forward(self, x):
        x = self.fc1(x.view(-1, 28*28))
        x = F.relu(x)
        x = self.fc2(x)

        return x


transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/train',
                   train=True,
                   download=True,
                   transform=transformations),
    batch_size=batch_size,
    shuffle=True
)

net = SimpleNeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

net.to(device)

for epoch in range(epochs):
    for batch_idx, (input, label) in enumerate(train_loader, 0):
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        output = net(input)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        # Print statistics
        if batch_idx%50 == 0:
            logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, loss.item()))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/test',
                   train=False,
                   download=True,
                   transform=transformations),
    batch_size=64,
    shuffle=True
)

net.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for _, (input, label) in enumerate(test_loader, 0):
        input = input.to(device)
        label = label.to(device)
        output = net(input)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    logger.info(f'Total Correct: {correct}')
    logger.info('Accuracy: %.3f' % (correct / len(test_loader.dataset)))