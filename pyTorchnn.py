import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv

with open('SPECT.train.csv', 'r') as f:
    train_data = list(csv.reader(f))
train_data = map(lambda x: map(lambda y: float(y), x), train_data) #the data comes in as strings.
with open('SPECT.test.csv', 'r') as f:
    test_data = list(csv.reader(f))
test_data = map(lambda x: map(lambda y: float(y), x), test_data)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(22, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

learning_rate = 0.1
for example in train_data:
    input = Variable(torch.FloatTensor(example[0:-1]))
    out = net(input)
    target = example[-1]
    criterion = nn.MSELoss()
    loss = criterion(out, Variable(torch.FloatTensor([target])))
    net.zero_grad()
    loss.backward()
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

errors = 0
for example in test_data:
        input = Variable(torch.FloatTensor(example[0:-1]))
        out = net(input)
        if abs(out.data[0] - example[-1]) > 0.5:
            errors += 1

print errors
print len(test_data)
