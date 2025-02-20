import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

# Define a simple two-layer MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example input and target
input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
target = torch.tensor([[10.0]])

# Forward pass
output = model(input_tensor)
loss = criterion(output, target)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Visualize the computation graph
dot = make_dot(loss, params=dict(model.named_parameters()))

# Save and display the graph
dot.format = 'png'
dot.render('backpropagation_graph')
dot.view()
