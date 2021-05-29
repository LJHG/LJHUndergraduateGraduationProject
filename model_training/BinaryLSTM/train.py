import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from MultiLayerLSTM import MultiLayerLSTM
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
sequence_length = 28
input_size = 28
#hidden_size = 128
#num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 5
#learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = MultiLayerLSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first 为 True则输入输出的数据格式为 (batch, seq, feature)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x (100,28,28)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 当num_layer为1时，h0 (1,100,128) 即为 (num_layer, batch_size, hidden_size) 为batch_size的每一个输入初始化一个h0 c0? 不过这里没区别哈，都是0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)  (100,28,128)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out








# 开始tune各种hyper params
hidden_size_list= [20,50,100,128,256]
num_layers_list = [1,2,3]
learning_rate_list = [0.001,0.01,0.1,0.5]

best_model = None
acc = 0.1
best_hidden_size = 0
best_num_layers = 0
best_learning_rate = 0

for num_layers in num_layers_list:
    for hidden_size in hidden_size_list:
        for learning_rate in learning_rate_list:
            print("***********************************")
            print("hidden_size:{}, num_layers:{}, learning_rate:{} ".format(hidden_size,num_layers,learning_rate))
            # plot settings
            iters = []
            losses = []
            model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            start = time.time()
            # Train the model
            total_step = len(train_loader)
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, sequence_length, input_size).to(device) # 这里的-1应该是表示待定，这里的意思就是把tensor转换为 a * 28 * 28

                    # break
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 100 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                        iters.append( (epoch)*600+i+1)
                        losses.append(loss.item())

            end = time.time()
            print("时间:",end - start)
            plt.plot(iters,losses)
            plt.savefig("hs{}_nl{}_rate{}.jpg".format(hidden_size,num_layers,learning_rate*1000))
            plt.clf()
            #plt.show()
            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                if( (100 * correct / total) > acc):
                    acc = (100 * correct / total)
                    best_model = model
                    best_hidden_size = hidden_size
                    best_learning_rate = learning_rate
                    best_num_layers = num_layers
                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
            print("***********************************")

# Save the model checkpoint
torch.save(best_model.state_dict(), 'model.ckpt')
print("best acc is {},   hidden_size:{}, num_layers:{}, learning_rate:{} ")