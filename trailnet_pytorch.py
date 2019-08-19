import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn
import collections, numpy, h5py

class TrailNet3Class(nn.Module):

    def __init__(self):
        super(TrailNet3Class, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4)
        self.pool3 = nn.MaxPool2d((2, 2), stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, padding=(1, 1))
        self.pool4 = nn.MaxPool2d((2, 2), stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(800, 200)
        self.fc2 = nn.Linear(200, 3)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cudnn.benchmark = True

net = TrailNet3Class()
net.cuda()
# print(net)

state_dict = h5py.File('trailnet_3class_real_vr_mix_color.h5', 'r')
net.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in net.named_parameters() if k in l})



# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight



# tmp1,tmp2 = 0,0
# test_step = 1
# batchsz = 256
# Drytest = 5
# for i in range(1,Drytest + test_step+1):
#     print 'i=', i
#     # Forward
#     input = Variable(torch.randn(batchsz, 1, 101, 101)).cuda()   ##Dummy input
#     torch.cuda.synchronize()
#     tf = time.time()
#     out = net(input)
#     torch.cuda.synchronize()
#     elapsed1= time.time() -tf
#     tmp1 += elapsed1
#     #print(out)
    
#     # Backward
#     '''
#     net.zero_grad()
#     out.backward(torch.randn(1, 7))            
#     '''
#     # Loss function
#     output = net(input)
#     target = Variable(torch.randn(batchsz, 7)).cuda()  # a dummy target, for example

#     Optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
#     criterion = nn.MSELoss()
#     loss = criterion(output, target)
#     print(loss)

#     #print(loss.grad_fn)  # MSELoss
#     #print(loss.grad_fn.next_functions[0][0])  # Linear
#     #print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

#     # Back prop.

#     net.zero_grad()     # zeroes the gradient buffers of all parameters
#     #print('conv1.bias.grad before backward')
#     #print(net.conv1.bias.grad)
#     tb = time.time()
#     loss.backward()
#     torch.cuda.synchronize()
#     elapsed2 = time.time()- tb
#     tmp2 += elapsed2
    
#     #print('conv1.bias.grad after backward')
#     #print(net.conv1.bias.grad)
    
#     if i<= Drytest:
#         tmp1,tmp2 = 0, 0

#     print tmp1, ' ', tmp2

# print 'fp:' ,tmp1/test_step, ' bp: ',tmp2/test_step





