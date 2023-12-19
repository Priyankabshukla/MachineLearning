import torch
import torch.nn as nn

def xavier_init(param):
    # TODO: Complete this to initialize the weights
    nn.init.xavier_uniform_(param.weight)
    if param.bias is not None:
        nn.init.zeros_(param.bias)
    
#     raise NotImplementedError

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Define the model architecture here
        self.fc1 = nn.Linear(784,100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,10)
#         raise NotImplementedError

        # TODO: Initalize weights by calling the
        # init_weights method
        self.init_weights()

    def init_weights(self):
        # TODO: Initalize weights by calling by using the
        # appropriate initialization function
        if type(self)==nn.Linear:
            self.xavier_init(param)
        
#         raise NotImplementedError

    def forward(self, x):
        # TODO: Define the forward function of your model

        '''
        Parameters:
            x - input data

        Returns:
            The prediction of the model
        '''
        x = x.view(x.shape[0],-1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
        
        
#         raise NotImplementedError
    
    def save(self, ckpt_path):
        # TODO: Save the checkpoint of the model
        torch.save(self.net.state_dict(),ckpt_path)
#         raise NotImplementedError

    def load(self, ckpt_path):
        # TODO: Load the checkpoint of the model
        torch.load(self.net.state_dict(),ckpt_path)
#         raise NotImplementedError