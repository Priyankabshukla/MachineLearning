import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from data import get_data_loader
from network import Network
from config import cfg

try:
    from termcolor import cprint
except ImportError:
    cprint = None

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def get_lr(optimizer):   # Check this.
    #TODO: Returns the current Learning Rate being used by
    # the optimizer
    for param_group in optimizer.param_groups:  
        return param_group['lr'] 
    
#     raise NotImplementedError

'''
Use the average meter to keep track of average of the loss or 
the test accuracy! Just call the update function, providing the
quantities being added, and the counts being added
'''
class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0
    
    def update(self, increment, count):
        self.qty += increment
        self.cnt += count
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else: 
            return self.qty/self.cnt


def run(net, epoch, loader, optimizer, criterion, logger, scheduler, train=True):
    # TODO: Performs a pass over data in the provided loader
    
    # TODO: Initalize the different Avg Meters for tracking loss and accuracy (if test)
    avg_train_loss = AvgMeter()
    avg_test_loss = AvgMeter()
    avg_test_acc = AvgMeter()

    # TODO: Iterate over the loader and find the loss. Calculate the loss and based on which
    # set is being provided update your model. Also keep track of the accuracy if we are running
    # on the test set.
    if train == True:
        for i, (image,label) in enumerate(loader): ###check the loop (enumerate or just loader?)
            optimizer.zero_grad()
            label = label.type(torch.LongTensor)
            outputs = net(image.float())
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            avg_train_loss.update(loss.item(),1)
        accuracy = 0 
        
        logger.add_scalar('Train Loss', avg_train_loss.get_avg(),epoch)
        return avg_train_loss.get_avg(), accuracy 
            
    else:   
        for i, (image,label) in enumerate(loader):
            label = label.type(torch.LongTensor)
            outputs = net(image.float())
            loss = criterion(outputs,label)
            pred = outputs.data.max(1, keepdim=True)[1]
            acc_i = pred.eq(label.data.view_as(pred)).cpu().float().mean()
            avg_test_acc.update(acc_i,1)
            avg_test_loss.update(loss.item(),1)
            
        logger.add_scalar('Test loss', avg_test_loss.get_avg(),epoch)
        return avg_test_loss.get_avg(),avg_test_acc.get_avg()
                

    # TODO: Log the training/testing loss using tensorboard. 
    
    
    # TODO: return the average loss, and the accuracy (if test set)
#     raise NotImplementedError
        

def train(net, train_loader, test_loader, logger):    
    # TODO: Define the SGD optimizer here. Use hyper-parameters from cfg
    optimizer = optim.SGD(net.parameters(),
                          lr = 0.01,
                          momentum = 0.9,
                          weight_decay = 0.0001,
                          nesterov = True)
    # TODO: Define the criterion (Objective Function) that you will be using
    criterion = nn.CrossEntropyLoss()
    # TODO: Define the ReduceLROnPlateau scheduler for annealing the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     factor=0.1,
                                                     patience = 0,
                                                     verbose = True)

    for i in range(cfg['epochs']):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, _ = run(net, i, train_loader, optimizer, criterion, logger, scheduler, train = True)

        
        # TODO: Get the current learning rate by calling get_lr() and log it to tensorboard
        logger.add_scalar("Learning Rate", get_lr(optimizer), i)
        
        
        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])

        
        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # TODO: HINT - you might need to perform some step before and after running the network
            # on the test set
            net.eval()
            
            # Run the network on the test set, and get the loss and accuracy on the test set 
            loss, acc = run(net, i, test_loader, optimizer, criterion, logger, scheduler, train=False)
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc*100.0)
            log_print(log_text, color='red', attrs=['bold'])

            # TODO: Perform a step on the scheduler, while using the Accuracy on the test set
            scheduler.step(acc)  
            # TODO: Use tensorboard to log the Test Accuracy 
            logger.add_scalar('Test accuracy', acc ,i) 
            
#            raise NotImplementedError

if __name__ == '__main__':
    # TODO: Create a network object
    net = Network()

    # TODO: Create a tensorboard object for logging
    writer = SummaryWriter()


    # TODO: Create train data loader
    train_loader = get_data_loader('train')


    # TODO: Create test data loader
    test_loader = get_data_loader('test')

    # Run the training!
    train(net, train_loader, test_loader, writer)