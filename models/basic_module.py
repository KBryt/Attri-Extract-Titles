#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):
    """
    Encapsulated nn. Module, mainly provides save and load two methods
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# the default name

    def load(self, path):
        """
        The model for the specified path can be loaded
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        Save the model, using "Model Name + Time" as the file name by default
        """
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    Put the reshape into（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
