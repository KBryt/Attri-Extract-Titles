import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'  # the visdom environment
    vis_port =8097 # the visdom port
    model = 'ORMTag2022'  # For the model used, the name must match the name in the models/__init__.py
    pretrained_bert_name = 'bert-base-uncased'

    pickle_path = 'Type.pkl'
    load_model_path = None  # The path to load the pre-trained model, which stands for None is not loaded

    batch_size = 32  # batch size
    embedding_dim = 768
    hidden_dim = 1024
    tagset_size = 4
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 20
    lr = 2e-5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # L2 regular
    dropout = 0.2
    seed = 1234
    device = 'cuda'


    def _parse(self, kwargs):
        """
        Update the config parameter according to the dictionary kwargs
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
