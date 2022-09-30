# positive 1 depressed
# negative 0 non-depressed

# import pickle
import pickle5 as pickle
import torch.nn.functional as F
import torch
from torchsummary import summary
from model import CNN
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from collections import Counter
from scipy import stats
from itertools import chain
# import lightgbm as lgb 
import time

import sys
np.set_printoptions(threshold=sys.maxsize)

import random, os
import torch
import numpy as np
seed = 32
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic =True

vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'N':5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'vowel_CNN.pth'

cnn = CNN().to(device)
cnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# print(summary(cnn, (1, 128, 28)))
# exit()

with open('../feature/dev_v4.pickle', 'rb') as handle:
    dev_features = pickle.load(handle)

with open('../feature/dev_labels.pickle', 'rb') as handle:
    dev_labels = pickle.load(handle)

for key in dev_features.keys():
    speaker_feature = []
    speaker_length = []
    speaker_grad = []
    print(key, len(dev_features[key]))

    for x in dev_features[key]:
        speaker_length.append(x.shape[1])
        x = Variable(torch.Tensor(x)).to(device)
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=0).requires_grad_()

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        cnn.eval()
        
        cnn.fc1.register_forward_hook(get_activation('fc1'))
        out = cnn(x)
        inter_value = activation['fc1'].clone()
        inter_value = torch.flatten(inter_value, start_dim=0).data.cpu().tolist()
        speaker_feature.append(inter_value)

        out = F.softmax(out, dim=1)
        output_idx = out.argmax(dim=1)
        output_max = out[0, output_idx]
        output_max.backward()

        saliency = x.grad.data.abs().reshape(128, speaker_length[-1]).cpu().numpy()

        saliency_total = np.sum(saliency)
        speaker_grad.append(saliency_total)

    np.save('length/'+str(key)+'.npy', speaker_length)
    np.save('saliency/'+str(key)+'.npy', speaker_grad)
    np.save('feature_fc1/'+str(key)+'.npy', speaker_feature)












