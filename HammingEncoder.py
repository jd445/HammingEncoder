import numpy as np
import torch
from data_generator import datainput, SequenceDataset
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class Round_by_columnMax(Function):
    @staticmethod
    def forward(self, input):
        idx = torch.argmax(input, dim=2, keepdims=True)
        output = torch.zeros_like(input).scatter_(2, idx, 1.)
        return output

    @ staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input



class Q_column_max(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def binary(self, input):
        # output = Binary_w01.apply(input)
        output = Round_by_columnMax.apply(input)
        return output

    def forward(self, input):
        output = self.binary(input)
        return output

class Conv2d_Q(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,A=2,W=2):
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.weight_quantizer = Q_column_max(W=W)

    def forward(self, input):
        tnn_bin_weight = self.weight_quantizer(self.weight)
        output = F.conv2d(
            input=input,
            weight=tnn_bin_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return output


def generate_kmer_pattern(kernel_weight, itemset):
    """
    Find the corresponding items in the itemset for each non-zero element in the kernel_weight matrix and reverse the one-hot encoding process.
    For columns that are all zeros, add the symbol '_', and for columns that are all ones, add the symbol '*'.

    Args:
        kernel_weight[1024*20*5]: 1024 matrices of size 20 * 5, where most columns (with 20 elements) have only one non-zero element and the rest are zeros,
                       some columns are all zeros, and some are all ones.
        itemset: a list containing all possible items.

    Returns:
        A list containing all generated sequences, where each sequence corresponds to the reverse result of a column in the kernel_weight matrix.
    """
    # cuda to cpu
    kernel_weight = kernel_weight.cpu().detach().numpy()
    # 对每个矩阵进行处理
    result = []
    for mat in kernel_weight:
        sequence = []
        # 对矩阵的每一列进行处理
        for col in mat.T:
            if np.all(col == 1):
                sequence.append('*')
            else:
                indices = np.where(col != 0)[0]
                if len(indices) > 0:
                    index = indices[0]
                    sequence.append(itemset[index])
                else:
                    sequence.append('_')
            # if the sequence only consists of _ and *, then remove it
        if set(sequence) != {'_'} and set(sequence) != {'*'} and set(sequence) != {'_', '*'} and set(sequence) != {'*', '_'}:
            result.append(sequence)
    result = list(set([tuple(t) for t in result]))
    return result


def gen_itemset(X):
    itemset = set([item for sublist in X for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    return itemset


def seq_picture(db, itemset):
    # convert the sequence to a target data, which is sequence picture.
    seq_picture = []
    for i in tqdm(db):
        temp_seq_picture = np.zeros(
            [1, len(itemset), len(i)], dtype='int8')  # Only create the necessary size, no padding here
        for j in range(len(itemset)):
            for k in range(len(i)):
                if i[k] == itemset[j]:
                    temp_seq_picture[0, j, k] = 1
        seq_picture.append(temp_seq_picture)
    return seq_picture

def collate_fn(batch, kmer= 6):
    # Find the longest sequence in this batch
    max_length = max(max([item[0].shape[2] for item in batch]), kmer)
    batch_seq_pictures = []
    batch_labels = []
    
    for seq_picture, label in batch:
        # Pad each sequence to have the same length
        padded_seq = np.pad(seq_picture, ((0, 0), (0, 0), (0, max_length - seq_picture.shape[2])), mode='constant')
        batch_seq_pictures.append(padded_seq)
        batch_labels.append(label)


    batch_seq_pictures = np.vstack(batch_seq_pictures).astype(np.float32)  # Stack along the first dimension,    # current torch.Size([32, 94, 108]), but we need torch.Size([32, 1, 94, 108])
    batch_seq_pictures = np.expand_dims(batch_seq_pictures, axis=1)
    if batch[0][1] is None:
        return torch.from_numpy(batch_seq_pictures)

    return torch.from_numpy(batch_seq_pictures), torch.tensor(batch_labels)
 
def calculate_intensity(sequence, kmer_pattern):

    window_size = len(kmer_pattern)
    max_intensity = 0
    for i in range(len(sequence)-window_size+1):
        temp_intensity = 0
        for j in range(window_size):
            if sequence[i+j] == kmer_pattern[j]:
                temp_intensity += 1
            # elif kmer_pattern[j] == '*':
            #     temp_intensity += 1
        if temp_intensity > max_intensity:
            max_intensity = temp_intensity
    # print(max_intensity, str(kmer_pattern)+"other" + str(sequence))
    return max_intensity


def generate_feature_vector(sequence, kmer_pattern_list):
    feature_vector = []
    for kmer_pattern in kmer_pattern_list:
        feature_vector.append(calculate_intensity(sequence, kmer_pattern))
    return feature_vector

        
        
class HammingEncoder(nn.Module):
    def __init__(self, X, Y, gap_constrain=5, label_number=2, Preset_set_pattern_num=100):
        super(HammingEncoder, self).__init__()

        self.X = X
        self.Y = Y
        self.item_set = gen_itemset(X) 
        item_size = len(self.item_set)
        self.cnn = nn.Sequential(
            Conv2d_Q(1,
                     Preset_set_pattern_num,
                     kernel_size=(item_size, gap_constrain),
                     stride=1,
                     padding=0,
                     bias=False),

        )

        # self.linear1 = Linear_Q(Preset_set_pattern_num,
        # label_number, bias=False)
        self.linear1 = nn.Linear(
            Preset_set_pattern_num, label_number)
        # self.linear2 = nn.Linear(int(Preset_set_pattern_num/2), label_number)

    def forward(self, x):
        out = self.cnn(x)

        out, _ = out.max(-1)

        out = out.view(out.size()[0], -1)
        out = torch.squeeze(out)
        # save out for debug
        # np.savetxt('out.txt', out.cpu().detach().numpy().reshape(-1), fmt='%f')
        out = self.linear1(out)
        # out = self.linear2(out)

        return out
    
    def get_kmers(self):
        # conv1_weight = model.state_dict()['module.cnn.0.weight']
        conv1_weight = self.cnn[0].weight
        conv1_weight = Round_by_columnMax.apply(conv1_weight)
        conv1_weight = conv1_weight.squeeze()
        # decoupling
        kmer_patterns = generate_kmer_pattern(conv1_weight, self.item_set)
        
        return kmer_patterns
    
    def fit(self, n_epochs=100, patience=50, batchsize=64, kmer_length=5, patten_number=256,device='cuda'):
        criterion = nn.CrossEntropyLoss()        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0003, weight_decay=1e-5)
        total_acc = []
        total_loss = []
        seq_pictures = seq_picture(self.X, self.item_set)
        data_set = SequenceDataset(
            seq_pictures, self.Y)
        data_loader = DataLoader(
            data_set, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True,  collate_fn=collate_fn)
        # start training
        stale = 0
        best_acc = 0
        best_train_loss = float('inf')
        for epoch in tqdm(range(n_epochs)):
            # Make sure the model is in train mode before training.
            self.train()
            # These are used to record information in training.
            train_loss_list = []
            train_accs = []
            for batch in data_loader:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = self(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_correct = (outputs.argmax(dim=-1)
                                    == labels).float().mean()
                train_loss_list.append(loss.item())
                train_accs.append(train_correct)
            # save loss list to csv
            train_loss = sum(train_loss_list) / len(train_loss_list)
            train_acc = sum(train_accs) / len(train_accs)

            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

            if train_loss < best_train_loss:
                best_model = self
                best_train_loss = train_loss
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(
                        f"No improvment {patience} consecutive epochs, early stopping")
                    break
            total_loss.append(train_loss)
            self = best_model
    
    def transform(self, input_seqs):
        kmer_patterns = self.get_kmers()
        feature_vectors = []
        for seq in input_seqs:
            feature_vectors.append(generate_feature_vector(seq, kmer_patterns))
        return feature_vectors
    
    