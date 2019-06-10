import os, numpy as np, torch, math
import torch.nn.functional as F

#device = torch.device('cpu')
device = torch.device('cuda')
max_seq_length = 500

class attn_net(torch.nn.Module):
    def __init__(self, l2=0.0001, dropout=0, input_dim=20,
                 output_dim = 3, hidden_size = 50,
                 conf_penalty = 0.05):
        super(attn_net, self).__init__()
        self.l2 = l2
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.conf_penalty = conf_penalty
        self.encoder_layer = encoder(hidden_size, input_dim, dropout)
        self.decoder_layer = attention_decoder(hidden_size)
        

    def forward(self, x, x_lengths, mask, current_device = 'cpu'):
        x, context = self.encoder_layer(x, x_lengths, mask)
        final_outputs = self.decoder_layer(mask, x_lengths, x, context, current_device)
        #anglepred = torch.matmul(x, self.angle_weights)*mask[:,:,0:4]
        return final_outputs#, anglepred

    def custom_loss(self, y_pred, y_true, output_mask, y_prob):
        logprobs = y_pred.clone()
        y_pred = y_pred * y_true * output_mask[:,:,0:self.output_dim]
        y_pred  = y_pred - self.conf_penalty * y_prob * logprobs
        loss = (-torch.sum(y_pred) / self.output_dim) / output_mask.shape[0]
        #angleloss = torch.sum((anglepred - phipsi_true)**2) / (4*output_mask.shape[0])
        return loss# + angleloss


    def train(self, seqs_data, struct_data, iterations = 31,
            minibatch = 20, lr=0.005, track_loss = True):
        torch.backends.cudnn.benchmark = True
        x, y = generate_3class_inputs(seqs_data, struct_data)
        #phipsi_tensor = generate_phi_psi(phi, psi)
        self.cuda()
        x.cuda()
        y.cuda()
        #phipsi_tensor.cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                 weight_decay = self.l2)
        print('lock and load')
        for i in range(0, iterations):
            mini_indices = np.random.choice(y.shape[0], minibatch,
                                    replace = False)
            x_mini = x[mini_indices,:,:].cuda()
            y_mini = y[mini_indices,:,:].cuda()
            #phi_mini = phipsi_tensor[mini_indices,:].cuda()
            #x_lengths = torch.sum(torch.sum(x_mini,2),1).cuda()
            x_lengths = torch.tensor([torch.nonzero(torch.sum(x_mini[j,:,:],1)).shape[0]
                                      for j in range(0, x_mini.shape[0])]).cuda()
            x_lengths,indices = torch.sort(x_lengths, descending=True)
            x_mini = x_mini[indices,:,:].float()
            y_mini = y_mini[indices,:,:].float()
            #phi_mini = phi_mini[indices,:,:].cuda()
            output_mask = generate_mask(x_mini, 100).cuda()
            y_prob = self.forward(x_mini, x_lengths,
                                            output_mask, current_device='cuda')
            y_pred = torch.log(y_prob.clamp(min=1*10**-10))
            loss = self.custom_loss(y_pred, y_mini,
                                    output_mask, y_prob.clone())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if track_loss == True and i % 100 == 0:
                print('Current loss: %s'%loss.item())

    def predict(self, sequences):
        with torch.no_grad():
            self.to(device='cpu')
            final_output_structs = []
            for j in range(0, int(len(sequences) / 100) + 1):
                if (j * 100) >= len(sequences):
                    break
                elif (j*100 + 100) >= len(sequences):
                    x = generate_3class_inputs(sequences[(j*100):])[0]
                else:
                    x = generate_3class_inputs(sequences[(j*100):(j*100 + 100)])[0]
                #x_lengths = torch.sum(torch.sum(x,2),1)
                x_lengths = torch.tensor([torch.nonzero(torch.sum(x[j,:,:],1)).shape[0]
                                      for j in range(0, x.shape[0])])
                x_lengths,indices = torch.sort(x_lengths, descending=True)
                original_indices = torch.zeros((x_lengths.shape[0])).long()
                for i in range(0, x_lengths.shape[0]):
                    original_indices[indices[i]] = i
                x = x[indices,:,:].float()
                output_mask = generate_mask(x, 100)
                output_structs = decode_output(self.forward(x, x_lengths, output_mask))[0]
                final_output_structs = final_output_structs + [output_structs[i]
                                                               for i in original_indices.numpy()]
                print(len(final_output_structs))
        return final_output_structs
            
############################
                                           
class encoder(torch.nn.Module):
    def __init__(self, hidden_size, input_dim, dropout=0):
        super(encoder, self).__init__()
        self.input_encoder = torch.nn.GRU(input_size = input_dim,
                                  hidden_size = hidden_size,
                                  num_layers = 1,
                                  batch_first = True,
                                  bidirectional = True,
                                  dropout = dropout)
    def forward(self, x, x_lengths, mask):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths,
                                                    batch_first = True)
        encoded_data, context_vector = self.input_encoder(x)
        context_vector = torch.cat([context_vector[0,:,:],
                                    context_vector[1,:,:]], dim=1)
        encoded_data = torch.nn.utils.rnn.pad_packed_sequence(encoded_data,
                                                            batch_first = True,
                                                            total_length = max_seq_length)[0]
        return encoded_data * mask, context_vector
    

class attention_decoder(torch.nn.Module):
    def __init__(self, hidden_size, dropout=0, output_dim = 3):
        super(attention_decoder, self).__init__()
        self.output_dim = output_dim
        self.attn_layer_weights = torch.nn.Parameter(torch.FloatTensor(
                                                2*hidden_size,
                                                max_seq_length))
        self.attn_layer_bias = torch.nn.Parameter(torch.FloatTensor(
                                                2*hidden_size))
        self.forget_weights = torch.nn.Parameter(torch.FloatTensor(
                                                1, 2*hidden_size + output_dim,
                                                2*hidden_size))
        self.forget_bias = torch.nn.Parameter(torch.FloatTensor(
                                                1,2*hidden_size))
        torch.nn.init.orthogonal_(self.attn_layer_weights.data)
        torch.nn.init.orthogonal_(self.attn_layer_bias.data)
        torch.nn.init.orthogonal_(self.combine_layer_weights.data)
        torch.nn.init.orthogonal_(self.combine_layer_bias.data)
        self.final_layer_weights = torch.nn.Parameter(torch.FloatTensor(
                                                1, 2*hidden_size,output_dim))
        torch.nn.init.orthogonal_(self.final_layer_weights.data)
        self.final_layer_bias = torch.nn.Parameter(torch.FloatTensor(
                                                1,output_dim))
        torch.nn.init.orthogonal_(self.final_layer_bias.data)
        
    def forward(self, src_mask, x_lengths, encoder_output, current_hidden,
                current_device = 'cpu'):
        max_length = torch.max(x_lengths)
        batch_size = x_lengths.shape[0]
        final_output = torch.zeros((encoder_output.shape[0],
                                    max_seq_length, self.output_dim)).to(current_device)
        for i in range(0, int(max_length.item())):
            attn_weights = torch.softmax((torch.matmul(current_cell[0:batch_size,:],
                                                       self.attn_layer_weights) + self.attn_layer_bias), dim=1)
            print(attn_weights.shape)
            modified_input = torch.matmul(attn_weights.unsqueeze(1), encoder_output).squeeze()
            print(modified_input.shape)
            current_hidden = current_hidden * torch.sigmoid(
            final_output[0:batch_size,i,:] = previous_output.squeeze()
            if i >= x_lengths[batch_size-1] and batch_size > 0:
                batch_size -= 1
        return final_output





































def load_file(filename, angles=False):
    seqs, structs, phi, psi = [], [], [], []
    with open(filename) as input_data:
        for line in input_data:
            seqs.append(line.split('\t')[0])
            structs.append(line.split('\t')[1])
            phi.append([float(current) for current in
                        line.split('\t')[2].split(',')])
            psi.append([float(current) for current in
                        line.split('\t')[3].strip().split(',')])
    if angles == True:
        return seqs, structs, phi, psi
    else:
        return seqs, structs

def augment_dataset(seqs, structs):
    expanded_seqset = []
    expanded_structset = []
    for i in range(0, len(seqs)):
        expanded_seqset.append(seqs[i])
        expanded_structset.append(structs[i])
        if i % 2 == 0:
            reverse_seq = [seqs[i][j] for j in range(len(seqs[i])-1, -1, -1)]
            reverse_struct = [structs[i][j] for j in range(len(structs[i])-1, -1, -1)]
            expanded_seqset.append(''.join(reverse_seq))
            expanded_structset.append(''.join(reverse_struct))
    return expanded_seqset, expanded_structset

def calc_accuracy(structs, preds, confmat = False, threeclass = True):
    sec_struct = ['-','H','G','T','S','I','E','B']
    three_class_dict = {'G':'H', 'H':'H', 'I':'H',
                        'E':'S', 'B':'S', '-':'L',
                        'S':'L', 'T':'L'}
    three_class_indices = ['H', 'S', 'L']
    total_preds, correct_8class_preds = 0, 0
    correct_3class_preds = 0
    if confmat:
        confusion_mat_8class = np.zeros((8,8))
        confusion_mat_3class = np.zeros((3,3))
    if len(structs) != len(preds):
        print('Number predictions does not match number of ground truth labels.')
        return
    for i in range(0, len(structs)):
        if len(structs[i]) != len(preds[i]):
            print('Number predictions does not match number of ground truth labels for '
                  'sequence %s'%i)
            return
        for j in range(0, len(structs[i])):
            total_preds += 1
            if threeclass == False:
                three_class_pred = three_class_dict[preds[i][j]]
                three_class_truth = three_class_dict[structs[i][j]]
            else:
                three_class_pred = preds[i][j]
                three_class_truth = structs[i][j]
            if structs[i][j] == preds[i][j]:
                correct_8class_preds += 1
            if three_class_pred == three_class_truth:
                correct_3class_preds += 1
            if confmat:
                if threeclass == False:
                    confusion_mat_8class[sec_struct.index(structs[i][j]),
                                     sec_struct.index(preds[i][j])] += 1
                confusion_mat_3class[three_class_indices.index(three_class_truth),
                                     three_class_indices.index(three_class_pred)] += 1
    if threeclass == False:
        print('8 class accuracy: %s'%(correct_8class_preds / total_preds))
    print('3 class accuracy: %s'%(correct_3class_preds / total_preds))
    if confmat:
        if threeclass == False:
            return confusion_mat_8class, confusion_mat_3class
        else:
            return confusion_mat_3class

def generate_mask(x, dim):
    with torch.no_grad():
        mask = torch.ones((x.shape[0], x.shape[1], dim))
        x_lengths = torch.tensor([torch.nonzero(torch.sum(x[j,:,:],1)).shape[0]
                                      for j in range(0, x.shape[0])])
        for i in range(0, x.shape[0]):
            mask[i, x_lengths[i]:, :] = 0
            
    return mask

def generate_phi_psi(phi, psi):
        phipsi_tensor = torch.zeros((len(phi), 500, 4), dtype=torch.float)
        for i in range(0, len(phi)):
                phipsi_tensor[i, 0:len(phi[i]), 0] = torch.tensor([math.sin(math.radians(angle))
                                                   for angle in phi[i]])
                phipsi_tensor[i, 0:len(phi[i]), 1] = torch.tensor([math.cos(math.radians(angle))
                                                   for angle in phi[i]])
                phipsi_tensor[i, 0:len(psi[i]), 2] = torch.tensor([math.sin(math.radians(angle))
                                                   for angle in psi[i]])
                phipsi_tensor[i, 0:len(psi[i]), 3] = torch.tensor([math.sin(math.radians(angle))
                                                   for angle in psi[i]])
        return phipsi_tensor

def generate_inputs(seqs_to_translate, structs_to_translate = None):
        aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W','Y', 'V']
        sec_struct = ['-','H','G','T','S','I','E','B']
        seq_tensor = torch.zeros((len(seqs_to_translate), 500, 20), dtype=torch.int8)
        struct_tensor = torch.zeros((len(seqs_to_translate), 500, 8),
                                                dtype = torch.long)
        for i in range(0, len(seqs_to_translate)):
            for j in range(0, len(seqs_to_translate[i])):
                seq_tensor[i,j,aas.index(seqs_to_translate[i][j])] = 1
                if structs_to_translate is None:
                    pass
                else:
                    struct_tensor[i,j,sec_struct.index(structs_to_translate[i][j])] = 1
        return seq_tensor, struct_tensor

def generate_3class_inputs(seqs_to_translate, structs_to_translate = None):
        aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
       'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W','Y', 'V']
        threeclass_dict = {'-':0, 'S':0, 'T':0,
                           'E':1, 'B':1,
                           'G':2, 'H':2, 'I':2}
        seq_tensor = torch.zeros((len(seqs_to_translate), 500, 20), dtype=torch.int8)
        struct_tensor = torch.zeros((len(seqs_to_translate), 500, 3),
                                                dtype = torch.long)
        for i in range(0, len(seqs_to_translate)):
            for j in range(0, len(seqs_to_translate[i])):
                seq_tensor[i,j,aas.index(seqs_to_translate[i][j])] = 1
                if structs_to_translate is None:
                    pass
                else:
                    struct_tensor[i,j,
                                  threeclass_dict[structs_to_translate[i][j]]] = 1
        return seq_tensor, struct_tensor

def generate_blosum_inputs(seqs_to_translate, structs_to_translate = None):
        aa_dict = {'A':torch.tensor([4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0]),
            'R':torch.tensor([-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3]),
            'N':torch.tensor([-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3]),
            'D':torch.tensor([-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3]),
            'C':torch.tensor([0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]),
            'Q':torch.tensor([-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2]),
            'E':torch.tensor([-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2]),
            'G':torch.tensor([0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3]),
            'H':torch.tensor([-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3,-1,-2,-1,-2,-1,-2,-2,2,-3]),
            'I':torch.tensor([-1, -3, -3, -3, -1, -3, -3, -4, -3,4,2,-3,1,0,-3,-2,-1,-3,-1,3]),
            'L':torch.tensor([-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1]),
            'K':torch.tensor([-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2]),
            'M':torch.tensor([-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1]),
            'F':torch.tensor([-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1]),
            'P':torch.tensor([-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2]),
            'S':torch.tensor([1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2]),
            'T':torch.tensor([0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0]),
            'W':torch.tensor([-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3]),
            'Y':torch.tensor([-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1]),
            'V':torch.tensor([0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4])  }
        threeclass_dict = {'-':0, 'S':0, 'T':0,
                           'E':1, 'B':1,
                           'G':2, 'H':2, 'I':2}
        seq_tensor = torch.zeros((len(seqs_to_translate), 500, 20), dtype=torch.int8)
        struct_tensor = torch.zeros((len(seqs_to_translate), 500, 3),
                                                dtype = torch.long)
        for i in range(0, len(seqs_to_translate)):
            for j in range(0, len(seqs_to_translate[i])):
                seq_tensor[i,j,:] = aa_dict[seqs_to_translate[i][j]]
                if structs_to_translate is None:
                    pass
                else:
                    struct_tensor[i,j,threeclass_dict[structs_to_translate[i][j]]] = 1
        return seq_tensor, struct_tensor

def convert_8to3(structs):
    output_structs = []
    threeclass_dict = {'-':'L', 'S':'L', 'T':'L',
                           'E':'S', 'B':'S',
                           'G':'H', 'H':'H', 'I':'H'}
    for struct in structs:
        temp_arr = []
        for i in range(0, len(struct)):
            temp_arr.append(threeclass_dict[struct[i]])
        output_structs.append(''.join(temp_arr))
    return output_structs

def decode_output(raw_output, decode_8class = False):
    output_structs = []
    output_arr = []
    sec_struct = ['-','H','G','T','S','I','E','B']
    threeclass_list = ['L', 'S', 'H']    
    for i in range(0, raw_output.shape[0]):
        temp_struct = []
        temp_arr = np.zeros((750))
        for j in range(0, raw_output[i].shape[0]):
            if torch.sum(raw_output[i,j,:]) == 0:
                break
            temp_arr[j] = np.argmax(raw_output[i,j,:]).item()
            if decode_8class == True:
                temp_struct.append(sec_struct[np.argmax(raw_output[i,j,:]).item()])
            elif decode_8class == False:
                temp_struct.append(threeclass_list[np.argmax(raw_output[i,j,:]).item()])
        output_structs.append(''.join(temp_struct))
        output_arr.append(temp_arr)
    return output_structs, output_arr
