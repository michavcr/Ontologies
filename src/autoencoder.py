import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np

from utils import timeit
from operator import itemgetter

class MaskedLinear(nn.Module):
    def __init__(self, n_input, n_output, mask, has_bias=False, activation='identity'):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        
        self.W = torch.nn.Parameter(torch.randn(n_input, n_output))
        self.mask = mask
        
        activ_dict = {'sigmoid': torch.nn.Sigmoid, 'identity': torch.nn.Identity, 'tanh': torch.nn.Tanh}
        self.activation = activ_dict[activation]()

    def forward(self, X):
        prod = torch.matmul(X, self.W*self.mask)
        #prod = self.W*self.mask
        
        return(self.activation(prod))

class GeneAutoEncoder(nn.Module):
    def __init__(self, n_genes, n_dense, mask, all_genes, all_go, activation='tanh'):
        super().__init__()
        self.all_genes = all_genes
        self.all_go = all_go
        
        self.mask = mask
        self.mask_t = torch.transpose(mask, 1, 0)
        
        self.N0 = mask.size()[0]
        self.N1 = mask.size()[1]
        self.N2 = n_dense
        
        self.encoder = nn.Sequential(MaskedLinear(self.N0, self.N1, self.mask, activation='tanh'), nn.Linear(self.N1, self.N2), nn.Tanh())
        
        self.decoder = nn.Sequential(nn.Linear(self.N2, self.N1), nn.Tanh(), MaskedLinear(self.N1, self.N0, self.mask_t, activation='tanh'))
        
    def forward(self, features):
        encoded = self.encoder(features)
        
        decoded = self.decoder(encoded)
        
        return(decoded)
    
    def get_terms(self, gene_id):
        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        res = [self.all_go[t] for t in term_idx]
        
        return(res)
        
    def get_genes(self, term_id):
        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        res = [self.all_genes[g] for g in gene_idx]
        
        return(res)
    
    def get_sorted_genes(self, term_id):
        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()

        res = [(self.all_genes[g], W[g, term_id].item()) for g in gene_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
    def get_sorted_terms(self, gene_id):
        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()
                
        res = [(self.all_go[t], W[gene_id, t].item()) for t in term_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
class GeneClassifier(nn.Module):
    def __init__(self, n_genes, n_dense, mask, n_classes, all_genes, all_go, activation='tanh'):
        super().__init__()
        self.all_genes = all_genes
        self.all_go = all_go
        
        self.mask = mask
        self.mask_t = torch.transpose(mask, 1, 0)
        
        self.N0 = mask.size()[0]
        self.N1 = mask.size()[1]
        self.N2 = n_dense
        self.Nc = n_classes
        
        self.encoder = nn.Sequential(MaskedLinear(self.N0, self.N1, self.mask, activation='tanh'), nn.Linear(self.N1, self.N2), nn.Tanh())
        
        self.clf = nn.Sequential(nn.Linear(self.N2, self.Nc), nn.Softmax(dim=1))
        
    def forward(self, features):
        encoded = self.encoder(features)
        
        res = self.clf(encoded)
        
        return(res)
    
    def get_terms(self, gene_id):
        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        res = [self.all_go[t] for t in term_idx]
        
        return(res)
    
    def get_genes(self, term_id):
        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        res = [self.all_genes[g] for g in gene_idx]
        
        return(res)
    
    def get_sorted_genes(self, term_id):
        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()

        res = [(self.all_genes[g], W[g, term_id].item()) for g in gene_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
    def get_sorted_terms(self, gene_id):
        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()
                
        res = [(self.all_go[t], W[gene_id, t].item()) for t in term_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))


@timeit
def ae_pipeline(mask, data_numpy, all_genes, all_go, n_epochs=10, batch_size=50, print_loss=100, output_file='model.pth', embed_file='embeddings_ae.csv'):
    N_genes = data_numpy.shape[1]
    
    ae = GeneAutoEncoder(N_genes, 500, mask, all_genes, all_go, activation='tanh')
    summary(ae, (1,N_genes))
    
    N = data_numpy.shape[0]

    data_numpy = min_max_normalisation(std_normalisation(data_numpy))
    data_tensor = torch.Tensor(data_numpy)
    train = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(ae.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            
            inputs.to(device)
      
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ae(inputs)
        
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i % print_loss == 0:
                print('[%d, %5d] loss: %.5f' %
                              (epoch + 1, i + 1, running_loss / (i+1)))

    print('Finished Training')
    torch.save(ae, output_file)

    embeddings = get_embeddings(train_loader, N, ae, size_encoded=500)
    
    np.savetxt(embed_file, embeddings, delimiter=',')

    return (ae, train_loader, embeddings)

@timeit
def clf_pipeline(mask, data_numpy, targets, n_epochs=10, batch_size=50, print_loss=100, output_file='model.pth', embed_file='embeddings_clf.csv'):
    N_genes = data_numpy.shape[1]
    N_classes = targets.max()+1
    
    clf = GeneClassifier(N_genes, 500, mask, N_classes, activation='tanh')
    summary(clf, (1,N_genes))
    N = data_numpy.shape[0]
    
    data_numpy = std_normalisation(data_numpy)
    data_tensor = torch.Tensor(data_numpy)
    targets_tensor = torch.Tensor(targets).long()
    train = torch.utils.data.TensorDataset(data_tensor, targets_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(clf.parameters(), lr=1e-4)

    # cross-entropy error loss
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0
        total = 0 
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs.to(device)
      
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = clf(inputs)
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            predict = outputs.argmax(1)
            correct += (predict == labels).sum().item()
            total += labels.shape[0]
            
            # print statistics
            running_loss += loss.item()
            
            if i % print_loss == 0:
                print('[%d, %5d] loss: %.5f, accuracy: %.5f' %
                              (epoch + 1, i + 1, running_loss / (i+1), correct / total))
    
    print('Finished Training')
    torch.save(clf, output_file)
    
    embeddings = get_embeddings(train_loader, N, clf, size_encoded=500)
    
    np.savetxt(embed_file, embeddings, delimiter=',')
    
    return (clf, train_loader, embeddings)

def get_embeddings(train_loader, N, model, size_encoded=100):
    trainiter = iter(train_loader)
    embeddings = np.zeros((N, size_encoded))
    
    for i, data in enumerate(trainiter):
        inputs, _ = data
        embedded = model.encoder(inputs).cpu().detach().numpy()
        batch_size = embedded.shape[0]
        embeddings[batch_size*i:batch_size*(i+1),:] = embedded
    
    return(embeddings)

def std_normalisation(matrix, e=1e-8):
    m = np.mean(matrix, axis=0)
    s = np.std(matrix, axis=0)
    r = (matrix - m)/(s+e)
    
    return(r)

def min_max_normalisation(matrix, a=-1, b=1, e=1e-8):
    M = np.max(matrix, axis=0)
    m = np.min(matrix, axis=0)
    r = (b-a) * (matrix-m) / (M-m+e) + a
    return(r) 