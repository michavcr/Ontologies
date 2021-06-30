import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np

from utils import timeit
from operator import itemgetter

class MaskedLinear(nn.Module):
    def __init__(self, n_input, n_output, mask, has_bias=False, activation='identity'):
        """
        A version of nn.Linear with a mask to define expertly the connections.

        Parameters
        ----------
        n_input : int
            Number of inputs.
        n_output : int
            Number of outputs.
        mask : bool pytorch Tensor of size (n_input, n_output)
            Mask to specify the connections.
        has_bias : bool, optional
            Whether a bias is used or not. The default is False.
        activation : string, optional
            Activation function to apply, choose between 'sigmoid', 'identity', 
            or 'tanh'. The default is 'identity'.

        Returns
        -------
        None.

        """
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        
        self.W = torch.nn.Parameter(torch.randn(n_input, n_output))
        
        self.var_W = torch.zeros((n_input, n_output))
        
        self.mean_W = self.W.detach()
        
        if has_bias:
            self.b = torch.nn.Parameter(torch.randn(1,n_output))
        else:
            self.b = torch.zeros(1,n_output)
            
        self.mask = mask
        
        activ_dict = {'sigmoid': torch.nn.Sigmoid, 'identity': torch.nn.Identity, 'tanh': torch.nn.Tanh}
        self.activation = activ_dict[activation]()

    def forward(self, X):
        prod = torch.matmul(X, self.W*self.mask) + self.b
        
        return(self.activation(prod))
    
    def update_weight_variance(self, n):
        """
        Update weights variance with a recurrence relation.
        """
        if n <= 1:
            return (self.var_W, self.mean_W)
        
        self.var_W = (n-2)/(n-1) * self.var_W + (self.W.detach() - self.mean_W)**2 / n
        self.mean_W = ((n-1)*self.mean_W + self.W.detach())/n
        
        return(self.mean_W, self.var_W)
    
    def initialize_weight_variance(self):
        """
        Initialize weights variance. Variance to zero and mean to first values.
        """
        self.var_W = torch.zeros((self.n_input, self.n_output))
        
        self.mean_W = self.W.detach()
        
        return(self.mean_W, self.var_W)
    
    def get_VIANN(self):
        """
        Get the VIANN importance score (Genes importance in relation to 
        ontology terms).
        VIANN(g) = Sum_t |w_{g,t}|*(Var(w)_{g,t})
        """
        W = self.W.detach()
        VIANN = torch.sum(abs(W*self.var_W), dim=1)
        
        return(VIANN)
    
    def get_VIANN_2(self):
        """
        Same as VIANN but with ontology terms importance in relation to genes.
        VIANN2(t) = Sum_g |w_{g,t}|*(Var(w)_{g,t})
        """

        W = self.W.detach()
        VIANN = torch.sum(abs(W*self.var_W), dim=0)
        
        return(VIANN)
        
class GeneAutoEncoder(nn.Module):
    def __init__(self, n_genes, n_dense, mask, all_genes, all_go, activation='tanh'):
        """
        Autoencoder with expertly defined connections on the top and the bottom
        of the network. The first/last layer represent genes while the second/
        penultimate layers represent go terms (biological processes, molecular
        functions...)

        Parameters
        ----------
        n_genes : int
            Number of genes as input of the neural network.
        n_dense : int
            Number of neurons on the latent layer.
        mask : bool pytorch Tensor
            Mask to use at the top and the bottom of the network.
        all_genes : list of strings
            List of gene names.
        all_go : list of strings
            List of go term ids.
        activation : string, optional
            Activation function to apply, choose between 'sigmoid', 'identity', 
            or 'tanh'. The default is 'identity'.

        Returns
        -------
        None.

        """
        super().__init__()
        self.all_genes = all_genes
        self.all_go = all_go
        
        self.N0 = mask.size()[0]
        self.mask = torch.cat((mask, torch.ones(self.N0, n_dense)), dim=1)
        self.mask_t = torch.transpose(self.mask, 1, 0)
        
        self.N1 = self.mask.size()[1]
        self.N2 = n_dense
        
        self.encoder = nn.Sequential(MaskedLinear(self.N0, self.N1, self.mask, activation='tanh'), nn.Linear(self.N1, self.N2), nn.Tanh())
        
        self.decoder = nn.Sequential(nn.Linear(self.N2, self.N1), nn.Tanh(), MaskedLinear(self.N1, self.N0, self.mask_t, activation='tanh'))
        
    def forward(self, features):
        encoded = self.encoder(features)
        
        decoded = self.decoder(encoded)
        
        return(decoded)
    
    def get_terms(self, gene_id):
        """Given a gene id, get indices of go terms which are connected to this 
        gene."""
        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        res = [self.all_go[t] for t in term_idx]
        
        return(res)
        
    def get_genes(self, term_id):
        """Given a go term id, get indices of genes which are connected to 
        this go term."""

        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        res = [self.all_genes[g] for g in gene_idx]
        
        return(res)
    
    def get_sorted_genes(self, term_id):
        """Given a go term id, get indices of genes which are connected to 
        this go term, sorted by descending weights."""

        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()

        res = [(self.all_genes[g], W[g, term_id].item()) for g in gene_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
    def get_sorted_terms(self, gene_id):
        """Given a gene id, get indices of go terms which are connected to this 
        gene, sorted by descending weights."""

        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()
                
        res = [(self.all_go[t], W[gene_id, t].item()) for t in term_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
class Baseline(nn.Module):
    def __init__(self, n_genes, n_dense, all_genes, all_go, activation='tanh'):
        """
        Fully connected autoencoder as a Baseline.

        Parameters
        ----------
        n_genes : int
            Number of genes as input of the neural network.
        n_dense : int
            Number of neurons on the latent layer.
        all_genes : list of strings
            List of gene names.
        all_go : list of strings
            List of go term ids.
        activation : string, optional
            Activation function to apply, choose between 'sigmoid', 'identity', 
            or 'tanh'. The default is 'identity'.
        Returns
        -------
        None.

        """

        super().__init__()
        
        self.N0 = n_genes
        self.N1 = len(all_go)
        self.N2 = n_dense
        
        self.encoder = nn.Sequential(nn.Linear(self.N0, self.N1), nn.Tanh(), nn.Linear(self.N1, self.N2), nn.Tanh())
        
        self.decoder = nn.Sequential(nn.Linear(self.N2, self.N1), nn.Tanh(), nn.Linear(self.N1, self.N0), nn.Tanh())
        
    def forward(self, features):
        encoded = self.encoder(features)
        
        decoded = self.decoder(encoded)
        
        return(decoded)
        
class GeneClassifier(nn.Module):
    def __init__(self, n_genes, n_dense, mask, n_classes, all_genes, all_go, activation='tanh', dense=False):
        """
        Classifier with expertly defined connections on the input layer
        of the network. The first layer represent genes while the second
        layer represent gene ontology terms (biological processes, molecular
        functions...)

        Parameters
        ----------
        n_genes : int
            Number of genes as input of the neural network.
        n_dense : int
            Number of neurons on the latent layer.
        mask : bool pytorch Tensor
            Mask to use at the top and the bottom of the network.
        n_classes : int
            Number of classes (define the number of neurons of the last layer).
        all_genes : list of strings
            List of gene names.
        all_go : list of strings
            List of go term ids.
        activation : string, optional
            Activation function to apply, choose between 'sigmoid', 'identity', 
            or 'tanh'. The default is 'identity'.
        dense : bool, optional
            Whether to add a fully connected part on the latent layer or not.
        Returns
        -------
        None.

        """

        super().__init__()
        self.all_genes = all_genes
        self.all_go = all_go
        
        self.N0 = mask.size()[0]
        
        if dense:
            self.mask = torch.cat((mask, torch.ones(self.N0, n_dense)), dim=1)
        else:
            self.mask = mask
            
        self.mask_t = torch.transpose(self.mask, 1, 0)
        
        self.N1 = self.mask.size()[1]
        self.N2 = n_dense
        self.Nc = n_classes
        
        self.encoder = nn.Sequential(MaskedLinear(self.N0, self.N1, self.mask, activation='tanh'), nn.Linear(self.N1, self.N2), nn.Tanh())
        
        self.clf = nn.Sequential(nn.Linear(self.N2, self.Nc), nn.Softmax(dim=1))

        self.var_S = torch.zeros((self.N1, self.Nc))
        
        W2 = torch.transpose(next(self.encoder[1].parameters()), 0, 1)
        W3 = torch.transpose(next(self.clf[0].parameters()), 0, 1)
        self.mean_S = torch.matmul(W2, W3).detach()

    def forward(self, features):
        encoded = self.encoder(features)
        
        res = self.clf(encoded)
        
        return(res)
    
    def get_terms(self, gene_id):
        """Given a gene id, get indices of go terms which are connected to this 
        gene."""

        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        res = [self.all_go[t] for t in term_idx]
        
        return(res)
    
    def get_genes(self, term_id):
        """Given a go term id, get indices of genes which are connected to 
        this go term."""

        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        res = [self.all_genes[g] for g in gene_idx]
        
        return(res)
    
    def get_sorted_genes(self, term_id):
        """Given a go term id, get indices of genes which are connected to 
        this go term, sorted by descending weights."""

        gene_idx = torch.nonzero(self.mask[:, term_id], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()

        res = [(self.all_genes[g], W[g, term_id].item()) for g in gene_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
    def get_sorted_terms(self, gene_id):
        """Given a gene id, get indices of go terms which are connected to this 
        gene, sorted by descending weights."""

        term_idx = torch.nonzero(self.mask[gene_id, :], as_tuple=True)[0]
        
        W = self.encoder[0].W.detach()
                
        res = [(self.all_go[t], W[gene_id, t].item()) for t in term_idx]
        
        return(sorted(res, key=itemgetter(1), reverse=True))
    
    def get_current_score(self):
        """
        Get current score W2*W3.
        """
        W2 = torch.transpose(next(self.encoder[1].parameters()), 0, 1)
        W3 = torch.transpose(next(self.clf[0].parameters()), 0, 1)

        return(torch.matmul(W2, W3).detach())
    
    def update_score_variance(self, n):
        """
        Update score (W2*W3) variance.
        """
        if n <= 1:
            return (self.mean_S, self.var_S)
        
        S = self.get_current_score()
        
        self.var_S = (n-2)/(n-1) * self.var_S + (S - self.mean_S)**2 / n
        self.mean_S = ((n-1)*self.mean_S + S)/n
        
        return(self.mean_S, self.var_S)
    
    def initialize_score_variance(self):
        """
        Initialize score (W2*W3) variance.

        """
        self.var_S = torch.zeros((self.N1, self.Nc))
        
        self.mean_S = self.get_current_score()
        
        return(self.mean_S, self.var_S)

class PartialClassifier(nn.Module):
    """
    Classifier architecture without the first layer (the first hidden layer
    becomes the input layer).
    """
    def __init__(self, clf):
        super().__init__()

        self.clf = clf
    
    def forward(self, features):
        encoded = self.clf.encoder[1:](features)
        res = self.clf.clf(encoded)
        
        return(res)
    
@timeit
def ae_pipeline(mask, data_numpy, all_genes, all_go, n_epochs=10, batch_size=50, print_loss=100, output_file='model.pth', embed_file='embeddings_ae.csv'):
    """
    Training pipeline for the GeneAutoEncoder.
    Includes:
        - Normalisation, data split and data loading.
        - Definition of the model.
        - Training.
        - Accuracy and loss computing (at each epoch).
        - Evaluation of the model on a test set.
        - Model saving, embeddings saving.

    Parameters
    ----------
    mask : bool Tensor
        A mask defining the genes - go terms connections in the early layers.
    data_numpy : float numpy ndarray of shape (n_samples, n_features)
        The data (train and test sets in a single array)
    all_genes : list of strings 
        List of gene symbols.
    all_go : list of strings
        List of go terms identifiers.
    n_epochs : int, optional
        Number of epochs to run. The default is 10.
    batch_size : int, optional
        Number of batch to batch the data. The default is 50.
    dense : int, optional
        Adding a fully-connected part (100 neurons) on the second layer? The 
        default is False.
    output_file : string, optional
        Filepath for the model output file. The default is 'model_ae.pth'.
    embed_file : string, optional
        Filepath for the embeddings output file. The default is 'embeddings_ae.csv'.

    Returns
    -------
    clf : GeneClassifier instance.
        The trained classifier.
    train_loader : DataLoader instance
        The data loader containing the batched data.
    embeddings : float Tensor of size (n_samples, size_embedding)
        The embeddings.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_genes = data_numpy.shape[1]
    
    size_encoded=100 
    
    ae = GeneAutoEncoder(N_genes, size_encoded, mask, all_genes, all_go, activation='tanh').to(device)
    summary(ae, (1,N_genes))
    
    N = data_numpy.shape[0]

    data_numpy = min_max_normalisation(std_normalisation(data_numpy))
    data_tensor = torch.Tensor(data_numpy)
    train = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(ae.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()
    
    # Initialize MaskedLinear weights variance
    _,_ = ae.encoder[0].initialize_weight_variance()
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        
        
        # Update MaskedLinear weights variance
        _,_ = ae.encoder[0].update_weight_variance(epoch+1)
        
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
            
        print('[%d, %5d] loss: %.5f' %
                             (epoch + 1, i + 1, running_loss / (i+1)))

    print('Finished Training')
    torch.save(ae, output_file)

    embeddings = get_embeddings(train_loader, N, ae, size_encoded=size_encoded)
    
    np.savetxt(embed_file, embeddings, delimiter=',')

    return (ae, train_loader, embeddings)

@timeit
def clf_pipeline(mask, data_numpy, targets, all_genes, all_go, n_epochs=10, batch_size=50, dense=False, output_file='model_clf.pth', embed_file='embeddings_clf.csv'):
    """
    Training pipeline for the GeneClassifier.
    Includes:
        - Normalisation, data split and data loading.
        - Definition of the model.
        - Training.
        - Accuracy and loss computing (at each epoch).
        - Evaluation of the model on a test set.
        - Model saving, embeddings saving.

    Parameters
    ----------
    mask : bool Tensor
        A mask defining the genes - go terms connections in the early layers.
    data_numpy : float numpy ndarray of shape (n_samples, n_features)
        The data (train and test sets in a single array)
    targets : int numpy ndarray of shape (n_samples,)
        The targets for each sample.
    all_genes : list of strings 
        List of gene symbols.
    all_go : list of strings
        List of go terms identifiers.
    n_epochs : int, optional
        Number of epochs to run. The default is 10.
    batch_size : int, optional
        Number of batch to batch the data. The default is 50.
    dense : int, optional
        Adding a fully-connected part (100 neurons) on the second layer? The 
        default is False.
    output_file : string, optional
        Filepath for the model output file. The default is 'model.pth'.
    embed_file : string, optional
        Filepath for the embeddings output file. The default is 'embeddings_clf.csv'.

    Returns
    -------
    clf : GeneClassifier instance.
        The trained classifier.
    train_loader : DataLoader instance
        The data loader containing the batched data.
    embeddings : float Tensor of size (n_samples, size_embedding)
        The embeddings.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_genes = data_numpy.shape[1]
    N_classes = targets.max()+1
    size_encoded=100
    
    clf = GeneClassifier(N_genes, size_encoded, mask, N_classes, all_genes, all_go, activation='tanh', dense=dense).to(device)
    summary(clf, (1,N_genes))
    N = data_numpy.shape[0]
    
    data_numpy = std_normalisation(data_numpy)
    data_tensor = torch.Tensor(data_numpy)
    targets_tensor = torch.Tensor(targets).long()
    dataset = torch.utils.data.TensorDataset(data_tensor, targets_tensor)
    
    train_size = int(0.8*N)
    val_size = N-train_size
    
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(clf.parameters(), lr=1e-4)

    # cross-entropy error loss
    criterion = nn.CrossEntropyLoss()
    
    _,_ = clf.initialize_score_variance()
    _,_ = clf.encoder[0].initialize_weight_variance()
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0.0
        total = 0 
        
        _,_ = clf.update_score_variance(epoch+1)
        _,_ = clf.encoder[0].update_weight_variance(epoch+1)
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs.to(device)
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
            
        print('[%d, %5d] loss: %.5f, accuracy: %.5f' %
                              (epoch + 1, i + 1, running_loss / (i+1), correct / total))
    
    print('Finished Training')
    
    print('Computing accuracy on val set')
    running_loss = 0.0
    correct = 0.0
    total = 0 

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
            
        inputs.to(device)
        labels.to(device)
        
        outputs = clf(inputs)
        
        loss = criterion(outputs, labels)

        predict = outputs.argmax(1)
        
        correct += (predict == labels).sum().item()
        total += labels.shape[0]
        running_loss += loss.item()
        
    print('loss: %.5f, accuracy: %.5f' %
                              (running_loss / (i+1), correct / total))

    torch.save(clf, output_file)
    
    embeddings = get_embeddings(train_loader, N, clf, size_encoded=size_encoded)
    
    np.savetxt(embed_file, embeddings, delimiter=',')
    
    return (clf, train_loader, embeddings)

def get_train_loader(data_numpy, targets, batch_size=50, shuffle=False):
    """
    Given a dataset (samples and targets), returns a pytorch DataLoader.

    Parameters
    ----------
    data_numpy : float numpy ndarray of shape (n_samples, n_features)
        The dataset.
    targets : int numpy ndarray of shape (n_samples,)
        Targets for each sample in the dataset.
    batch_size : int, optional
        Number of samples in a sigle batch, to batch the data. The default 
        is 50.
    shuffle : bool, optional
        Whether to shuffle or not the samples. The default is False.
    Returns
    -------
    The TensorDataset containing the data, and the batched DataLoader.

    """
    data_numpy = std_normalisation(data_numpy)
    data_tensor = torch.Tensor(data_numpy)
    targets_tensor = torch.Tensor(targets).long()
    train = torch.utils.data.TensorDataset(data_tensor, targets_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    
    return(train, train_loader)

def get_embeddings(train_loader, N_samples, model, size_encoded=100):
    """
    Given a DataLoader and a GeneClassifier or a GeneAutoEncoder (the 'model'),
    returns the embeddings in the latent space.

    Parameters
    ----------
    train_loader : DataLoader instance
        The train loader containing the batched samples.
    N_samples : int
        Number of samples.
    model : TYPE
        The model (either GeneClassifier or GeneAutoEncoder) giving the 
        embeddings.
    size_encoded : int, optional
        The embeddings size. The default is 100.

    Returns
    -------
    A numpy ndarray of size (N_samples, size_encoded) containing the embeddings.

    """
    trainiter = iter(train_loader)
    embeddings = np.zeros((N_samples, size_encoded))
    
    for i, data in enumerate(trainiter):
        inputs, _ = data
        embedded = model.encoder(inputs).cpu().detach().numpy()
        batch_size = embedded.shape[0]
        embeddings[batch_size*i:batch_size*(i+1),:] = embedded
    
    return(embeddings)

def std_normalisation(matrix, e=1e-8):
    """
    Standard normalisation of a matrix (mean = 0, std = 1) along axis 0.

    Parameters
    ----------
    matrix : a (int or float or bool or double) numpy matrix
        The matrix to normalise (along its axis 0).
    e : float, optional
        Constant to avoid division by zero. The default is 1e-8.

    Returns
    -------
    The normalised matrix.

    """
    m = np.mean(matrix, axis=0)
    s = np.std(matrix, axis=0)
    r = (matrix - m)/(s+e)
    
    return(r)

def min_max_normalisation(matrix, a=-1, b=1, e=1e-8):
    """
    Min-max normalisation of a matrix (put the min to a, and max to b) along 
    axis 0.

    Parameters
    ----------
    matrix : a (int or float or bool or double) numpy matrix
        The matrix to normalise (along its axis 0).
    a : int or float, optional
        The value to which we want to put a column's min. The default is -1.
    b : int or float, optional
        The value to which we want to put a column's max. The default is 1.
    e : float, optional
        Constant to avoid division by zero. The default is 1e-8.

    Returns
    -------
    The normalised matrix.

    """

    M = np.max(matrix, axis=0)
    m = np.min(matrix, axis=0)
    r = (b-a) * (matrix-m) / (M-m+e) + a
    
    return(r) 