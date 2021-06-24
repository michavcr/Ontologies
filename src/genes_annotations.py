from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import  GODag
import torch

import numpy as np
import h5py
import pandas as pd
from utils import timeit

def read_h5(filename):
    """
    Reading a h5 file.

    Parameters
    ----------
    filename : string
        Filepath of the h5.

    Returns
    -------
    The h5's content.

    """
    h5 = h5py.File(filename,'r')

    return(h5)

def get_all_genes(h5):
    """
    Listing the genes involved in h5 object.

    Parameters
    ----------
    h5 : h5py File object
        The h5 object to consider.

    Returns
    -------
    The list of genes.

    """
    s = []
    genes = np.unique(np.array(h5['matrix']['features']['name']))

    for g in genes:
        s.append(g.decode('UTF-8'))
     
    return(s)

def select_terms(godag, namespaces, levels, take_obsolete):
    """
    Select goterms in the right namespaces, levels, and whether they can be 
    obsolete or not.

    Parameters
    ----------
    godag : GODag Object 
        GODag object containing the ontology.
    namespaces : list of strings
        Namespaces in which go terms can be (either 'biological_process', 
        'molecular_function', or 'cellular_compenent').
    levels : list of int
        The list go terms levels to select. 
    take_obsolete : bool
        Whether or not obsolete terms are accepted.

    Returns
    -------
    A dictionnary of GO terms. 

    """
    selected = {}
    
    for k, term in godag.items():
        if term.level in levels and (take_obsolete or not term.is_obsolete) and term.namespace in namespaces:
            selected[k]=term
    
    return(selected)

def get_all_gene_annotations(h5, levels=[3], kinds=['biological_process', 'molecular_function']):
    """
    Get annotated genes and related go terms responding to some criterias 
    (about the level in the ontology graphs, the namespaces it must be in).

    Parameters
    ----------
    h5 : h5py File object
        The h5 object to consider.
    levels : list of int, optional
        List of wanted levels for the go terms to select. The default is [3].
    kinds : list of strings, optional
        List of wanted kinds (or namespaces) for the go terms to select. The 
        default is ['biological_process', 'molecular_function'].

    Returns
    -------
    goterm : GoTerm object
        GoTerm object (facilitates access to id and GO name).
    genes_go : dict
        Dictionnary containing Gene - GO term relationships;
        keys are genes;
        values are sets of GO id.
    all_go : list of strings
        List of selected go terms.
    all_genes : list of strings
        List of selected genes.

    """
    goterm = GOTerm()
    
    selected_terms = select_terms(goterm.godag, kinds, levels, False)
    
    genes = get_all_genes(h5)

    genes_go = dict()
    all_go = set()
    all_genes = set()

    for a in goterm.ogaf.get_associations():
        if a.DB_Symbol in genes and a.GO_ID in selected_terms.keys():
            if a.DB_Symbol not in genes_go:
                genes_go[a.DB_Symbol] = {a.GO_ID}
            else:
                genes_go[a.DB_Symbol].add(a.GO_ID)
    
            all_go.add(a.GO_ID)
            all_genes.add(a.DB_Symbol)
    
    all_go = sorted(list(all_go))
    all_genes = sorted(list(all_genes))

    return (goterm, genes_go, all_go, all_genes)

class GOTerm:
    def __init__(self):
        self.ogaf = GafReader("../goa_human.gaf")
        self.godag = GODag("../go.obo", optional_attrs={'consider', 'replaced_by'}, load_obsolete=True)
     
    def get_go_name(self, go_id):
        return(self.godag[go_id].name)

def largest_terms(df, goterm, all_go, n=10):
    """
    

    Parameters
    ----------
    df : pandas DataFrame 
        A DataFrame containing GoTerm scores in the first column.
    goterm : GoTerm object.
        GoTerm object. Just run `goterm = GoTerm()`.
    all_go : list of strings name
        List of GO terms name.
    n : TYPE, optional
        The number of largest values to select. The default is 10.

    Returns
    -------
    A list of names of the go term with n largest scores in df.

    """
    ix = df[0].nlargest(n).index
    t = [all_go[i] for i in ix]
    
    res=[]
    
    for p in t:
        res.append(goterm.get_go_name(p))
    
    return(res)

def build_mask(h5, genes_go, all_go, all_genes):
    """
    Build a (n_genes, n_go_terms) mask.

    Parameters
    ----------
    h5 : h4py File object
        The h5 object to consider.
    genes_go : dict
        Dictionnary containing Gene - GO term relationships;
        keys are genes;
        values are sets of GO id.
    all_go : list of strings
        List of go term names.
    all_genes : list of strings
        List of gene names.

    Returns
    -------
    The mask.

    """
    genes_to_index = {k:v for v,k in enumerate(all_genes)}
    go_to_index = {k:v for v,k in enumerate(all_go)}
    
    N_go = len(all_go)
    N_genes = len(all_genes)

    mask = torch.zeros(N_genes, N_go, dtype=bool)

    for gene, terms in genes_go.items():
        for t in terms:
            mask[genes_to_index[gene], go_to_index[t]] = True

    return(mask)

@timeit
def build_expr_mat(h5):
    """
    Build the expression matrix.

    Parameters
    ----------
    h5 : h4py File object.
        The h5 object to consider.

    Returns
    -------
    expr_mat : float numpy ndarray
        Returns the expression matrix.

    """
    indcell = np.array(h5['matrix']['indptr'])
    N_cells = indcell.shape[0]-1
    N_genes = np.unique(np.array(h5['matrix']['features']['name'])).shape[0]

    expr_mat = np.zeros((N_cells, N_genes), dtype=np.float32)
    data = np.array(h5['matrix']['data'], dtype=np.float32)
    indgenes = np.array(h5['matrix']['indices'])

    for i in range(N_cells):
        expr_mat[i,indgenes[indcell[i]:indcell[i+1]]] = data[indcell[i]:indcell[i+1]]
    
    return expr_mat

@timeit
def build_expr_mat_torch(h5):
    """
    Build the expression matrix.

    Parameters
    ----------
    h5 : h4py File object.
        The h5 object to consider (containing the data).

    Returns
    -------
    expr_mat : float pytorch Tensor
        Returns the expression matrix.

    """

    indcell = torch.Tensor(h5['matrix']['indptr']).long()
    N_cells = indcell.shape[0]-1
    N_genes = np.unique(np.array(h5['matrix']['features']['name'])).shape[0]

    expr_mat = torch.zeros((N_cells, N_genes))
    data = torch.Tensor(np.array(h5['matrix']['data']))
    indgenes = torch.Tensor(np.array(h5['matrix']['indices'])).long()

    for i in range(N_cells):
        expr_mat[i,indgenes[indcell[i]:indcell[i+1]]] = data[indcell[i]:indcell[i+1]]
    
    return expr_mat

@timeit
def select_gene_expr_old(h5, expr_mat, selected_genes):
    """
    <<!>> Use select_gene_expr instead.
    
    Select the columns in the expression matrix that corresponds to the 
    selected genes.

    Parameters
    ----------
    h5 : h4py File object.
        The h5 object to consider (containing the data).
    expr_mat : float numpy ndarray (or pytorch Tensor)
        The gene expression matrix.
    selected_genes : list of (binary) strings 
        The list of selected genes.

    Returns
    -------
    The new expression matrix with only the selected genes' columns.

    """
    selected_genes_index = []
    
    a=np.array(h5['matrix']['features']['name'])

    for g in selected_genes:
        i = np.where(a == g.encode('utf-8'))[0]

        if i.shape[0]>0:
            selected_genes_index.append(i[0])

    expr_mat = expr_mat[:, selected_genes_index]

    return(expr_mat)

@timeit
def select_gene_expr(h5, expr_mat, selected_genes):
    """
    <<!>> Use select_gene_expr_v2 instead.
    
    Select the columns in the expression matrix that corresponds to the 
    selected genes.

    Parameters
    ----------
    h5 : h4py File object.
        The h5 object to consider (containing the data).
    expr_mat : float numpy ndarray (or pytorch Tensor)
        The gene expression matrix.
    selected_genes : list of (binary) strings 
        The list of selected genes.

    Returns
    -------
    The new expression matrix with only the selected genes' columns.

    """

    selected_genes_index = []
    
    a=np.array(h5['matrix']['features']['name'])
    
    selected_genes = np.array(list(map(lambda x: x.encode('utf-8'), selected_genes)))

    selected_genes_index = np.where(np.in1d(a, selected_genes))
    
    expr_mat = expr_mat[:, selected_genes_index[0]]

    return(expr_mat)

def get_cells_by_patient(filename, patient, df=None):
    """
    Select cells taken from a given patient.

    Parameters
    ----------
    filename : string
        Path to a csv containing cells metadata (should have a column
                                                   'Patient').
    patient : generic (most likely a string)
        Patient id.
    df : panda DataFrame, optional
        Contains the metadata. The default is None.

    Returns
    -------
    A numpy ndarray containing the index of a given patient's cells.

    """
    if df is None:
        df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[df['Patient']==patient].index))

def get_cells_by_cluster(filename, cluster, df=None):
    """
    Select cells taken in a given cluster.    

    Parameters
    ----------
    filename : string
        Path to a csv containing cells metadata (should have a column
    cluster : generic (most likely a string or an int)
        Cluster id.
    df : panda DataFrame, optional
        Contains the metadata. The default is None.

    Returns
    -------
    A numpy ndarray containing the index of a given cluster's cells.

    """
    if df is None:
        df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[df['Cluster']==cluster].index))

def get_cells_by_mask(filename, mask):
    """
    Select cells with a mask (bool array).    

    Parameters
    ----------
    filename : string
        Path to a csv containing cells metadata (should have a column
    mask : a bool numpy ndarray
        A mask to use to select given cells in a csv.

    Returns
    -------
    A numpy ndarray containing the index of cells selected by the mask.

    """
    df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[mask].index))

def group_cells_by_patient(filename):
    """
    Group cells by the patient they were taken from.

    Parameters
    ----------
    filename : string
        Path to a csv containing cells metadata (should have a column

    Returns
    -------
    A dictionnary with patient ids as keys and matching cells list of indices 
    as values.

    """
    df = pd.read_csv(filename, delimiter='\t')
    patients = df['Patient'].unique()
    d = dict()

    for p in patients:
        cells = get_cells_by_patient(filename, p, df=df)
        d[p] = cells
    
    return(d)

class GeneExpr:
    def __init__(self, h5):
        self.h5 = h5
        self.expr_mat = build_expr_mat(h5)
        self.name_genes = np.array(h5['matrix']['features']['name'])
        self.barcodes = np.array(h5['matrix']['barcodes'])

    def get_gene(self, j=0, name=None):
        if name is None:
            return self.expr_mat[:,j]
        else:
            encoded = name.encode('utf-8')
            try:
                k = np.where(self.name_genes == encoded)[0][0]
            except:
                print(name, "not a name.")

            return(self.expr_mat[:, k])
    
    def get_matrix(self):
        return(self.expr_mat)

    def get_cell(self, i=0, bcode=None):
        if bcode is None:
            return self.expr_mat[i,:]
        else:
            encoded = bcode.encode('utf-8')
            try:
                k = np.where(self.barcodes == encoded)[0][0]
            except:
            	print(bcode, "not a barcode.")

            return(self.expr_mat[k,:])
    
    def get_patient(self):
    	pass