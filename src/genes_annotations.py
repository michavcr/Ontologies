from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import  GODag
import torch

import numpy as np
import h5py
import pandas as pd
from utils import timeit

def read_h5(filename):
    h5 = h5py.File(filename,'r')

    return(h5)

def get_all_genes(h5):
    s = []
    genes = np.unique(np.array(h5['matrix']['features']['name']))

    for g in genes:
        s.append(g.decode('UTF-8'))
     
    return(s)

def select_terms(godag, namespaces, levels, take_obsolete):
    selected = {}
    
    for k, term in godag.items():
        if term.level in levels and (take_obsolete or not term.is_obsolete) and term.namespace in namespaces:
            selected[k]=term
    
    return(selected)

def get_all_gene_annotations(h5, levels=[3], kinds=['biological_process', 'molecular_function']):
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

##TODO : faire une fonction pour retrouver le nom du type de cellule
# reentrainer sans les molécular functions et avec niveau 4 ou 5
#* existe il des outils pour visualiser les poids des réseaux de neurones ?
##* demander à fabien ou yufei un jeu de données qu'ils connaissent bien
#* mettre le dernier code pour afficher les n plus grands termes dans une fonction

class GOTerm:
    def __init__(self):
        self.ogaf = GafReader("../goa_human.gaf")
        self.godag = GODag("../go.obo", optional_attrs={'consider', 'replaced_by'}, load_obsolete=True)
     
    def get_go_name(self, go_id):
        return(self.godag[go_id].name)

def largest_terms(df, cell, goterm, all_go, n=10):
    ix = df[0].nlargest(10).index
    t = [all_go[i] for i in ix]
    
    res=[]
    
    for p in t:
        res.append(goterm.get_go_name(p))
    
    return(res)

def build_mask(h5, genes_go, all_go, all_genes):
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
def select_gene_expr(h5, expr_mat, selected_genes):
    selected_genes_index = []
    
    a=np.array(h5['matrix']['features']['name'])

    for g in selected_genes:
        i = np.where(a == g.encode('utf-8'))[0]

        if i.shape[0]>0:
            selected_genes_index.append(i[0])

    expr_mat = expr_mat[:, selected_genes_index]

    return(expr_mat)

@timeit
def select_gene_expr_v2(h5, expr_mat, selected_genes):
    selected_genes_index = []
    
    a=np.array(h5['matrix']['features']['name'])
    
    selected_genes = np.array(list(map(lambda x: x.encode('utf-8'), selected_genes)))

    selected_genes_index = np.where(np.in1d(a, selected_genes))
    
    expr_mat = expr_mat[:, selected_genes_index[0]]

    return(expr_mat)

def get_cells_by_patient(filename, patient, df=None):
    if df is None:
        df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[df['Patient']==patient].index))

def get_cells_by_cluster(filename, cluster, df=None):
    if df is None:
        df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[df['Cluster']==cluster].index))

def get_cells_by_mask(filename, mask):
    df = pd.read_csv(filename, delimiter='\t')
    return(np.array(df[mask].index))

def group_cells_by_patient(filename):
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