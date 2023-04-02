The aim of this project is to use interpretable neural network architectures to segment single cell RNA-seq data. 

# Gene ontologies for interpretable neural networks
Our way to design interpretable neural networks lies in the use of genomic ontology terms (see the [**G**ene **O**ntology](http://geneontology.org/) project) to expertly assign biological meaning to the neural network activations on its first hidden layer. This type of architectures has already been proposed in [1].

## Biologically interpretable neural networks
Whether we want to segment our data in a supervised or unsupervised way, two types of architectures can be considered: a MLP classifier with a final softmax activation or an autoencoder to do dimension reduction. In both tasks, the way to make these neural networks interpretable is the same: define expertly the relations between the input layer (genomic features) and the first hidden layer (GO terms). The architectures of these neural networks are then no longer dense.

# References
[1] Peng, J., Wang, X. Shang, X. Combining gene ontology with deep
neural networks to enhance the clustering of single cell RNA-Seq data.
BMC Bioinformatics 20, 284 (2019). https://doi.org/10.1186/s12859-
019-2769-6
