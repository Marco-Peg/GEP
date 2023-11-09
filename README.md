# Geometric Paratope Epitope Prediction
Official repository of the paper "Geometric Epitope and Paratope Prediction"

## Environement

To run polymol on docker, first you need to run this line on the terminal

```
apt-get update && apt-get install libgl1
```

## Preprocessing

Run the processe_features to go from the pdb to pdb_l, pdb_r.  
The pdb pdb_l, pdb_r contains the antigene and the antibody respectiivally.

Run the Processing_features.py to go from the pdb to pdb_l, pdb_r and pickle that contains the processed data. You
should only need to modify the path and the name of the csv to the data you are interested in processing. The processed
data contains the

### Antibody (CDR)

- feature_cdr:  A list of size number of cdr containing tensors with size (nbr of ag residue, nbr of features). The
  features are one hot, aa (nbr of features=28).

- coords_cdr : A list of size number of cdr containing the 3D coordinates of the residuals
- centers_cdr: List of the 3D coordinates of the center of the molecules
- atoms_cdr: List of the 3D coordinates of the atom of the cdr region. The atoms are grouped by residuals
- atoms_ab: List of the 3D coordinates af all the atoms in the antibody saved as numpy ndarray of size n_atoms x 3.
- atoms_cdr_distances : List of distance from each cdr atoms to the nearest atom in the ab

- edges_cdr: A list of size number of cdr containing the edges of each cdr in a tensors of size (nbr of edges, 2).

- lenghts_cdr:  A liste of the number of residue in each cdr (nbr of cdr residue,).

- lbls_cdr: Residues in contact with ag (nbr of cdr residue,).

### Antigene (AG)

feature_ag: A list of size number of ag containing tensors with size (nbr of cdr residue, nbr of features). The features
are one hot, aa (nbr of features=28).

- coords_ag : A list of size number of ag containing the 3D coordinates of the residuals
- centers_ag : List of the 3D coordinates of the center of the molecules
- atoms_ag:  List of the 3D coordinates of the atom of the cdr region. The atoms are grouped by residuals
- atoms_ag_distances : List of distance from each ag atoms to the nearest atom in the ab
- edges_ag: A list of size number of ag containing the edges of each ag in a tensors of size (nbr of edges, 2).
- length_ag: A liste of the number of residue in each ag (nbr ag residue,).

- lbls_ag: Residues in contact with cdr (nbr of ag residue,).

### Surface features

To compute the surface and then transfer the features on it run:

```
python pdb2surface.py --pdb-folder --dest-folder
```

For each pdb in the folder, a .wrl will be saved in the destination folder. Then it will save the data in pickels files:

- surfaces_points.p: the 3D coordinates
- surface_faces.p: the triangulation of each surface
- surface_color.p: the color produced by PyMol for each vertex
- surface_normal.p: the normal for each vertex
- surface_feats.p: the features on each vertex
- surface_lbls.p: the ground truth labels for each vertex

## EpiEpmp 
  
This runs only the epitope prediction of the EPMP (https://arxiv.org/pdf/2106.00757.pdf).
Run the run test_file.


## Pi-Diff-Net
    
Run ...
