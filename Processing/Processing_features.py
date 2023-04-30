"""
Running cross-validation, processing results
"""
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch

torch.set_printoptions(threshold=50000)
from Bio.PDB import Polypeptide
from os.path import exists, isfile
import pickle
from sklearn.neighbors import NearestNeighbors

from Processing.parsing import *
from Processing.search import *


NUM_FEATURES = len(aa_s) + 7  # one-hot + extra features + chain one-hot
AG_NUM_FEATURES = len(aa_s) + 7


# example on how to open the processed data
def open_dataset(dataset_cache="processed-dataset.p"):
    if exists(dataset_cache) and isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        raise Exception(f"No dataset cache found : {dataset_cache} ")
    return dataset

def create_dataset(dataset_csv):
    summary_file = pd.read_csv(dataset_csv)
    cache_file = os.path.join(os.path.split(dataset_csv)[0], "processed-dataset.p")
    print("Computing and storing the dataset...")
    dataset = process_dataset(summary_file,path=os.path.split(dataset_csv)[0])
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f, protocol=2)

    return dataset

def get_pdb_structure_seperate(pdb_file_name, ab_h_chain, ab_l_chain, ag_chain):
    in_file = open(pdb_file_name, 'r')
    print(pdb_file_name)
    f_ag = open(pdb_file_name[:-4] + '_ag' + '.pdb', "w")
    f_ab = open(pdb_file_name[:-4] + '_cdr' + '.pdb', "w")
    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom = Atom(line)
            res_full_name = atom.res_full_name
            chain_id = atom.chain_id
            chain_id = chain_id

            # WARNING: there are difference with respect to parsing.get_pdb_structure
            if res_full_name[0] == 'A' or res_full_name[0] == " ":
                if chain_id == ab_h_chain:
                    f_ab.write(line)
                if chain_id == ab_l_chain:
                    f_ab.write(line)
                if chain_id == ag_chain:
                    f_ag.write(line)
                if " | " in ag_chain:
                    c1, c2 = ag_chain.split(" | ")
                    if chain_id == c1:
                        f_ag.write(line)
                    if chain_id == c2:
                        f_ag.write(line)
                else:
                    if chain_id == ag_chain:
                        f_ag.write(line)
    f_ag.close()
    f_ab.close()
    print('close')


def get_residuals_atoms(residauls_chains):
    for chain_key in residauls_chains:
        chain = residauls_chains(chain_key)


def process_dataset(csv_file, path, max_cdr_length=MAX_CDR_LENGTH, max_ag_length=MAX_AG_LENGTH):
    num_in_contact = 0
    num_residues = 0
    all_cdrs = []
    all_cdrscoord = []
    all_cdrcenters = []
    all_cdr_atoms_coords = []
    all_cdr_atoms_contact_dist = []
    all_ab_atoms_coords = []
    all_lbls_distances=[]
    all_cdr_indices = []
    all_lbls = []
    all_lengths = []
    all_ag = []
    all_agcoord = []
    all_agcenters = []
    all_ag_atoms_coords = []
    all_ag_atoms_contact_dist = []
    all_ag_lengths = []
    all_edges_ag = []
    all_edges_mat = []
    all_epitope_lbls = []
    all_epitope_lbls_distances = []
    all_pdbname = []
    all_max = 0
    for cdrs_searchs, ag_search, cdrs_residuals, pdb, ag_residuals, complex_model in load_chains(csv_file, path):
        print("Processing PDB ", pdb)
        # split pdb file in ag and ab
        complex_model.save_pdb_cdr(path)
        complex_model.save_pdb_ag(path)
        cdrs, cdrs_coord, lbls, lbls_epitope, lbls_distance, lbls_epitope_distance, (
            numresidues, numincontact), lengths, ag, ag_coord, ag_length, edges_ag, edges, \
        cdr_atoms_coords, ag_atoms_coords, cdr_atoms_contact_dist, ag_atoms_contact_dist = \
            complex_process_chains(ag_search=ag_search, cdrs_search=cdrs_searchs, cdrs=cdrs_residuals,
                                   max_cdr_length=MAX_CDR_LENGTH, ag=ag_residuals, max_ag_length=MAX_AG_LENGTH)

        # cdr_atoms = complex_model.cdratoms
        # cdr_res_dims = [l.shape[0] for l in cdr_atoms_coords]
        # np.sum(cdr_res_dims)

        # cdr indices in atoms
        ab_atoms_coords = coords_atoms(complex_model.cdratoms).numpy()
        nbrs = NearestNeighbors(n_neighbors=1).fit(ab_atoms_coords)
        dists, cdr_indices = nbrs.kneighbors(
            np.concatenate(cdr_atoms_coords, 0))  # (n_cdr_atoms, 1) : ab_atoms -> cdr_atoms
        cdr_indices = np.squeeze(cdr_indices)
        u, c = np.unique(cdr_indices, return_counts=True)
        assert np.max(c) < 2
        # assert np.max(dists) < 1e-5

        ag_atoms = complex_model.agatoms
        ag_res_dims = [l.shape[0] for l in ag_atoms_coords]
        assert np.sum(ag_res_dims) == len(ag_atoms)

        num_in_contact += numincontact
        num_residues += numresidues
        # center the coordinates to the origin
        cdr_center = cdrs_coord.mean(0)
        # cdrs_coord = cdrs_coord - cdr_center
        ag_center = ag_coord.mean(0)
        # ag_coord = ag_coord - ag_center

        all_cdrs.append(cdrs)
        all_cdrscoord.append(cdrs_coord)
        all_cdrcenters.append(cdr_center)
        all_cdr_atoms_coords.append(cdr_atoms_coords)
        all_cdr_atoms_contact_dist.append(cdr_atoms_contact_dist)
        all_ab_atoms_coords.append(ab_atoms_coords)
        all_cdr_indices.append(cdr_indices)
        all_lbls.append(lbls)
        all_epitope_lbls.append(lbls_epitope)
        all_lbls_distances.append(lbls_distance)
        all_epitope_lbls_distances.append(lbls_epitope_distance)
        all_lengths.append(np.sum(lengths))
        all_ag.append(ag)
        all_agcoord.append(ag_coord)
        all_agcenters.append(ag_center)
        all_ag_atoms_coords.append(ag_atoms_coords)
        all_ag_atoms_contact_dist.append(ag_atoms_contact_dist)
        all_ag_lengths.append(ag_length)
        all_edges_ag.append(edges_ag)
        all_edges_mat.append(edges)
        all_pdbname.append(pdb)

    return {
        "pdb": all_pdbname,
        "feature_cdr": all_cdrs,
        "coords_cdr": all_cdrscoord,
        "centers_cdr": all_cdrcenters,
        "atoms_cdr": all_cdr_atoms_coords,
        "atoms_cdr_distances": all_cdr_atoms_contact_dist,
        "atoms_ab": all_ab_atoms_coords,
        "cdr_indices": all_cdr_indices,
        "lbls_cdr": all_lbls,
        "lbls_ag": all_epitope_lbls,
        "lbls_cdr_distances": all_lbls_distances,
        "lbls_ag_distances": all_epitope_lbls_distances,
        "lengths_cdr": all_lengths,
        "features_ag": all_ag,
        "coords_ag": all_agcoord,
        "centers_ag": all_agcenters,
        "atoms_ag": all_ag_atoms_coords,
        "atoms_ag_distances": all_ag_atoms_contact_dist,
        "lengths_ag": all_ag_lengths,
        "edges_ag": all_edges_ag,
        "edges_cdr": all_edges_mat,
    }

def load_chains(csv_file, path="data/"):
    PDBS_FORMAT = path + "{}.pdb"
    print("in load_chains")
    i=0
    for _, column in csv_file.iterrows():
        pdb_name = column['pdb']
        ab_h_chain = column['Hchain']
        ab_l_chain = column['Lchain']
        antigen_chain = column['antigen_chain']
        cdrs_atoms, cdrs, ag_atoms, ag, ag_names, model = get_pdb_structure(PDBS_FORMAT.format(pdb_name), ab_h_chain,
                                                                            ab_l_chain, antigen_chain)
        ag_search = NeighbourSearch(ag_atoms)  # replace this
        cdrs_search = NeighbourSearch(cdrs_atoms)
        yield cdrs_search, ag_search, cdrs, pdb_name, ag, model
        i = i + 1

def complex_process_chains(ag_search, cdrs_search, cdrs, max_cdr_length, ag, max_ag_length):
    num_residues = 0
    num_in_contact = 0
    contact = {}
    contact_epitope = {}
    distance_contact = {}
    distance_contact_epitope = {}

    for cdr_name, cdr_chain in cdrs.items():
        contact[cdr_name] = [residue_in_contact_with(res, ag_search, CONTACT_DISTANCE) for res in cdr_chain]
        distance_contact[cdr_name] = [residue_in_contact_with_distance(res, ag_search) for res in cdr_chain]
        num_residues += len(contact[cdr_name])
        num_in_contact += sum(contact[cdr_name])

    for ag_name, ag_chain in ag.items():
        contact_epitope[ag_name] = [residue_in_contact_with(res, cdrs_search, CONTACT_DISTANCE) for res in ag_chain]
        distance_contact_epitope[ag_name] = [residue_in_contact_with_distance(res, cdrs_search) for res in ag_chain]

    if num_in_contact < 5:
        print("Antibody has very few contact residues: ", num_in_contact, file=f)

    cdr_mats = []
    cdr_coords = []
    cdr_atoms_coords = []
    cont_mats = []
    distance_cont_mats = []
    contact_epitope_mats = []

    distance_cont_epitope_mats = []
    distance_contact_epitope_mat = []
    cdr_masks = []
    lengths = []
    ag_mats = []
    ag_coords = []
    ag_atoms_coords = []
    cdr_atoms_contact_dist = []
    ag_atoms_contact_dist = []
    ag_masks = []
    ag_lengths = []
    all_dist_mat = []
    all_edges_mat = []
    # edges_cdr = compute_edges_cdr(cdrs)
    # edges_ag = compute_edges_ag(ag)

    for ag_name, ag_chain in ag.items():
        # Converting residues to amino acid sequences
        agc = residue_seq_to_one(ag[ag_name])
        # ag_residues = ag[ag_name]
        ag_lengths = len(agc)
        ag_mat = ag_seq_to_one_hot(agc)
        ag_mats.append(ag_mat)
        ag_coord, ag_res_atoms = coords(ag[ag_name])
        ag_coords.append(ag_coord)
        ag_atoms_coords += ag_res_atoms
        # contact distance
        ag_atoms_contact_dist += distance_contact_epitope[ag_name]

        if len(contact_epitope[ag_name]) > 0:
            contact_epitope_mat = torch.FloatTensor(contact_epitope[ag_name])
        else:
            contact_epitope_mat = torch.zeros(max_cdr_length, 1)
        contact_epitope_mats.append(contact_epitope_mat)

        for i in range(len(distance_contact_epitope[ag_name])):
            if len(distance_contact_epitope[ag_name][i]) > 0:
                distance_contact_epitope_mat.append(min(distance_contact_epitope[ag_name][i]))
            distance_contact_epitope_tensor = torch.FloatTensor(distance_contact_epitope_mat)
        distance_cont_epitope_mats.append(distance_contact_epitope_tensor)

        maxi = 0
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Converting residues to amino acid sequences
        cdr_coord, cdr_res_atoms = coords(cdrs[cdr_name])
        cdr_coords.append(cdr_coord)
        cdr_atoms_coords += cdr_res_atoms
        # contact distance
        cdr_atoms_contact_dist += distance_contact[cdr_name]

        cdr_chain = cdrs[cdr_name]
        enc_cdr_chain = residue_seq_to_one(cdrs[cdr_name])
        chain_encoding = find_chain(cdr_name)
        cdr_mat = seq_to_one_hot(enc_cdr_chain, chain_encoding)

        if cdr_mat is not None:
            cdr_mats.append(cdr_mat)
            lengths.append(cdr_mat.shape[0])

            if len(contact[cdr_name]) > 0:
                cont_mat = torch.FloatTensor(contact[cdr_name])
            else:
                cont_mat = torch.zeros(max_cdr_length, 1)
            cont_mats.append(cont_mat)
            distance_contact_mat = []
            for i in range(len(distance_contact[cdr_name])):

                distance_contact_mat.append(min(distance_contact[cdr_name][i]))

            if len(distance_contact[cdr_name]) > 0:
                distance_contact_tensor = torch.FloatTensor(distance_contact_mat)
            else:
                distance_contact_tensor = torch.zeros(max_cdr_length, 1)

          # per residual?
            distance_cont_mats.append(distance_contact_tensor)

        # if len(distance_contact[cdr_name]) > 0:
        # distance_cont_mat = torch.FloatTensor(distance_contact[cdr_name])
        # else:
        # distance_cont_mat = torch.zeros(max_cdr_length, 1)
        # distance_cont_mats.append(distance_cont_mat)
    cdrs = torch.reshape(torch.cat(cdr_mats), [-1, NUM_FEATURES])
    ag = torch.reshape(torch.cat(ag_mats), [-1, NUM_FEATURES])
    lbls = torch.reshape(torch.cat(cont_mats), [-1])
    lbls_epitope = torch.reshape(torch.cat(contact_epitope_mats), [-1])
    lbls_distances = torch.reshape(torch.cat(distance_cont_mats), [-1])
    lbls_epitope_distances = torch.reshape(torch.cat(distance_cont_epitope_mats), [-1])
    cdr_coords = torch.reshape(torch.cat(cdr_coords), [-1, 3])
    ag_coords = torch.reshape(torch.cat(ag_coords), [-1, 3])
    edges_cdr = compute_edges(cdr_coords, k=15, d=10, as_edges=True)
    edges_cdr = torch.tensor(edges_cdr).transpose(0, 1).type(torch.int64)
    edges_ag = compute_edges(ag_coords, k=15, d=10, as_edges=True)
    edges_ag = torch.tensor(edges_ag).transpose(0, 1).type(torch.int64)

    return cdrs, cdr_coords, lbls, lbls_epitope, lbls_distances, lbls_epitope_distances, \
        (num_residues, num_in_contact), lengths, \
        ag, ag_coords, ag_lengths, edges_ag, edges_cdr, \
        cdr_atoms_coords, ag_atoms_coords, cdr_atoms_contact_dist, ag_atoms_contact_dist


def compute_edges_cdr(cdrs):
    edge_cdr_mat_list = []
    add_l = 0
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        cdr_chain = cdrs[cdr_name]
        edge_cdr_mat = []
        l=len(cdr_chain)
        if l != 0:
            for i in range(0,l):
                cdr_res=cdr_chain[i]
                cdr_search = NeighbourSearch(cdr_res.get_unpacked_list())
                for j in range(i+1,l):
                    cdr_res_2=cdr_chain[j]
                    if residue_in_contact_with(cdr_res_2, cdr_search, 10) == 1:
                        add_edge_cdr = [i + add_l, j + add_l]
                        add_edge_cdr_2 = [j + add_l, i + add_l]
                        edge_cdr_mat.append(add_edge_cdr)
                        edge_cdr_mat.append(add_edge_cdr_2)
            edge_cdr_mat = torch.tensor(edge_cdr_mat)
            edge_cdr_mat_list.append(edge_cdr_mat)
            add_l += l
    edge_cdr_mat_tensor = torch.cat(edge_cdr_mat_list).type(torch.int64)
    edge_cdr_mat_tensor_t = torch.transpose(edge_cdr_mat_tensor, 0, 1)
    return edge_cdr_mat_tensor_t


def compute_edges(coords, k=15, d=10, as_edges=True):
    # Compute the pairwise Euclidean distances between points
    dists = torch.cdist(coords, coords)

    # Find the indices of the k nearest neighbors for each point
    _, indices = torch.topk(dists, k=k + 1, largest=False)

    # Remove the first column of indices (self-indices)
    indices = indices[:, 1:]

    if as_edges:
        edges = []
        for i in range(indices.shape[0]):
            for j in indices[i]:
                if dists[i, j] < d:
                    edges.append((i, j.item()))
                    edges.append((j.item(), i))
        return edges
    else:
        # Create the k-NN graph as an adjacency matrix
        adj = torch.zeros((100, 100))
        adj.scatter_(1, indices, 1)

        # Filter the graph by distance d
        mask = dists <= d
        adj = adj * mask
        return adj


def compute_edges_ag(ag):
    edge_ag_mat_list = []
    add_l = 0
    for ag_name, ag_chain in ag.items():
        edge_ag_mat = []
        l = len(ag_chain)
        for i in range(0, l):
            ag_res = ag_chain[i]
            ag_search = NeighbourSearch(ag_res.get_unpacked_list())
            for j in range(i + 1, l):
                ag_res_2 = ag_chain[j]
                if residue_in_contact_with(ag_res_2, ag_search, 10) == 1:
                    add_edge_ag = [i + add_l, j + add_l]
                    add_edge_ag_2 = [j + add_l, i + add_l]
                    edge_ag_mat.append(add_edge_ag)
                    edge_ag_mat.append(add_edge_ag_2)
        edge_ag_mat = torch.tensor(edge_ag_mat)
        edge_ag_mat_list.append(edge_ag_mat)
        add_l += l
    edge_ag_mat_tensor = torch.cat(edge_ag_mat_list).type(torch.int64)
    edge_ag_mat_tensor_t = torch.transpose(edge_ag_mat_tensor, 0, 1)
    return edge_ag_mat_tensor_t

def residue_seq_to_one(seq):
    """
    Standard mapping from 3-letters amino acid type encoding to one.
    """
    three_to_one = lambda r: Polypeptide.three_to_one(r.name) \
        if r.name in Polypeptide.standard_aa_names else 'U'
    return list(map(three_to_one, seq))


def one_to_number(res_str):
    return [aa_s.index(r) for r in res_str]


def coords_atoms(atoms_list):
    coords = torch.ones((len(atoms_list), 3))
    for i, atom in enumerate(atoms_list):
        coords[i, :] = torch.tensor([atom.x_coord, atom.y_coord, atom.z_coord])
    return coords


def coords(res_str):
    res_coord = torch.ones((len(res_str), 3))
    i = 0
    atoms_res = list()
    for residue in res_str:
        ag_atoms_list = residue.child_list
        x_coord_list = []
        y_coord_list = []
        z_coord_list = []

        for atom in ag_atoms_list:
            x_coord_list.append(atom.x_coord)
            y_coord_list.append(atom.y_coord)
            z_coord_list.append(atom.z_coord)
        atoms_res.append(np.stack([x_coord_list, y_coord_list, z_coord_list], 1))
        res_coord[i, :] = torch.Tensor([np.average(x_coord_list), np.average(y_coord_list), np.average(z_coord_list)])
        i = i + 1
    return res_coord, atoms_res

def find_chain(cdr_name):
    if cdr_name == "H1":
        #print("H1")
        return [1, 0, 0, 0, 0, 0,]
    if cdr_name == "H2":
        #print("H2")
        return [0, 1, 0, 0, 0, 0]
    if cdr_name == "H3":
        #print("H3")
        return [0, 0, 1, 0, 0, 0]
    if cdr_name == "L1":
        #print("L1")
        return [0, 0, 0, 1, 0, 0]
    if cdr_name == "L2":
        #print("L2")
        return [0, 0, 0, 0, 1, 0]
    if cdr_name == "L3":
        #print("L3")
        return [0, 0, 0, 0, 0, 1]

def residue_in_contact_with(res, c_search, dist):
    """
    Computing ground truth values
    :param res: antibody amino acid
    :param c_search: KDTree using antigen atoms
    :param dist: threshold distance for which amino acid is considered binding.
    :return:
    """
    return any(c_search.search(a, dist) > 0   # search(self, centre, radius) - for each atom in res (antibody)
               for a in res.get_unpacked_list())


def residue_in_contact_with_distance(res, c_search):
    """
    Computing ground truth values
    :param res: antibody amino acid
    :param c_search: KDTree using antigen atoms
    :param dist: threshold distance for which amino acid is considered binding.
    :return:
    """
    distance = []
    for a in res.get_unpacked_list():
        distance.append(
            c_search.get_distance_neighbourgh(a))  # search(self, centre, radius) - for each atom in res (antibody)
    return distance

def antigene_in_contact_with(res, c_search, dist):
    """
    Computing ground truth values
    :param res: antibody amino acid
    :param c_search: KDTree using antigen atoms
    :param dist: threshold distance for which amino acid is considered binding.
    :return:
    """

    return any(c_search.search(a, dist) > 0   # search(self, centre, radius) - for each atom in res (antibody)
               for a in res.get_unpacked_list())


def seq_to_one_hot(res_seq_one, chain_encoding):
    ints = one_to_number(res_seq_one)
    if (len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))
        chain_encoding = torch.Tensor(chain_encoding)
        chain_encoding = chain_encoding.expand(onehot.shape[0], 6)
        concatenated = torch.cat((onehot, feats), 1)
        return torch.cat((onehot, feats), 1)
    else:
        return None


def ag_seq_to_one_hot(agc):
    ints = one_to_number(agc)
    if (len(ints) > 0):
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        onehot = to_categorical(ints, num_classes=len(aa_s))

        return torch.cat((onehot, feats), 1)
    else:
        return None

def aa_features():
    # Meiler's features
    prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]
    return torch.Tensor(prop1)

def to_categorical(y, num_classes):
    """ Converts a class vector to binary class matrix. """
    new_y = torch.LongTensor(y)
    n = new_y.size()[0]
    categorical = torch.zeros(n, num_classes)
    arangedTensor = torch.arange(0, n)
    intaranged = arangedTensor.long()
    categorical[intaranged, new_y] = 1
    return categorical

path = '../Data/data_epipred/data_val/'
csv_name = 'val_epitope.csv'  # 'val_epitope.csv' Test.csv train_epitope.csv


if __name__ == "__main__":
    # define argparser to get path and csv name
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, default='../Data/data_epipred/data_val/',
                        help='path to data')
    parser.add_argument('--csv_name', type=str, default='val_epitope.csv',
                        help='name of csv file')
    args = parser.parse_args()
    path = args.path
    csv_name = args.csv_name

    create_dataset(path+csv_name)
    dataset = open_dataset(path + "processed-dataset.p")
    pass