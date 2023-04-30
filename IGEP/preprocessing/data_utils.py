import pickle

import numpy as np
import torch
from IGEP.preprocessing.epi_preprocess import *


def load_pickle(file):
    with open(file, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


def compute_dist_mat(lenghts, ag_lenghts):
    """
    Return matrices of ones to retrieve bipartite graph between the molecules
    @param lenghts:
    @param ag_lenghts:
    @return:
    """
    n_cdrs = len(lenghts)
    dist_mats_list = []
    for i in range(n_cdrs):
        length_i = lenghts[i]
        ag_length_i = ag_lenghts[i]
        dist_mat = torch.ones((length_i, ag_length_i))
        dist_mats_list.append(dist_mat)
    return dist_mats_list


def load_dataset(filename, args):
    unpickled = load_pickle(filename)

    unpickled["feature_cdr"] = change_features(unpickled["feature_cdr"])
    unpickled["features_ag"] = change_features(unpickled["features_ag"])
    unpickled["lengths_cdr"] = torch.tensor(unpickled["lengths_cdr"])
    unpickled["lengths_ag"] = torch.tensor(unpickled["lengths_ag"])

    # fully connected distance matrix
    dist_mats = compute_dist_mat(unpickled["feature_cdr"], unpickled["lengths_cdr"], unpickled["lengths_ag"])

    # centered
    if args.centered:
        unpickled["coords_cdr_nc"] = unpickled["coords_cdr"].copy()
        unpickled["coords_ag_nc"] = unpickled["coords_ag"].copy()
        coords_cdr_centered = centering_coord(unpickled["coords_cdr"], unpickled["centers_cdr"])
        # unpickled["coords_cdr"] = coords_cdr_centered
        coords_ag_centered = centering_coord(unpickled["coords_ag"], unpickled["centers_ag"])
        # unpickled["coords_ag"] = coords_ag_centered

    if "coords" in args.feats:
        if args.centered:
            unpickled["feature_cdr"], unpickled["features_ag"] = add_coords_as_features(unpickled["feature_cdr"],
                                                                                        unpickled["features_ag"],
                                                                                        coords_cdr_centered,
                                                                                        coords_ag_centered)
        else:
            unpickled["feature_cdr"], unpickled["features_ag"] = add_coords_as_features(unpickled["feature_cdr"],
                                                                                        unpickled["features_ag"],
                                                                                        unpickled["coords_cdr"],
                                                                                        unpickled["coords_ag"])

    # Create a list of protein complexes where each protein complex is represented by a dictionary
    protein_list = []
    for i in range(len(unpickled["features_ag"])):
        protein = {"name": unpickled['pdb'][i], 'cdrs': unpickled["feature_cdr"][i], 'ags': unpickled["features_ag"][i],
                   'edge_index_cdr': unpickled["edges_cdr"][i], 'edge_index_ag': unpickled["edges_ag"][i],
                   'cdr_lbls': unpickled["lbls_cdr"][i], 'ag_lbls': unpickled["lbls_ag"][i],
                   'dist_mat': dist_mats[i],
                   'coords_cdr': unpickled["coords_cdr"][i], 'coords_ag': unpickled["coords_ag"][i],
                   'distances_lbls_cdr': unpickled["lbls_cdr_distances"][i],
                   'distances_lbls_ag': unpickled["lbls_ag_distances"][i]}
        if args.centered:
            protein.update({'centered_cdr': coords_cdr_centered[i], 'centered_ag': coords_ag_centered[i], })
        protein_list.append(protein)
    # protein_list = create_dictionary_list(unpickled["feature_cdr"], unpickled["features_ag"], unpickled["edges_cdr"], unpickled["edges_ag"],
    #                                             unpickled["lbls_cdr"], unpickled["lbls_ag"], dist_mats, unpickled["coords_cdr"],
    #                                             unpickled["coords_ag"], unpickled["centers_cdr"], unpickled["centers_ag"],unpickled["lbls_cdr_distances"],unpickled["lbls_ag_distances"])
    return protein_list


def centering_coord(coords_ag_test, coords_ag_centers_test):
    # Create a list of protein complexes where each protein complex is represented by a dictionary
    centered_coords = []
    for i in range(len(coords_ag_test)):
        centered_coord = coords_ag_test[i] - coords_ag_centers_test[i]
        centered_coords.append(centered_coord)
    return centered_coords


def add_coords_as_features(cdr_list, ag_list, coords_cdr, coords_ag):
    # Create a list of protein complexes where each protein complex is represented by a dictionary
    cdr_list_feat = []
    ag_list_feat = []
    for i in range(len(ag_list)):
        concat_cdr = torch.cat((cdr_list[i], coords_cdr[i]), -1)
        concat_ag = torch.cat((ag_list[i], coords_ag[i]), -1)
        cdr_list_feat.append(concat_cdr)
        ag_list_feat.append(concat_ag)
    return cdr_list_feat, ag_list_feat


def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())

    theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen)
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R)


import torch.utils.data as data


class epipredDataset(data.Dataset):

    def __init__(self, filename, feats=["bio"], random_rotation=True, centered=True):
        self.load_data(filename)
        self.random_rotation = random_rotation
        self.centered = centered
        self.feats = feats

    def get_standard_item(self, index):
        raise NotImplementedError("Non implemented")

    def load_data(self, filename):
        self.unpickled = load_pickle(filename)

        # unpickled["feature_cdr"] = change_features(unpickled["feature_cdr"])
        # unpickled["features_ag"] = change_features(unpickled["features_ag"])
        self.unpickled["lengths_cdr"] = torch.tensor(self.unpickled["lengths_cdr"])
        self.unpickled["lengths_ag"] = torch.tensor(self.unpickled["lengths_ag"])

        # fully connected distance matrix
        self.dist_mats = compute_dist_mat(self.unpickled["lengths_cdr"], self.unpickled["lengths_ag"])

    def __getitem__(self, item):
        protein = {"name": self.unpickled['pdb'][item], 'cdrs': torch.tensor([]), 'ags': torch.tensor([]),
                   'edge_index_cdr': self.unpickled["edges_cdr"][item],
                   'edge_index_ag': self.unpickled["edges_ag"][item],
                   'cdr_lbls': self.unpickled["lbls_cdr"][item], 'ag_lbls': self.unpickled["lbls_ag"][item],
                   'dist_mat': self.dist_mats[item], 'distances_lbls_cdr': self.unpickled["lbls_cdr_distances"][item],
                   'distances_lbls_ag': self.unpickled["lbls_ag_distances"][item],
                   "coords_cdr": self.unpickled["coords_cdr"][item], 'coords_ag': self.unpickled["coords_ag"][item]
                   }

        if self.centered:
            protein["coords_cdr"] = protein["coords_cdr"] - self.unpickled["centers_cdr"][item]
            protein["coords_ag"] = protein["coords_ag"] - self.unpickled["centers_ag"][item]
        else:
            complex_coords = torch.cat((protein["coords_cdr"], protein["coords_ag"]), 0)
            protein["coords_cdr"] = protein["coords_cdr"] - torch.mean(complex_coords, 0)
            protein["coords_ag"] = protein["coords_ag"] - torch.mean(complex_coords, 0)

        if self.random_rotation:
            if self.centered:
                protein["coords_cdr"] = random_rotate_points(protein["coords_cdr"])
                protein["coords_ag"] = random_rotate_points(protein["coords_ag"])
            else:
                complex_coords = random_rotate_points(complex_coords)
                protein["coords_cdr"] = complex_coords[:protein["coords_cdr"].shape[0], :]
                protein["coords_ag"] = complex_coords[protein["coords_cdr"].shape[0]:, :]
                # protein["coords_cdr"] = random_rotate_points(
                #     protein["coords_cdr"] - self.unpickled["centers_cdr"][item]) + self.unpickled["centers_cdr"][item]
                # protein["coords_ag"] = random_rotate_points(protein["coords_ag"] - self.unpickled["centers_ag"][item]) + \
                #                        self.unpickled["centers_ag"][item]

        # out_data["ab_feats"] = torch.concat([self.get_feats(item, t, "cdr") for t in self.features], -1)
        if "bio" in self.feats:
            protein["cdrs"] = torch.cat((protein["cdrs"], self.unpickled["feature_cdr"][item]), -1)
            protein["ags"] = torch.cat((protein["ags"], self.unpickled["features_ag"][item]), -1)
        if "coords" in self.feats:
            protein["cdrs"] = torch.cat((protein["cdrs"], protein["coords_cdr"]), -1)
            protein["ags"] = torch.cat((protein["ags"], protein["coords_ag"]), -1)

        return protein

    def __len__(self):
        return len(self.unpickled['pdb'])
