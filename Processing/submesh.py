import argparse
import os
import pickle
from os.path import exists, isfile

import igl
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

CRED = '\033[91m'
CEND = '\033[0m'


def save_pickle(file_name, vals):
    with open(file_name, "wb") as f:
        pickle.dump(vals, f, protocol=2)
    print(file_name)


def load_pickle(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        raise Exception(f"{file_path} could not be found")
    return data


def res_surface_correspondences(atoms_coords, sur_coords, resid, res_atoms_indices=None):
    """
    computes the nearest points on the surface to the residuals and atoms
    @param atoms_coords: [n_atoms x 3]
    @param sur_coords: [n_points x 3]
    @param resid: [n_atoms_cdr x 3]
    @param res_atoms_indices: index of the atom that are valid : None, [n_atoms_cdr]
    @return: nearest_res: index of the nearest residual to the point on the surface [n_points],
            nearest_atom: index of the nearest atom to the point on the surface [n_points],
            dists: distance of points on the surface to the nearest atom [n_points],
    """
    # find the atom nearest to the surface,
    # then transfer the feature of the residual associated to the atom to the atom's surface points

    # atoms_coords = np.concatenate(res_atoms, 0)
    # resid = np.concatenate([np.full((l.shape[0],), i) for i, l in enumerate(atoms_coords)], 0)
    nbrs = NearestNeighbors(n_neighbors=1).fit(atoms_coords)
    dists, nearest_atom = nbrs.kneighbors(sur_coords)  # (n_surf,1),(n_surf,1)
    nearest_atom = np.squeeze(nearest_atom)  # surface -> atoms
    dists = np.squeeze(dists)
    if res_atoms_indices is not None:
        #  take the point on the surface that are near to an atom with residual
        nearest_res = np.ones(nearest_atom.shape) * -1  # n_points
        for i_cdr in range(len(res_atoms_indices)):
            i_points_cdr = np.nonzero(nearest_atom == res_atoms_indices[i_cdr])[0]
            nearest_res[i_points_cdr] = resid[i_cdr]
    else:
        nearest_res = resid[nearest_atom]  # surface -> residuals
        # dists_res = dists[nearest_atom] # surface -> residuals

    return nearest_res, nearest_atom, dists


def open_dataset(dataset_cache="processed-dataset.p"):
    if exists(dataset_cache) and isfile(dataset_cache):
        print("Precomputed dataset found, loading...")
        with open(dataset_cache, "rb") as f:
            dataset = pickle.load(f)
    else:
        raise Exception(f"No dataset cache found : {dataset_cache} ")
    return dataset


import potpourri3d as pp3d
import scipy
import scipy.sparse.linalg as sla


def test_eigdecomp(coord, face):
    try:
        # l = igl.cotmatrix(coord, face)
        eigs_sigma = eps = 1e-8
        L = pp3d.cotan_laplacian(coord, face, denom_eps=1e-10)
        L_coo = L.tocoo()
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        massvec_np = pp3d.vertex_areas(coord, face)
        massvec_np += eps * np.mean(massvec_np)
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        evals_np, evecs_np = sla.eigsh(L_eigsh, k=10, M=Mmat, sigma=eigs_sigma)
    except Exception as e:
        print(e)
        return False
    return True

import pymeshlab as ml

def add_decimated_mesh(f, all_coord, all_face, all_color, n_points=0):
    ms = ml.MeshSet()
    ms.load_new_mesh(f)
    m = ms.current_mesh()

    print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    numFaces = 100 + 2*n_points

    #Simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > n_points):
        # ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture', targetfacenum=numFaces, preservenormal=True)
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
        ms.apply_filter('meshing_repair_non_manifold_vertices')
        ms.apply_filter('meshing_repair_non_manifold_edges', method=0)
        if ms.current_mesh().vertex_number() == n_points:
            break
        elif ms.current_mesh().vertex_number() < n_points:
            ms.apply_filter('meshing_surface_subdivision_loop', loopweight=0)
        # print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
        #Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (ms.current_mesh().vertex_number() - n_points)
    
    m = ms.current_mesh()
    # ms.save_current_mesh('output.off')
    coord = m.vertex_matrix()
    face = m.face_matrix()
    color = m.vertex_color_matrix()

    if ms.current_mesh().vertex_number() != n_points:
        raise Exception(f"Decimated mesh has {ms.current_mesh().vertex_number()} vertex instead of {n_points}")
    # else:
    #     print("Decimated to", n_points, "vertex mesh has", ms.current_mesh().vertex_number(), "vertex")

    print("Irregular vertex? ", np.any(igl.is_irregular_vertex(coord, face)))
    print("Is delaunay? ", np.all(igl.is_delaunay(coord, face)))
    print("Is edge manifold? ", igl.is_edge_manifold(face))
    # assert is_edge_manifold(face)
    assert test_eigdecomp(coord, face)
    # assert is_edge_manifold(face)
    all_coord.append(torch.tensor(coord))
    all_face.append(torch.tensor(face))
    all_color.append(torch.tensor(color))



def parse_params():
    parser = argparse.ArgumentParser(description='Compute surface from pdb.')
    parser.add_argument('-pf', '--pdb-folder', dest='pdb_folder', default= 'Data/data_epipred/val_pecan_unbound_aligned/', # 'Data/data_epipred/val_pecan/', 'Data/AF/gt_pdbs/'
                        type=str, help='folder containing the pdb to process')
    parser.add_argument('--processed-file', default='processed-dataset',
                        help="processed dataset file")
    parser.add_argument('--subset', default=None, 
                        help='subset of the dataset to process')
    parser.add_argument('-n', '--n-points', default=2000,
                        type=int, help='number of nodes for the simplified mesh. If <=0, it is unchanged')
    parser.add_argument('-t', '--pdb-type', default=None,
                        help="extension of the pdb to process: ag or cdr. If not specied, compute the surfaces of all the pdb in the folder")
    parser.add_argument('-dt', '--distance_threshold', default=4.5,
                        help="atoms distance to other molecule threshold")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_params()
    args.off_folder = os.path.join(args.pdb_folder, "off")

    print("Generating simplified meshes from high-quality wrl")
    all_coord = {"cdr": [], "ag": []}
    all_face = {"cdr": [], "ag": []}
    all_color = {"cdr": [], "ag": []}
    all_normal = {"cdr": [], "ag": []}
    all_feats = {"cdr": [], "ag": []}
    all_lbls = {"cdr": [], "ag": []}
    all_nn = {"cdr": [], "ag": []}
    all_pdb = []
    processed_path = os.path.join(args.pdb_folder, args.processed_file)
    if args.subset is not None:
        processed_path += '_' + args.subset
    processed_path += ".p"
    dataset = open_dataset(processed_path)
    for i_pdb in tqdm(range(len(dataset['pdb']))):
        pdb_name = dataset['pdb'][i_pdb]
        f = os.path.join(args.off_folder, f"{pdb_name}_cdr.off")
        tqdm.write("\t" + pdb_name)
        tqdm.write("AB")
        all_pdb.append(pdb_name)
        # cdr
        add_decimated_mesh(f, all_coord["cdr"], all_face["cdr"], all_color["cdr"], n_points=args.n_points)
        # features
        cdr_coords = np.concatenate(dataset["atoms_cdr"][i_pdb], 0)
        all_atoms = dataset["atoms_ab"][i_pdb]
        cdr_indices = dataset["cdr_indices"][i_pdb]
        resid = np.concatenate([np.full((l.shape[0],), i) for i, l in enumerate(dataset["atoms_cdr"][i_pdb])], 0)
        nearest_res_cdr, nearest_atom_cdr, dist_atom_cdr = res_surface_correspondences(all_atoms,
                                                                                       all_coord["cdr"][i_pdb],
                                                                                       resid,
                                                                                       res_atoms_indices=cdr_indices)
        res_coords = dataset["coords_cdr"][i_pdb].numpy()
        # nbrs = NearestNeighbors(n_neighbors=1).fit(res_coords)
        # _, indices = nbrs.kneighbors(all_coord["cdr"][i_pdb])
        # indices = np.squeeze(indices)
        # sanity check: does every residual have a point on the surface?
        sub_indices = np.unique(nearest_res_cdr)
        if sub_indices.size != len(res_coords):
            tqdm.write(f"Warning: {len(res_coords) - sub_indices.size} residual are not represented on the surface")
            out_gt = dataset["lbls_cdr"][i_pdb][np.setdiff1d(np.arange(len(res_coords)), sub_indices)]
            if out_gt.sum().item() > 0:
                # raise Exception(f"You are removing gt residuals from {pdb_name} cdrs")
                tqdm.write(f"{CRED}You are removing {out_gt.sum().item()} gt residuals from {pdb_name} cdrs {CEND}")
        cdr_sur = np.nonzero(nearest_res_cdr != -1)[0]
        feats_cdr = np.zeros((all_coord["cdr"][i_pdb].shape[0], dataset["feature_cdr"][i_pdb].shape[1]))
        feats_cdr[cdr_sur, :] = dataset["feature_cdr"][i_pdb][nearest_res_cdr[cdr_sur], :]
        all_feats["cdr"].append(torch.tensor(feats_cdr))
        # add threshold of atoms
        lbls_cdr = torch.zeros((all_coord["cdr"][i_pdb].shape[0]), dtype=torch.float32)
        lbls_cdr[cdr_sur] = dataset["lbls_cdr"][i_pdb][nearest_res_cdr[cdr_sur]]
        lbls_cdr_atoms = np.concatenate(dataset["atoms_cdr_distances"][i_pdb], 0) < args.distance_threshold
        lbls_atoms = np.zeros(all_atoms.shape[0], dtype=bool)
        lbls_atoms[cdr_indices] = lbls_cdr_atoms
        lbls_dist = lbls_atoms[nearest_atom_cdr]
        print("CDR Distance labels removed:",
              ((lbls_cdr.sum() - np.logical_and(lbls_dist, lbls_cdr).sum()) / lbls_cdr.sum()), " %")
        print("CDR Distance labels removed from dist:",
              ((lbls_dist.sum() - np.logical_and(lbls_dist, lbls_cdr).sum()) / lbls_dist.sum()), " %")
        lbls_cdr = np.logical_and(lbls_dist, lbls_cdr)
        all_lbls["cdr"].append(lbls_cdr.clone().detach())
        all_nn["cdr"].append({"nearest_res": nearest_res_cdr, "nearest_atom": nearest_atom_cdr,
                              "dist_atom": dist_atom_cdr})

        # ag
        tqdm.write("AG")
        f_ag = os.path.join(os.path.split(f)[0], f"{pdb_name}_ag.off")
        add_decimated_mesh(f_ag, all_coord["ag"], all_face["ag"], all_color["ag"], n_points=args.n_points)
        # features
        ag_coords = np.concatenate(dataset["atoms_ag"][i_pdb], 0)
        resid = np.concatenate([np.full((l.shape[0],), i) for i, l in enumerate(dataset["atoms_ag"][i_pdb])], 0)
        nearest_res_ag, nearest_atom_ag, dist_atom_ag = res_surface_correspondences(ag_coords,
                                                                                    all_coord["ag"][i_pdb],
                                                                                    resid)
        res_coords = dataset["coords_ag"][i_pdb].numpy()
        # nbrs = NearestNeighbors(n_neighbors=1).fit(dataset["coords_ag"][i_pdb])
        # _, indices = nbrs.kneighbors(all_coord["ag"][i_pdb])
        # indices = np.squeeze(indices)
        # sanity check: does every residual have a point on the surface?
        sub_indices = np.unique(nearest_res_ag)
        if sub_indices.size != len(res_coords):
            tqdm.write(f"Warning: {len(res_coords) - sub_indices.size} residual are not represented on the surface")
            out_gt = dataset["lbls_ag"][i_pdb][np.setdiff1d(np.arange(len(res_coords)), sub_indices)]
            if out_gt.sum().item() > 0:
                # raise Exception(f"You are removing gt residuals from {pdb_name} ag")
                tqdm.write(f"{CRED} You are removing {out_gt.sum().item()} gt residuals from {pdb_name} ag {CEND}")
        all_feats["ag"].append(dataset["features_ag"][i_pdb][nearest_res_ag, :])
        lbls_ag = dataset["lbls_ag"][i_pdb][nearest_res_ag]
        # add threshold of atoms
        lbls_atoms = np.concatenate(dataset["atoms_ag_distances"][i_pdb], 0) < args.distance_threshold
        lbls_dist = lbls_atoms[nearest_atom_ag]
        print("AG Distance labels removed:",
              (lbls_ag.sum() - np.logical_and(lbls_dist, lbls_ag).sum()) / lbls_ag.sum(), " %")
        print("AG Distance labels removed from dist:",
              ((lbls_dist.sum() - np.logical_and(lbls_dist, lbls_ag).sum()) / lbls_dist.sum()), " %")
        lbls_ag = np.logical_and(lbls_dist, lbls_ag)

        all_lbls["ag"].append(lbls_ag)
        all_nn["ag"].append({"nearest_res": nearest_res_ag, "nearest_atom": nearest_atom_ag,
                             "dist_atom": dist_atom_ag})

    print("done")
    # save as pickle
    print("Saving the pickles...", end='')
    save_pickle(
        os.path.join(args.pdb_folder, f"surfaces_points{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
        all_coord)
    save_pickle(
        os.path.join(args.pdb_folder, f"surfaces_faces{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
        all_face)
    save_pickle(
        os.path.join(args.pdb_folder, f"surfaces_color{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
        all_color)
    # with open(os.path.join(args.pdb_folder, "surfaces_normal.p"), "wb") as f:
    #     pickle.dump(all_normal, f, protocol=2)
    # print(os.path.join(args.pdb_folder, "surfaces_normal.p"))
    save_pickle(
        os.path.join(args.pdb_folder, f"surfaces_feats{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
        all_feats)
    save_pickle(
        os.path.join(args.pdb_folder, f"surfaces_lbls{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
        all_lbls)
    save_pickle(os.path.join(args.pdb_folder, f"surfaces_nn{ args.subset if  args.subset is not None else ''}{'_' + str(args.n_points) if args.n_points > 0 else ''}.p"),
                all_nn)
    print("done")
