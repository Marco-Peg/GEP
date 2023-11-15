# given a pdb file generate the "Connoly" surface of the protein
import argparse
import glob
import os
import pickle

import igl
import numpy as np
import torch
from pymol import cmd  # ,stored
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from Processing_features import open_dataset
from utils_surface.wrl_parse import read_wrl2


def load_pickle(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        raise Exception(f"{file_path} could not be found")
    return data


def pdb2wrl(file_path, output_path):
    f = os.path.split(file_path)[1]
    cmd.load(os.path.join(file_path))
    cmd.set('surface_quality', '0')
    cmd.set('surface_mode', '1')
    cmd.set('cavity_cull', 30)
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(output_path)
    cmd.delete('all')
    return


def collapse_repeated_vertices(verts, faces):
    verts_collapsed, orginal2collapsed, collapsed2original = np.unique(verts, return_index=True, return_inverse=True,
                                                                       axis=0)
    # collapse faces
    for i_vert in range(verts.shape[0]):
        i_face = np.nonzero(faces == i_vert)[0]
        faces[i_face] = collapsed2original[i_vert]

    np.sort([])

    return verts_collapsed


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

    return nearest_res, nearest_atom, dists


def parse_params():
    parser = argparse.ArgumentParser(description='Compute surface from pdb.')
    parser.add_argument('-pf', '--pdb-folder', dest='pdb_folder', default='../Data/data_epipred/data_train',
                        type=str, help='folder containing the pdb to process')
    parser.add_argument('-df', '--dest-folder', dest='dest_folder', default='../Data/data_epipred/data_val/wrl',
                        type=str, help='destination folder where to save the wrl files')
    parser.add_argument('-t', '--pdb-type', default=None,
                        help="extension of the pdb to process: ag or cdr. If not specied, compute the surfaces of all the pdb in the folder")
    parser.add_argument('-w', '--wrl', default=True,
                        help="compute wrl files")
    parser.add_argument('-p', '--pickle', default=True,
                        help="generate pickle files from wrl")
    parser.add_argument('-dt', '--distance_threshold', default=4.5,
                        help="atoms distance to other molecule threshold")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_params()
    args.dest_folder = os.path.join(args.pdb_folder, "off")

    if args.wrl:
        # create wrl folder if it doesn't exist
        if not os.path.exists(args.dest_folder):
            print("Creating the destination folder")
            os.mkdir(args.dest_folder)
        # compute and save the wrl for each pdb
        print("Generating wrl from folder: {}".format(args.pdb_folder))
        temp_file = os.path.join(args.dest_folder, "temp.wrl")
        list_files = list(glob.iglob(os.path.join(
            args.pdb_folder, "*{}.pdb".format("" if args.pdb_type is None else "_" + args.pdb_type))))
        for f in tqdm(list_files):
            # print("\t",os.path.split(f)[1])
            pdb_name = os.path.split(f)[1][:-4]
            tqdm.write("\t" + pdb_name)
            pdb2wrl(f, temp_file)
            coord, face, color, normal = read_wrl2(temp_file)
            [SV, SVI, SVJ, SF] = igl.remove_duplicate_vertices(coord, face, 1e-7)
            SC = color[SVI, :]
            # sometimes there may be inner surfaces, we removed them as smaller connected components
            comps = igl.vertex_components(SF)
            # remove the smaller components
            ids, counts = np.unique(comps, return_counts=True)
            if counts.shape[0] > 1:
                id_big = ids[np.argmax(counts)]
                inds_big = np.nonzero(comps == id_big)[0]
                tqdm.write("Removing smaller connected components")
                ind_face_keep = np.all(np.isin(SF, inds_big), 1)
                SF = SF[ind_face_keep, :]
                SV, SF, IM, J = igl.remove_unreferenced(SV, SF)
                SC = color[J, :]

            dest_off = os.path.join(args.dest_folder, os.path.split(f)[1][0:-4] + '.off')
            igl.write_off(dest_off, SV, SF, SC)

    # generate the pickle files with the coordinates
    if args.pickle:
        if not os.path.exists(args.dest_folder):
            raise NotADirectoryError(args.dest_folder + " is not a folder")
        print("Generating surfaces pickles from wrl")
        all_coord = {"cdr": [], "ag": []}
        all_face = {"cdr": [], "ag": []}
        all_color = {"cdr": [], "ag": []}
        all_normal = {"cdr": [], "ag": []}
        all_pdb = []
        dataset = open_dataset(os.path.join(args.pdb_folder, "processed-dataset.p"))
        for i_pdb in tqdm(range(len(dataset['pdb']))):
            pdb_name = dataset['pdb'][i_pdb]
            f = os.path.join(args.dest_folder, f"{pdb_name}_cdr.off")
            tqdm.write("\t" + pdb_name)
            all_pdb.append(pdb_name)
            # cdr
            coord, face, color = igl.read_off(f)
            all_coord["cdr"].append(torch.tensor(coord))
            all_face["cdr"].append(torch.tensor(face))
            all_color["cdr"].append(torch.tensor(color))
            # ag
            f_ag = os.path.join(os.path.split(f)[0], f"{pdb_name}_ag.off")
            coord, face, color = igl.read_off(f_ag)
            all_coord["ag"].append(torch.tensor(coord))
            all_face["ag"].append(torch.tensor(face))
            all_color["ag"].append(torch.tensor(color))
        print("done")
        # save as pickle
        print("Saving the pickles...", end='')
        with open(os.path.join(args.pdb_folder, "surfaces_points.p"), "wb") as f:
            pickle.dump(all_coord, f, protocol=2)
        print(os.path.join(args.pdb_folder, "surfaces_points.p"))
        with open(os.path.join(args.pdb_folder, "surfaces_faces.p"), "wb") as f:
            pickle.dump(all_face, f, protocol=2)
        print(os.path.join(args.pdb_folder, "surfaces_faces.p"))
        with open(os.path.join(args.pdb_folder, "surfaces_color.p"), "wb") as f:
            pickle.dump(all_color, f, protocol=2)
        print(os.path.join(args.pdb_folder, "surfaces_color.p"))
        print("done")

        # transfer features from residuals to surface points
        print("Computing the features on the surface")
        all_coord = load_pickle(os.path.join(args.pdb_folder, "surfaces_points.p"))
        wrl_pdbs = all_pdb
        all_feats = {"cdr": [], "ag": []}
        all_lbls = {"cdr": [], "ag": []}
        all_nn = {"cdr": [], "ag": []}
        for i_pdb in tqdm(range(len(dataset['pdb']))):
            pdb_name = dataset['pdb'][i_pdb]
            assert all_pdb[i_pdb] == pdb_name

            # ag
            ag_coords = np.concatenate(dataset["atoms_ag"][i_pdb], 0)
            resid = np.concatenate([np.full((l.shape[0],), i) for i, l in enumerate(dataset["atoms_ag"][i_pdb])], 0)
            nearest_res_ag, nearest_atom_ag, dist_atom_ag = res_surface_correspondences(ag_coords,
                                                                                        all_coord["ag"][i_pdb],
                                                                                        resid)
            all_feats["ag"].append(dataset["features_ag"][i_pdb][nearest_res_ag, :])
            lbls_ag = dataset["lbls_ag"][i_pdb][nearest_res_ag]
            # add threshold of atoms
            lbls_atoms = np.concatenate(dataset["atoms_ag_distances"][i_pdb], 0) < args.distance_threshold
            lbls_dist = lbls_atoms[nearest_atom_ag]
            print("AG Distance labels removed:",
                  (lbls_ag.sum() - np.logical_and(lbls_dist, lbls_ag).sum()) / lbls_ag.sum(), " %")
            print("AG Distance labels removed from dist:",
                  (lbls_dist.sum() - np.logical_and(lbls_dist, lbls_ag).sum()) / lbls_dist.sum(), " %")
            lbls_ag = np.logical_and(lbls_dist, lbls_ag)

            all_lbls["ag"].append(lbls_ag)
            all_nn["ag"].append({"nearest_res": nearest_res_ag, "nearest_atom": nearest_atom_ag,
                                 "dist_atom": dist_atom_ag})

            # cdr
            cdr_coords = np.concatenate(dataset["atoms_cdr"][i_pdb], 0)
            all_atoms = dataset["atoms_ab"][i_pdb]
            cdr_indices = dataset["cdr_indices"][i_pdb]
            resid = np.concatenate([np.full((l.shape[0],), i) for i, l in enumerate(dataset["atoms_cdr"][i_pdb])], 0)
            nearest_res_cdr, nearest_atom_cdr, dist_atom_cdr = res_surface_correspondences(all_atoms,
                                                                                           all_coord["cdr"][i_pdb],
                                                                                           resid,
                                                                                           res_atoms_indices=cdr_indices)
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
                  (lbls_cdr.sum() - np.logical_and(lbls_dist, lbls_cdr).sum()) / lbls_cdr.sum(), " %")
            print("CDR Distance labels removed from dist:",
                  (lbls_dist.sum() - np.logical_and(lbls_dist, lbls_cdr).sum()) / lbls_dist.sum(), " %")
            lbls_cdr = np.logical_and(lbls_dist, lbls_cdr)
            all_lbls["cdr"].append(torch.tensor(lbls_cdr))

            all_nn["cdr"].append({"nearest_res": nearest_res_cdr, "nearest_atom": nearest_atom_cdr,
                                  "dist_atom": dist_atom_cdr})

        print("Saving the features and labels...", end='')
        with open(os.path.join(args.pdb_folder, "surfaces_feats.p"), "wb") as f:
            pickle.dump(all_feats, f, protocol=2)
        print("Generated: ", os.path.join(args.pdb_folder, "surfaces_feats.p"))
        with open(os.path.join(args.pdb_folder, "surfaces_lbls.p"), "wb") as f:
            pickle.dump(all_lbls, f, protocol=2)
        print("Generated: ", os.path.join(args.pdb_folder, "surfaces_lbls.p"))
        with open(os.path.join(args.pdb_folder, "surfaces_nn.p"), "wb") as f:
            pickle.dump(all_nn, f, protocol=2)
        print("Generated: ", os.path.join(args.pdb_folder, "surfaces_nn.p"))
        print("done")

    feats = load_pickle(os.path.join(args.pdb_folder, "surfaces_feats.p"))
