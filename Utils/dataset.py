import collections
import json
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data

import OGEP.Models.diffusion_net as diffusion_net
from OGEP.Models.diffusion_net.geometry import compute_hks_autoscale
from OGEP.Models.diffusion_net.utils import random_rotate_points


def to_device(tensors, device=torch.device('cuda:0')):
    if isinstance(tensors, collections.abc.Mapping):
        for key in tensors:
            tensors[key] = tensors[key].to(device)
    elif isinstance(tensors, collections.abc.Sequence):
        for i_tensor in range(len(tensors)):
            tensors[i_tensor] = tensors[i_tensor].to(device)
    return tensors


class ShapeNetDataset3(data.Dataset):
    def __init__(self,
                 root,
                 npoints=6000,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=False,
                 need_operators=False,
                 k_eig=128,
                 rs=0,
                 precompute_data=False):

        self.root = root
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        # self.seg_classes = {}
        self.feat_dim = 2
        self.rs = rs

        self.need_operators = need_operators
        self.k_eig = k_eig

        self.precomputed = False

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split',
                                 'shuffled_{}_file_list_{}.json'.format(split, class_choice[0][0]))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.seg_classes[ls[0]] = int(ls[1])
        # self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print(self.seg_classes, self.num_seg_classes)

        if precompute_data:
            self.precompute()

    def precompute(self, use_cache=True):

        self.op_cache_dir = os.path.join(self.root, "op_cache") if use_cache else None
        self.verts_list = [None] * len(self.datapath)
        self.feats_list = [None] * len(self.datapath)
        self.faces_list = [torch.tensor([])] * len(self.datapath)
        self.labels_list = [None] * len(self.datapath)

        for i_datapath in range(len(self.datapath)):
            self.verts_list[i_datapath], self.feats_list[i_datapath], self.labels_list[
                i_datapath] = self.get_standard_item(i_datapath)
        self.verts_list = tuple(self.verts_list)
        self.feats_list = tuple(self.feats_list)
        self.faces_list = tuple(self.faces_list)
        self.labels_list = tuple(self.labels_list)

        # Precompute operators
        if self.need_operators:
            self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(
                self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        self.precomputed = True

    def get_standard_item(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        if self.rs == 1:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            # resample
            point_set = point_set[choice, :]
            seg = seg[choice]

        # noemalize flag
        cnorm = 0

        coordset = point_set[:, 0:3]
        coordset = coordset - np.expand_dims(np.mean(coordset, axis=0), 0)  # center
        featset = point_set[:, 3:]
        # size normalize
        if cnorm == 1:
            dist = np.max(np.sqrt(np.sum(coordset ** 2, axis=1)), 0)
            coordset = coordset / dist  # scale

        coordset = torch.from_numpy(coordset)
        featset = torch.from_numpy(featset)

        if self.classification:
            cls = self.classes[self.datapath[index][0]]
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))
            return coordset, featset, cls
        else:
            seg = torch.from_numpy(seg)
            return coordset, featset, seg

    def __getitem__(self, index):
        if self.precomputed:
            if self.need_operators:
                return self.verts_list[index], self.feats_list[index], self.faces_list[index], self.frames_list[index], \
                       self.massvec_list[index], \
                       self.L_list[index], self.evals_list[index], self.evecs_list[index], self.gradX_list[index], \
                       self.gradY_list[
                           index], self.labels_list[index]
            else:
                return self.verts_list[index], self.feats_list[index], self.labels_list[index]
        verts, feats, labels = self.get_standard_item(index)
        if self.data_augmentation:
            choice = np.random.choice(verts.size(1), self.npoints, replace=True)
            # resample
            verts = verts[choice, :]
            feats = feats[choice, :]
            labels = labels[choice]

        if self.need_operators:
            faces = torch.tensor([])
            frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.get_operators(verts, faces, k_eig=self.k_eig,
                                                                                         op_cache_dir=None)
            return verts, feats, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels
        else:
            return verts, feats, labels

    def __len__(self):
        return len(self.datapath)


class epiapbsconDataset(ShapeNetDataset3):
    def __init__(self, **kwargs):
        super().__init__(os.path.join("data", "epiapbscon"), **kwargs)


## data epipred ##
class epipredDataset(data.Dataset):
    def __init__(self, split="validation", as_mesh=True, npoints=0, centered=True, data_augmentation=False,
                 features=["bio"], hks_dim=16, get_faces=False,
                 need_operators=False, k_eig=128, precompute_data=False,
                 device=torch.device("cpu"), dtype=torch.float64, load_submesh=False, load_residuals=False,
                 **kwargs):
        data_splits = {"train": "train", "test": "test", "validation": "val"}
        if split not in list(data_splits.keys()):
            raise ValueError(f"{split} is not a valid split. Options are {data_splits.keys()}")
        self.root = os.path.join("Data", "data_epipred", "data_" + data_splits[split])

        self.as_mesh = as_mesh
        self.get_faces = get_faces

        self.centered = centered
        self.npoints = npoints  # if 0, then no subsampling
        self.load_submesh = load_submesh and self.npoints > 0
        self.load_residuals = load_residuals
        self.data_augmentation = data_augmentation  # rotation

        self.features = features
        self.hks_dim = hks_dim

        self.need_operators = need_operators
        self.k_eig = k_eig

        self.device = device
        self.dtype = dtype

        self.load_data()

        self.precomputed = False
        if precompute_data:
            self.precompute()


    def load_data(self):
        # vertices
        with open(os.path.join(self.root, f"surfaces_points{'_' + str(self.npoints) if self.load_submesh else ''}.p"),
                  "rb") as f:
            self.verts = pickle.load(f)
            for mol in ["cdr", "ag"]:
                for i_v in range(len(self.verts[mol])):
                    self.verts[mol][i_v] = self.verts[mol][i_v].to(self.dtype)
                # self.ab_verts = all_coord["cdr"] # list of numpy array
            # self.ag_verts = all_coord["ag"] # list of numpy array
        if self.centered:
            for mol in self.verts:
                for i_v in range(len(self.verts[mol])):
                    self.verts[mol][i_v] -= torch.mean(self.verts[mol][i_v], 0)
        # faces
        if self.as_mesh:
            with open(
                    os.path.join(self.root, f"surfaces_faces{'_' + str(self.npoints) if self.load_submesh else ''}.p"),
                    "rb") as f:
                self.faces = pickle.load(f)
                # self.ab_faces = all_face["cdr"]  # list of numpy array
                # self.ag_faces = all_face["ag"]  # list of numpy array
        else:
            self.faces = {"cdr": None, "ag": None}
        # features
        if "bio" in self.features:
            with open(
                    os.path.join(self.root, f"surfaces_feats{'_' + str(self.npoints) if self.load_submesh else ''}.p"),
                    "rb") as f:
                self.feats = pickle.load(f)
                for mol in ["cdr", "ag"]:
                    for i_v in range(len(self.feats[mol])):
                        self.feats[mol][i_v] = self.feats[mol][i_v].to(self.dtype)
        # lbls
        with open(os.path.join(self.root, f"surfaces_lbls{'_' + str(self.npoints) if self.load_submesh else ''}.p"),
                  "rb") as f:
            self.labels = pickle.load(f)
        # nn
        with open(os.path.join(self.root, f"surfaces_nn{'_' + str(self.npoints) if self.load_submesh else ''}.p"),
                  "rb") as f:
            self.nn = pickle.load(f)
        # residuals
        if self.load_residuals:
            with open(os.path.join(self.root, f"processed-dataset.p"), "rb") as f:
                res_dataset = pickle.load(f)
                self.residuals = {"lbls_ab_res": res_dataset["lbls_cdr"],
                                  "lbls_ag_res": res_dataset["lbls_ag"], }

    def precompute(self, use_cache=True):
        # Precompute operators
        if self.need_operators:
            self.op_cache_dir = os.path.join(self.root, "op_cache") if use_cache else None

            self.frames_list = {"cdr": [], "ag": []}
            self.massvec_list = {"cdr": [], "ag": []}
            self.L_list = {"cdr": [], "ag": []}
            self.evals_list = {"cdr": [], "ag": []}
            self.evecs_list = {"cdr": [], "ag": []}
            self.gradX_list = {"cdr": [], "ag": []}
            self.gradY_list = {"cdr": [], "ag": []}
            # antibody
            self.frames_list["cdr"], self.massvec_list["cdr"], self.L_list["cdr"], \
            self.evals_list["cdr"], self.evecs_list["cdr"], \
            self.gradX_list["cdr"], self.gradY_list["cdr"] = diffusion_net.geometry.get_all_operators(
                self.verts["cdr"], [face.to(torch.int64) for face in self.faces["cdr"]], k_eig=self.k_eig,
                op_cache_dir=self.op_cache_dir)
            # antigene
            self.frames_list["ag"], self.massvec_list["ag"], self.L_list["ag"], \
            self.evals_list["ag"], self.evecs_list["ag"], \
            self.gradX_list["ag"], self.gradY_list["ag"] = diffusion_net.geometry.get_all_operators(
                self.verts["ag"], [face.to(torch.int64) for face in self.faces["ag"]], k_eig=self.k_eig,
                op_cache_dir=self.op_cache_dir)

        self.precomputed = True

    def get_standard_item(self, index):
        raise NotImplementedError("Non implemented")

    def get_feats(self, index, type="bio", mol="ag", choice=None, **kwargs):
        if type == "bio":
            if choice is None:
                return self.feats[mol][index]
            else:
                return self.feats[mol][index][choice, :]
        if type == "xyz":
            if "verts" in kwargs:
                return kwargs["verts"]
            else:
                if choice is None:
                    return self.verts[mol][index]
                else:
                    return self.verts[mol][index][choice, :]
        if type == "hks":
            if "evals" not in kwargs or "evecs" not in kwargs or kwargs["evals"] is None or kwargs["evecs"] is None:
                v = self.verts[mol][index]
                if choice is not None:
                    v = v[choice, :]
                _, _, _, kwargs["evals"], kwargs["evecs"], _, _ = diffusion_net.get_operators(v,
                                                                                              self.verts[mol][
                                                                                                  index] if self.as_mesh else torch.tensor(
                                                                                                  []),
                                                                                              k_eig=self.k_eig,
                                                                                              op_cache_dir=None)

            return compute_hks_autoscale(kwargs["evals"], kwargs["evecs"], kwargs["hks_dim"])

    def __getitem__(self, index):
        out_data = {"ab_verts": self.verts["cdr"][index], "ag_verts": self.verts["ag"][index],
                    "ab_labels": self.labels["cdr"][index], "ag_labels": self.labels["ag"][index]}
        # cdr mask
        cdr_sur = self.nn["cdr"][index]["nearest_res"] != -1
        out_data.update({"cdr_mask": torch.from_numpy(cdr_sur)})

        if self.load_residuals:
            out_data.update({"nearest_res_ab": torch.tensor(self.nn["cdr"][index]["nearest_res"], dtype=int),
                             "nearest_res_ag": torch.tensor(self.nn["ag"][index]["nearest_res"], dtype=int),
                             "lbls_ab_res": self.residuals["lbls_ab_res"][index],
                             "lbls_ag_res": self.residuals["lbls_ag_res"][index],
                             })

        choice_ab = None
        choice_ag = None

        if self.get_faces:
            out_data.update({"ab_faces": self.faces["cdr"][index], "ag_faces": self.faces["ag"][index]})
        # if self.as_mesh:
        #     out_data.update({"ab_faces": self.faces["cdr"][index], "ag_faces": self.faces["ag"][index]})
        # else:
        #     out_data.update({"ab_faces": None, "ag_faces": None})

        if self.data_augmentation:
            out_data["ab_verts"] = random_rotate_points(out_data["ab_verts"])
            out_data["ag_verts"] = random_rotate_points(out_data["ag_verts"])

        if self.precomputed:
            if self.need_operators:
                out_data.update(
                    {"ab_frames": self.frames_list["cdr"][index], "ab_massvec": self.massvec_list["cdr"][index],
                     "ab_L": self.L_list["cdr"][index],
                     "ab_evals": self.evals_list["cdr"][index], "ab_evecs": self.evecs_list["cdr"][index],
                     "ab_gradX": self.gradX_list["cdr"][index], "ab_gradY": self.gradY_list["cdr"][index],
                     "ag_frames": self.frames_list["ag"][index], "ag_massvec": self.massvec_list["ag"][index],
                     "ag_L": self.L_list["ag"][index],
                     "ag_evals": self.evals_list["ag"][index], "ag_evecs": self.evecs_list["ag"][index],
                     "ag_gradX": self.gradX_list["ag"][index], "ag_gradY": self.gradY_list["ag"][index],
                     })
        else:
            if self.npoints > 0 and not self.load_submesh and not self.as_mesh:
                choice_ab = np.random.choice(np.arange(out_data["ab_verts"].size(0)), self.npoints, replace=True)
                out_data["ab_verts"] = out_data["ab_verts"][choice_ab, :]
                # out_data["ab_feats"] = out_data["ab_feats"][choice_ab, :]
                out_data["ab_labels"] = out_data["ab_labels"][choice_ab]
                out_data["cdr_mask"] = out_data["cdr_mask"][choice_ab]
                out_data["choice_ab"] = torch.tensor(choice_ab, dtype=int)
                # ag
                choice_ag = np.random.choice(np.arange(out_data["ag_verts"].size(0)), self.npoints, replace=True)
                out_data["ag_verts"] = out_data["ag_verts"][choice_ag, :]
                # out_data["ag_feats"] = out_data["ag_feats"][choice_ag, :]
                out_data["ag_labels"] = out_data["ag_labels"][choice_ag]
                out_data["choice_ag"] = torch.tensor(choice_ag, dtype=int)


            if self.need_operators:
                faces = torch.tensor([])
                try:
                    # ab
                    faces_ab = self.faces["cdr"][index].to(torch.int64) if self.as_mesh else torch.tensor([])
                    frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.get_operators(out_data["ab_verts"],
                                                                                                 faces_ab,
                                                                                                 k_eig=self.k_eig,
                                                                                                 op_cache_dir=None)
                    out_data.update(
                        {"ab_frames": frames, "ab_massvec": massvec, "ab_L": L,
                         "ab_evals": evals, "ab_evecs": evecs, "ab_gradX": gradX, "ab_gradY": gradY})
                except RuntimeError as rerr:
                    print("During ab operator computation")
                    print(rerr)
                    return None

                    # out_data.update({"ab_L": torch.tensor([]), "ag_L": torch.tensor([])})
                    # return out_data
                try:
                    # ag
                    faces_ag = self.faces["ag"][index].to(torch.int64) if self.as_mesh else torch.tensor([])
                    frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.get_operators(out_data["ag_verts"],
                                                                                                 faces_ag,
                                                                                                 k_eig=self.k_eig,
                                                                                                 op_cache_dir=None)
                    out_data.update(
                        {"ag_frames": frames, "ag_massvec": massvec, "ag_L": L,
                         "ag_evals": evals, "ag_evecs": evecs, "ag_gradX": gradX, "ag_gradY": gradY})
                except RuntimeError as rerr:
                    print("During ag operator computation")
                    print(rerr)
                    return None

                    # out_data.update({"ag_L": torch.tensor([])})
                    # return out_data

        # features
        out_data["ab_feats"] = torch.concat([self.get_feats(index, t, "cdr",
                                                            verts=out_data["ab_verts"],
                                                            choice=choice_ab,
                                                            evals=out_data.get("ab_evals", None),
                                                            evecs=out_data.get("ab_evecs", None),
                                                            hks_dim=self.hks_dim,
                                                            ) for t in self.features], -1)
        out_data["ag_feats"] = torch.concat([self.get_feats(index, t, "ag",
                                                            verts=out_data["ag_verts"],
                                                            choice=choice_ag,
                                                            evals=out_data.get("ag_evals", None),
                                                            evecs=out_data.get("ag_evecs", None),
                                                            hks_dim=self.hks_dim,
                                                            ) for t in self.features], -1)

        return out_data

    def __len__(self):
        return len(self.verts["cdr"])

class datasetCreator():
    datasets = {"epiapbscon": epiapbsconDataset, "epipred": epipredDataset, }

    @staticmethod
    def get_dataset(name, **kwargs):
        return datasetCreator.datasets[name](**kwargs)
