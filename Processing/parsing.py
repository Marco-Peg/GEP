"""
Custom parser for extracting data from PDB files.
"""

import os
from Processing.constants import *


class Entity(object):
    """
    Parent class for residue, chain - has name, id and list of children.
    """

    def __init__(self, name, id):
        self.name = name
        self._id_ = id
        self.child_list = []

    def has_child(self, child_id):
        return (child_id in self.child_list)

    def has_empty_child_list(self):
        return (self.child_list == [])

    def get_id(self):
        return self._id_

    def get_name(self):
        return self.name

    def add_child(self, child):
        if (not (self.has_child(child.get_id()))):
            self.child_list.append(child)

    def get_child_list(self):
        return self.child_list


class Chain(Entity):
    """
    Chain representation - has chain_id and child_list is formed from its residues.
    """
    def __init__(self, name, id = 0):
        Entity.__init__(self, name, id)

    def __repr__(self):
        full_id = (self.name)
        return "<Chain chain=%s>" % full_id

    def get_unpacked_list(self):
        return self.child_list


class Residue(Entity):
    """
    Residue representation - has sequence_number and child_list is formed from its atoms.
    """
    def __init__(self, name, id, full_name, full_seq_num):
        Entity.__init__(self, name, id)
        self.full_seq_num = full_seq_num
        self.full_name = full_name

    def __repr__(self):
        seq = self.get_id()
        full_id = (self.full_seq_num, seq)
        return "<Residue %s resseq=%s>" % full_id

    def get_unpacked_list(self):
        return self.child_list

    def get_full_name(self):
        return self.full_name


class AGResidue(Residue):
    def __init__(self, name, id, full_name, full_seq_num, x_pos, y_pos, z_pos):
        Residue.__init__(self, name, id, full_name, full_seq_num, x_pos, y_pos, z_pos)
        self.full_seq_num = full_seq_num
        self.full_name = full_name
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.nr_atoms = 1

    def __repr__(self):
        seq = self.get_id()
        full_id = (self.full_seq_num, seq)
        return "<AGResidue %s resseq=%s>" % full_id

    def get_unpacked_list(self):
        return self.child_list

    def get_full_name(self):
        return self.full_name

    def add_atom(self, atom):
        self.x_pos = ((self.x_pos * self.nr_atoms) +atom.x_coord) / (self.nr_atoms+1)
        self.y_pos = ((self.x_pos * self.nr_atoms) + atom.y_coord) / (self.nr_atoms+1)
        self.z_pos = ((self.x_pos * self.nr_atoms) + atom.z_coord) / (self.nr_atoms+1)
        self.nr_atoms = self.nr_atoms + 1


class Atom(object):
    """
    Atom representation obtained after extracting information from "ATOM" lines in PDB files.
    """
    def __init__(self, line):
        atom_features = line.split(" ")
        atom_features = [f for f in atom_features if f != '' and f != '\n']
        self.serial_num = int(line[6:11])
        self.name = line[12:16]
        self.res_name = line[17:20]
        self.res_full_name = line[16:20]
        self.chain_id = line[21]
        self.res_full_seq_num = line[22:27]
        self.res_seq_num = int(line[22:26])
        self.x_coord = float(line[30:38])
        self.y_coord = float(line[38:46])
        self.z_coord = float(line[46:54])
        self.coord = [self.x_coord, self.y_coord, self.z_coord]
        self.line = line

    def get_coord(self):
        return [self.x_coord, self.y_coord, self.z_coord]

    def __repr__(self):
        full_id = (self.get_id(), self.serial_num)
        return "<Atom %s %s>" % full_id

    def get_id(self):
        return self.name

class Model(object):
    """
    An encapsulation of what needs to be provided as input for training and testing the neural networks.
    """

    def __init__(self, pdb_name=""):
        self.cdrs = {name: [] for name in cdr_names}  # cdr chains
        self.agatoms = []
        self.cdratoms = []
        self.ag_names = []  # chains names
        self.ag = {}
        self.ab_h_chain = None
        self.ab_l_chain = None
        self.ag_chain = None
        self.pdb_name = pdb_name

    def get_ag(self):
        return self.ag

    def get_cdrs(self):
        return self.cdrs

    def get_agatoms(self):
        return self.agatoms

    def save_pdb_ag(self, path):
        with open(path + self.pdb_name + '_ag.pdb', "w") as f_ag:
            for atom in self.agatoms:
                f_ag.write(atom.line)
        print("Saved ", path + self.pdb_name + '_ag.pdb')

    def save_pdb_cdr(self, path):
        with open(path + self.pdb_name + '_cdr.pdb', "w") as f_cdr:
            for atom in self.cdratoms:
                f_cdr.write(atom.line)
        print("Saved ", path + self.pdb_name + '_cdr.pdb')

    def add_cdratom(self, cdr_atom):
        if cdr_atom not in self.cdratoms:
            self.cdratoms.append(cdr_atom)

    def add_agatom(self, ag_atom):
        if ag_atom not in self.agatoms:
            self.agatoms.append(ag_atom)

    def cdr_list_has_res(self, res_list, res_name, res_full_seq_num):
        for res in res_list:
            if res.full_seq_num == res_full_seq_num and res.get_name() == res_name:
                return res
        return None

    def ag_list_has_res(self, ag_res_list, res_name, res_full_seq_num):
        for res in ag_res_list:
            if res.full_seq_num == res_full_seq_num and res.get_name() == res_name:
                return res
        return None

    def agatoms_list_has_atom(self, ag_atom):
        return ag_atom in self.agatoms

    def add_residue(self, res, cdr_name):
        if res not in self.cdrs[cdr_name]:
            self.cdrs[cdr_name].append(res)

    def add_ag_residue(self, res, ag_name):
        if res not in self.ag[ag_name]:
            self.ag[ag_name].append(res)

    def add_atom_to_ab_h_chain(self, atom):
        if self.ab_h_chain == None:
            self.ab_h_chain = Chain(atom.chain_id)
        already_exists = False
        for res in self.ab_h_chain.child_list:
            if atom.res_full_seq_num == res.full_seq_num:
                already_exists = True
                res.child_list.append(atom)
                #res.add_atom(atom)
        if not already_exists:
            #res = Residue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num,
            #              atom.x_coord, atom.y_coord, atom.z_coord)
            res = Residue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num)
            res.child_list.append(atom)
            self.ab_h_chain.child_list.append(res)

    def add_atom_to_ab_l_chain(self, atom):
        if self.ab_l_chain == None:
            self.ab_l_chain = Chain(atom.chain_id)
        already_exists = False
        for res in self.ab_l_chain.child_list:
            if atom.res_full_seq_num == res.full_seq_num:
                already_exists = True
                res.child_list.append(atom)
                #res.add_atom(atom)
        if not already_exists:
            # res = Residue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num,
            #              atom.x_coord, atom.y_coord, atom.z_coord)
            res = Residue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num)
            res.child_list.append(atom)
            self.ab_l_chain.child_list.append(res)

    def add_atom_to_ag_chain(self, atom):
        if self.ag_chain == None:
            self.ag_chain = Chain(atom.chain_id)
        already_exists = False
        for res in self.ag_chain.child_list:
            if atom.res_full_seq_num == res.full_seq_num:
                already_exists = True
                res.child_list.append(atom)
                #res.add_atom(atom)
        if not already_exists:
            #res = AGResidue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num,
            #              atom.x_coord, atom.y_coord, atom.z_coord)
            res = Residue(atom.res_name, atom.res_seq_num, atom.res_full_name, atom.res_full_seq_num)
            res.child_list.append(atom)
            self.ag_chain.child_list.append(res)

"""""
 get antigen_chain by:
 get field from csv - let's say X|Y;
 search in .pdb where atom && chain_id = X or hetatm && chain_id.startsWith(X)
"""

def get_pdb_structure(pdb_file_name, ab_h_chain, ab_l_chain, ag_chain):
    """
    Extracting the data from PDB files.
    :param pdb_file_name: PDB file name
    :param ab_h_chain: Name of antibody heavy chain in that PDB file.
    :param ab_l_chain: Name of the anitbody light chain in that PDB file.
    :param ag_chain: Name of the antigen chain(s) in that PDB file.
    :return: cdrs, antigen atoms, antigen amino acids.
    """
    in_file = open(pdb_file_name, 'r')
    pdb_name = os.path.basename(pdb_file_name)
    model = Model(pdb_name[:-4])

    if " | " in ag_chain:
        c1, c2 = ag_chain.split(" | ")
        model.ag_names.append(c1)
        model.ag_names.append(c2)
    else:
        model.ag_names.append(ag_chain)
    for name in model.ag_names:
        print("name", name)

    model.ag = {name: [] for name in model.ag_names}
    ag = model.get_ag()
    cdrs = model.get_cdrs()

    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):

            atom = Atom(line)
            res_name = atom.res_name
            res_full_name = atom.res_full_name
            res_seq_num = atom.res_seq_num
            res_full_seq_num = atom.res_full_seq_num
            chain_id = atom.chain_id
            chain_id = chain_id

            if res_full_name[0] == 'A' or res_full_name[0] == " ":
                if chain_id == ab_h_chain:
                    model.add_atom_to_ab_h_chain(atom)
                    model.add_cdratom(atom)

                if chain_id == ab_l_chain:
                    model.add_atom_to_ab_l_chain(atom)
                    model.add_cdratom(atom)

                if chain_id == ag_chain:
                    model.add_atom_to_ag_chain(atom)


                for cdr_name in cdrs:
                    cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
                    cdr_range = range(-NUM_EXTRA_RESIDUES + cdr_low, cdr_hi +
                                      NUM_EXTRA_RESIDUES + 1)
                    if ((chain_id == ab_h_chain and cdr_name.startswith('H'))\
                        or (chain_id == ab_l_chain and cdr_name.startswith('L'))) \
                                    and res_seq_num in cdr_range:
                        residue = model.cdr_list_has_res(cdrs[cdr_name], res_name, res_full_seq_num)
                        if residue is None:
                            #residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num,
                            #                  atom.x_coord, atom.y_coord, atom.z_coord)
                            residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num)
                        residue.add_child(atom)
                        #residue.add_atom(atom)
                        model.add_residue(residue, cdr_name)
                if " | " in ag_chain:
                    c1, c2 = ag_chain.split(" | ")
                    if chain_id == c1:
                        model.add_agatom(atom)
                        residue = model.ag_list_has_res(ag[c1], res_name, res_full_seq_num)
                        if residue is None:
                            #residue = AGResidue(res_name, res_seq_num, res_full_name, res_full_seq_num,
                            #                  atom.x_coord, atom.y_coord, atom.z_coord)
                            residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num)
                        residue.add_child(atom)
                        #residue.add_atom(atom)
                        model.add_ag_residue(residue, c1)
                    if chain_id == c2:
                        model.add_agatom(atom)
                        residue = model.ag_list_has_res(ag[c2], res_name, res_full_seq_num)
                        if residue == None:
                            residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num)
                        residue.add_child(atom)
                        model.add_ag_residue(residue, c2)

                else:
                    if chain_id == ag_chain:
                        model.add_agatom(atom)
                        residue = model.ag_list_has_res(ag[chain_id], res_name, res_full_seq_num)
                        if residue is None:
                            #residue = AGResidue(res_name, res_seq_num, res_full_name, res_full_seq_num,
                            #                  atom.x_coord, atom.y_coord, atom.z_coord)
                            residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num)
                        residue.add_child(atom)
                        #residue.add_atom(atom)
                        model.add_ag_residue(residue, chain_id)

    return model.cdratoms, cdrs, model.agatoms, ag, model.ag_names, model



def get_pdb_structure_without_ag(pdb_file_name, ab_h_chain, ab_l_chain):
    in_file = open(pdb_file_name, 'r')
    model = Model()

    cdrs = model.get_cdrs()

    #print("new cdrs", cdrs)
    #print(ab_h_chain, ab_l_chain, ag_chain)
    for line in in_file:
        if line.startswith('ATOM') or line.startswith('HETATM'):

            atom = Atom(line)
            res_name = atom.res_name
            res_full_name = atom.res_full_name
            res_seq_num = atom.res_seq_num
            res_full_seq_num = atom.res_full_seq_num
            chain_id = atom.chain_id
            chain_id = chain_id
            #print("res_name", res_name, file=f)
            #print("res_full_name", res_full_name, file=f)
            if res_full_name[0] == 'A' or res_full_name[0] == " ":
                if chain_id == ab_h_chain:
                    model.add_atom_to_ab_h_chain(atom)

                if chain_id == ab_l_chain:
                    model.add_atom_to_ab_l_chain(atom)

                for cdr_name in cdrs:
                    cdr_low, cdr_hi = chothia_cdr_def[cdr_name]
                    cdr_range = range(-NUM_EXTRA_RESIDUES + cdr_low, cdr_hi +
                                      NUM_EXTRA_RESIDUES + 1)
                    if ((chain_id == ab_h_chain and cdr_name.startswith('H'))\
                        or (chain_id == ab_l_chain and cdr_name.startswith('L'))) \
                                    and res_seq_num in cdr_range:
                        residue = model.cdr_list_has_res(cdrs[cdr_name], res_name, res_full_seq_num)
                        if residue is None:
                            #residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num,
                            #                  atom.x_coord, atom.y_coord, atom.z_coord)
                            residue = Residue(res_name, res_seq_num, res_full_name, res_full_seq_num)
                        residue.add_child(atom)
                        #residue.add_atom(atom)
                        model.add_residue(residue, cdr_name)
    return cdrs

f = open('chains.txt','w')