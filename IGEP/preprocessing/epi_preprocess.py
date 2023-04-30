import torch
# **Functions for preprocessing**

def change_lbls_required_form(lbls_list):
  new_list=[]
  for i in range(len(lbls_list)):
    lbl=lbls_list[i]
    new_lbl=lbl[0:lbl.size()[0]-1,1] #get last column
    new_lbl[new_lbl==-1] = 0 #turn -1 into 0
    new_list.append(new_lbl)
  return new_list

def change_lengths(lengths):
  new_list=[]
  for i in range(len(lengths)):
    new_list.append(lengths[i]-1)
  return new_list

def change_features(features_list):
  #get rid of final dummy node
  new_list=[]
  for i in range(len(features_list)):
    feature=features_list[i]
    feature=torch.unsqueeze(feature,dim=0)
    new_list.append(feature)
  return new_list

def create_dictionary_list(cdr_list,ag_list,edge_cdr_list,edge_ag_list,cdr_lbls,ag_lbls,dist_mats,coords_cdr,coords_ag,centers_coords_cdr,centers_coords_ag,distances_lbls_cdr,distances_lbls_ag):
  #Create a list of protein complexes where each protein complex is represented by a dictionary
  protein_list=[]
  for i in range(len(ag_list)):
    protein={'cdrs':cdr_list[i],'ags':ag_list[i],'edge_index_cdr':edge_cdr_list[i],'edge_index_ag':edge_ag_list[i],'cdr_lbls':cdr_lbls[i],'ag_lbls':ag_lbls[i],'dist_mat':dist_mats[i],'coords_cdr':coords_cdr[i],'coords_ag':coords_ag[i],'centeres_cdr':centers_coords_cdr[i],'centeres_ag':centers_coords_ag[i],'distances_lbls_cdr':distances_lbls_cdr[i],'distances_lbls_ag':distances_lbls_ag[i]}
    protein_list.append(protein)
  return protein_list

def add_coords_as_features(cdr_list,ag_list,coords_cdr,coords_ag):

  #Create a list of protein complexes where each protein complex is represented by a dictionary
  print('coordinates as features')
  cdr_list_feat = []
  ag_list_feat = []
  coords_cdr = change_features(coords_cdr)
  coords_ag = change_features(coords_ag)
  for i in range(len(ag_list)):
    concat_cdr=torch.cat((cdr_list[i],coords_cdr[i]), -1)
    concat_ag= torch.cat((ag_list[i],coords_ag[i]),-1)
    cdr_list_feat.append(concat_cdr)
    ag_list_feat.append(concat_ag)
  return  cdr_list_feat, ag_list_feat

def centering_coord(coords_ag_test, coords_ag_centers_test):
  #Create a list of protein complexes where each protein complex is represented by a dictionary
  centered_coords=[]
  print('centred_ coordinates')
  for i in range(len(coords_ag_test)):
    centered_coord  =coords_ag_test[i]-coords_ag_centers_test[i]
    centered_coords.append(centered_coord)
  return centered_coords