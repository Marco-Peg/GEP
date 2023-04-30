"""
Constants used in the implemented models.
Includes values for input sizes, output sizes and hyperparameters.
"""
import torch

MAX_CDR_LENGTH = 32

MAX_AG_LENGTH = 1269

use_cuda = torch.cuda.is_available()

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
chothia_cdr_def = {
    "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }
cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]

aa_s = "CSTPAGNDEQHRKMILVFYWU" # U for unknown

NUM_EXTRA_RESIDUES = 2 # The number of extra residues to include on the either side of a CDR
CONTACT_DISTANCE = 4.5 # Contact distance between atoms in Angstroms
RES_DISTANCE= 10
AG_DISTANCE = 15

NUM_FEATURES = len(aa_s) + 7 +3 # one-hot + extra features + chain one-hot
AG_NUM_FEATURES = len(aa_s) + 7+ 3

NUM_ITERATIONS = 10
NUM_SPLIT = 10

epochs = 20

batch_size = 32

visualisation_pdb_number = 0
visualisation_flag = False
DATA_DIRECTORY = 'data/'
PDBS_FORMAT = 'data/{}.pdb'
CSV_NAME = 'sabdab_27_jun_95_90.csv'


visualisation_pdb = "4bz1"

visualisation_pdb_file_name = PDBS_FORMAT.format(visualisation_pdb)

vis_dataset_file = "visualisation-dataset.p"

track_f = open("track_file.txt", "w")

print_file = open("open_cv.txt", "w")
prob_file = open("prob_file.txt", "w")
data_file = open("dataset.txt", "w")

monitoring_file = open("monitor.txt", "w")

indices_file = open("indices.txt", "w")

sort_file = open("sort_file.txt", "w")

attention_file = open("attention.txt", "w")

visualisation_file = open("visualisation.txt", "w")
