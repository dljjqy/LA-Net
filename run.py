from utils import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [257],
"batch_sizes" : [8],
"net" : ['AttUNet', 'UNet'],
"features" : [16],
"data_type": ['Locs', 'BigLocs', 'One', 'Four', 'BigOne', 'BigFour'],
"boundary_type":[ 'D', 'N'],
"numerical_method":['fv', 'fd'],
"backward_type": ['cg'], 
"lr":[1e-3], 
"max_epochs":[150], 
"ckpt":[False]}

log_dir = '../lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter,)
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
