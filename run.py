from utils import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [65, 127],
"batch_sizes" : [32],
"net" : ['UNet', 'AttUNet'],
"features" : [16],
"data_type": ['Locs', 'One', 'Four', 'BigLocs', 'BigOne', 'BigFour'],
"boundary_type":['D'],
"numerical_method":['fd', 'fv'],
"backward_type": ['cg'], 
"lr":[1e-3], 
"max_epochs":[150],
"ckpt":[False]}

log_dir = '../lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter, gpus="1")
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
    
