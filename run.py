from utils import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [33],
"batch_sizes" : [32],
"net" : ['UNet', 'AttUNet'],
"features" : [16],
"data_type": ['Locs'],
"boundary_type":['D'],
"numerical_method":['fd'],
"backward_type": ['cg'], 
"lr":[1e-3], 
"max_epochs":[50], 
"ckpt":[False]}

log_dir = '../lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter,)
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
