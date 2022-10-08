from utils import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [129],
"batch_sizes" : [16],
"net" : ['UNet'],
"features" : [32],
"data_type": ['BigOne'],
"boundary_type":['D', 'N'],
"numerical_method":['fd'],
"input_type":['F1', 'F2'],
"backward_type": ['jac', 'cg'], "lr":[1e-3], "max_epochs":[150], "ckpt":[False]}

log_dir = '../lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter,)
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
