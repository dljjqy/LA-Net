from utils import *
from itertools import product
from pathlib import Path

hyper_parameters_dict = {
"grid_sizes" : [129, 257, 513],
"batch_sizes" : [8],
"net" : ['UNet'],
"features" : [16, 32],
"data_type": ['One', 'Four', 'BigOne', 'BigFour'],
"boundary_type":['D', 'N'],
"input_type":['F'],
"backward_type": ['conv', 'jac', 'cg' ], "lr":[1e-3], "max_epochs":[150], "ckpt":[False]}

log_dir = '../lightning_logs/'

for parameter in product(*hyper_parameters_dict.values()):
    case = gen_hyper_dict(*parameter,)
    path = Path(f"{log_dir}{case['name']}")
    if not path.exists():
        print(f"\nExperiment Name: {case['name']}\n")
        main(case)
