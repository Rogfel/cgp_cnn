from pycgp import CGP, CGPES, Evaluator, MaskEvaluator
from genetics import cgp_functions
from pycgp.ipfunctions import *
import pandas as pd
import sys
import time

col=30
row=2
nb_ind=5
mutation_rate_nodes=0.15
mutation_rate_outputs=0.3
n_it=20
genome=None

library = cgp_functions.build_cnn()
# library = build_funcLib()

dirname = 'genetics/pyCGP-master/datasets/coins/'
# dirname = 'dataset/PetImages/'

dataset_name = 'dataset.csv'

e = MaskEvaluator(dirname = dirname,
                  dataset_name = dataset_name,
                  display_dataset=False,
                  resize = 0.25,
                  include_hsv = True,
                  include_hed = False,
                  number_of_evaluated_images=-1)

if genome is None:
    cgpFather = CGP.random(num_inputs=e.n_inputs, num_outputs=e.n_outputs, 
                           num_cols=col, num_rows=row, library=library, 
                           recurrency_distance=1.0, recursive=False, 
                           const_min=0, const_max=255, 
                           input_shape=e.input_channels[0][0].shape, dtype='uint8')
else:
    cgpFather = CGP.load_from_file(genome, library)

output_dirname = dirname+'evos/run_'+str(round(time.time() * 1000000))
print("Starting evolution. Genomes will be saved in: "+ output_dirname)
es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, output_dirname)
es.run(n_it)

es.father.to_function_string(['ch_'+str(i) for i in range(e.n_inputs)], ['mask_'+str(i) for i in range(e.n_outputs)])

e.evaluate(es.father, 0, True)