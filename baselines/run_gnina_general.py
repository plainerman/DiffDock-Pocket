# small script to extract the ligand and save it in a separate file because GNINA will use the ligand position as
# initial pose
import os
import shutil
import subprocess
import sys

import time
from argparse import ArgumentParser, FileType
from datetime import datetime

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem, MolToPDBFile
from rdkit.Geometry import Point3D
from scipy.spatial.distance import cdist

from datasets.pdbbind import read_mol
from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt

parser = ArgumentParser()
parser.add_argument('--ligand_dir', type=str, default='data/difflinker_ZINC', help='')
parser.add_argument('--protein_path', type=str, default='data/difflinker_docking/3hz1_protein.pdb', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--results_path', type=str, default='results/difflinker_predictions', help='')
parser.add_argument('--seed_molecule_path', type=str, default='data/difflinker_docking/3hz1_ligand.pdb', help='Use the molecules at seed molecule path as initialization and only search around them')
parser.add_argument('--smina', action='store_true', default=False, help='')
parser.add_argument('--no_gpu', action='store_true', default=False, help='')
parser.add_argument('--exhaustiveness', type=int, default=8, help='')
parser.add_argument('--num_cpu', type=int, default=16, help='')
parser.add_argument('--pocket_mode', action='store_true', default=False, help='')
parser.add_argument('--pocket_cutoff', type=float, default=5, help='')
parser.add_argument('--num_modes', type=int, default=10, help='')
parser.add_argument('--autobox_add', type=int, default=4, help='')
parser.add_argument('--use_p2rank_pocket', action='store_true', default=False, help='')
parser.add_argument('--skip_p2rank', action='store_true', default=False, help='')
parser.add_argument('--prank_path', type=str, default='', help='')
parser.add_argument('--skip_existing', action='store_true', default=False, help='')
parser.add_argument('--flexdist', type=float, default=-1, help='If this is -1 then no sidechains are flexible')
parser.add_argument('--flex_max', type=int, default=-1, help='If this is -1 then no sidechains are flexible')



args = parser.parse_args()

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def log(*args):
    print(f'[{datetime.now()}]', *args)



if os.path.exists(args.results_path) and not args.skip_existing:
    shutil.rmtree(args.results_path)
os.makedirs(args.results_path, exist_ok=True)
sys.stdout = Logger(logpath=f'{args.results_path}/gnina.log', syspart=sys.stdout)
sys.stderr = Logger(logpath=f'{args.results_path}/error.log', syspart=sys.stderr)


all_times = []
start_time = time.time()
ligand_names = os.listdir(args.ligand_dir)
for i, ligand_name in enumerate(ligand_names):
    ligand_path = os.path.join(args.ligand_dir, ligand_name)
    os.makedirs(os.path.join(args.results_path, ligand_path), exist_ok=True)
    log('\n')
    log(f'complex {i} of {len(ligand_names)}')
    # call gnina to find binding pose
    rec_path = args.protein_path
    prediction_output_name = os.path.join(args.results_path, ligand_name)
    log_path = os.path.join(args.results_path, ligand_name + '.log')

    single_time = time.time()

    log(f'processing {ligand_path}')
    return_code = subprocess.run(f"gnina --receptor {rec_path} --ligand {ligand_path} --num_modes {args.num_modes} -o {prediction_output_name} {'--no_gpu' if args.no_gpu else ''} --autobox_ligand {args.seed_molecule_path} --autobox_add {args.autobox_add} --log {log_path} --exhaustiveness {args.exhaustiveness} --cpu {args.num_cpu} {'--cnn_scoring none' if args.smina else ''}",
            shell=True)
    log(return_code)
    all_times.append(time.time() - single_time)

    log("single time: --- %s seconds ---" % (time.time() - single_time))
    log("time so far: --- %s seconds ---" % (time.time() - start_time))
    log('\n')
log(all_times)
log("--- %s seconds ---" % (time.time() - start_time))
