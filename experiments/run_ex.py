import os
import random
import sys
import json
sys.path.append('../')
from src.train_results import FitResult

def run_experiment(run_name,trainer,dl_train,dl_test, out_dir='./experiments',early_stopping=None, checkpoints=None,num_epochs=50,print_every=10,**kw):

    fit_res = trainer.fit(dl_train,dl_test,num_epochs,checkpoints,early_stopping,print_every=print_every,**kw)
    save_experiment(run_name, out_dir, fit_res)

def save_experiment(run_name, out_dir, fit_res):
    output = dict(
        results=fit_res._asdict()
    )

    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)
    fit_res = FitResult(**output['results'])

    return fit_res
