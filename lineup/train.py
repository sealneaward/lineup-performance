"""train.py
Usage:
    train.py <f_data_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''

Example:
    train.py lineups.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml


import lineup.config as CONFIG
from lineup.model.previous import Previous as model

def train(data_config, data):
    previous = model(data_config)
    previous.prep_data(data=data)
    previous.train()


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))
    matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'lineups-min.csv'))

    train(data_config, matchups)