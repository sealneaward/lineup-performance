''"""train_previous.py
Usage:
    train_previous.py <f_data_config> <f_model_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''
    <f_data_config>  example ''abilities.yaml''

Example:
    train_previous.py lineups.yaml abilities.yaml
    train_previous.py lineups.yaml previous.yaml
    train_previous.py lineups.yaml adjusted.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml
from sklearn.metrics import classification_report
import importlib

import lineup.config as CONFIG
from lineup.model.abilities import Abilities
from lineup.model.previous import Previous
from lineup.model.adjusted import Adjusted

def train(data_config, model_config, data):
    model = getattr(importlib.import_module(model_config['model']['module']), model_config['model']['model'])(
        data_config=data_config,
        model_config=model_config,
        data=data
    )
    # model.prep_data(data=data, home_abilities=home_abilities, away_abilities=away_abilities)
    model.train()
    predictions = model.model.predict(model.val_x)
    print(classification_report(model.val_y, predictions))

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))
    f_model_config = '%s/%s' % (CONFIG.model.config.dir, arguments['<f_model_config>'])
    model_config = yaml.load(open(f_model_config, 'rb'))
    matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'lineups-min.csv'))

    train(data_config=data_config, data=matchups, model_config=model_config)