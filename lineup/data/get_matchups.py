"""get_matchups.py
Usage:
    get_matchups.py <f_data_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''

Example:
    get_matchups.py lineups.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml

import lineup.config as CONFIG


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))