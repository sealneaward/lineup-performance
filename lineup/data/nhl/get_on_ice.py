"""get_on_ice.py
Usage:
    get_on_ice.py <f_data_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''

Example:
    get_on_ice.py lineups.yaml
"""

from docopt import docopt
import hockey_scraper
import yaml

import lineup.config as CONFIG

# Scrapes the 2015 & 2016 season with shifts and stores the data in a Csv file
def scrape_seasons(data_config):
    """
    Scrape seasons

    Parameters
    ----------
    data_config: yaml config
    """
    for season in data_config['years']:
        hockey_scraper.scrape_seasons([int(season)], True)

if __name__ == '__main__':
	arguments = docopt(__doc__)
	print("...Docopt... ")
	print(arguments)
	print("............\n")

	f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
	data_config = yaml.load(open(f_data_config, 'rb'))
	scrape_seasons(data_config)
