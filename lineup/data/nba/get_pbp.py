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
from tqdm import tqdm

import lineup.config as CONFIG
from lineup.data.utils import parse_nba_play, roster

def _pbp(game):
	"""
	Scrape basketball reference game play-by-play by ID
	Args:
		game_ID (str): bball reference gameID
	Returns: None
		pickles pbp DataFrame to data directory
	"""
	url = ('http://www.basketball-reference.com/boxscores/pbp/{ID}.html').format(ID=game)
	pbp = pd.read_html(url)[0]
	pbp.columns = pbp.iloc[1]
	pbp.columns = ['TIME', 'VISITORDESCRIPTION', 'VISITORRESULTS', 'SCORE', 'HOMERESULTS', 'HOMEDESCRIPTION']
	pbp = pbp.drop(pbp.index[1])
	pbp['QUARTER'] = pbp.TIME.str.extract('(.*?)(?=Q)', expand=False).str[0]
	pbp['QUARTER'] = pbp['QUARTER'].fillna(method='ffill')
	pbp['GAME'] = game
	pbp = pbp.loc[~pbp.TIME.isin(['Time', '1st Q', '2nd Q', '3rd Q', '4th Q']), :]

	hm_roster = roster(game)

	plays = []

	for ind, play in pbp.iterrows():
		play = parse_nba_play(play, hm_roster)
		if play is None:
			continue
		else:
			plays.append(play)

	plays = pd.DataFrame(plays)
	plays['game'] = game
	plays['season'] = year
	return plays

def season_pbp(data_config, lineups, year):
	"""
	Get the play by play data

	Parameters
	----------
	data_config: dict
		additional config setting
	lineups: pandas.DataFrame
		information on single team lineup at time T
	year: str
	"""
	pbp_season = pd.DataFrame()
	gameids = lineups.loc[:, 'game'].drop_duplicates(inplace=False).values

	for game in tqdm(gameids):
		try:
			pbp_game = _pbp(game)
			pbp_season = pbp_season.append(pbp_game)
		except Exception:
			continue

	return pbp_season


if __name__ == '__main__':
	arguments = docopt(__doc__)
	print("...Docopt... ")
	print(arguments)
	print("............\n")

	f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
	data_config = yaml.load(open(f_data_config, 'rb'))

	years = data_config['years']
	for year in years:
		lineups = pd.read_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'lineups-%s.csv' % year))
		pbp_season = season_pbp(data_config, lineups, year)
		pbp_season.to_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'pbp-%s.csv' % year), index=False)
