"""get_lineups.py
Usage:
    get_lineups.py <f_data_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''

Example:
    get_lineups.py lineups.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml

import lineup.config as CONFIG
from lineup.data.utils import parse_play

class MatchupException(Exception):
	pass

def _cols(data_config):
	"""
	Get column names
	"""
	away_ids = ["away_%s" % id for id in list(range(5))]
	home_ids = ["home_%s" % id for id in list(range(5))]

	if data_config['time_seperator'] == 'min':
		cols = ['game', 'season', 'home_team', 'away_team', 'starting_min', 'end_min']
	else:
		cols = ['game', 'season', 'home_team', 'away_team', 'starting_sec', 'end_sec']

	cols.extend(home_ids)
	cols.extend(away_ids)

	return cols


def _matchup(lineups, game, season, cols, time_start, time_end):
	"""
	Get lineup at time t
	"""
	lineup_ids = map(str, list(range(5)))
	if lineups.empty:
		raise MatchupException('no lineups at time')
	if not len(lineups) == 2:
		raise MatchupException('too many lineups at time')

	start_time = lineups.loc[:, time_start].max()
	end_time = lineups.loc[:, time_end].min()

	for ind, lineup in lineups.iterrows():
		home_id = lineup['game'][-3:]
		if lineup['team'] == home_id:
			home_team = lineup['team']
			home_lineup = lineup
			home_players = home_lineup[lineup_ids].values
		else:
			away_team = lineup['team']
			away_lineup = lineup
			away_players = away_lineup[lineup_ids].values

	data = [game, season, home_team, away_team, start_time, end_time]
	data.extend(home_players)
	data.extend(away_players)
	matchup = pd.DataFrame(data=[data], columns=cols)

	return matchup


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

	for ind, play in pbp.iterrows():
		play = parse_play(play)
		if not play.empty:
			pbp.iloc[ind] = play

	return pbp


def _game_matchups(lineups, game, season, cols):
	"""
	Get matchups for game
	"""

	if data_config['time_seperator'] == 'min':
		time_start = 'starting_minute'
		time_end = 'end_minute'
		time_range = range(48)
	else:
		time_start = 'starting_sec'
		time_end = 'end_sec'
		time_range = range(2880)

	matchups = pd.DataFrame()
	starting_lineups = lineups.loc[(lineups[time_start] <= 0) & (lineups[time_end] >= 0), :]
	current_matchup = _matchup(game=game, season=season, lineups=starting_lineups, cols=cols, time_start=time_start, time_end=time_end)

	for time in time_range:
		try:
			time_lineups = lineups.loc[(lineups[time_start] <= time) & (lineups[time_end] >= time), :]
			matchup = _matchup(game=game, season=season, lineups=time_lineups, cols=cols, time_start=time_start, time_end=time_end)
			if not matchup.equals(current_matchup):
				# lineup change detected
				# record previous matchup
				matchups = matchups.append(current_matchup)
				current_matchup = matchup
		except MatchupException:
			continue

	matchups = matchups.append(current_matchup)

	return matchups


def _matchups(data_config, lineups):
	"""
	Form lineup matches for embedding purposes.
	For each minute form ten man lineup consisting of team A and team B at time T

	Parameters
	----------
	data_config: dict
		additional config setting
	lineups: pandas.DataFrame
		information on single team lineup at time T
	"""
	matchups = pd.DataFrame()
	cols = _cols(data_config)

	# debugging purposes
	season = '2016'
	if data_config['gameid'] is not None:
		gameids = [lineups.loc[lineups.game == data_config['gameid'], 'game'].values[0], '']
	else:
		gameids = lineups.loc[:, 'game'].drop_duplicates(inplace=False).values

	for game in gameids:
		pbp = _pbp(game)
		game_lineups = lineups.loc[lineups.game == game, :]
		game_matchups = _game_matchups(lineups=game_lineups, cols=cols, game=game, season=season)
		matchups = matchups.append(game_matchups)

	return matchups


if __name__ == '__main__':
	arguments = docopt(__doc__)
	print("...Docopt... ")
	print(arguments)
	print("............\n")

	f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
	data_config = yaml.load(open(f_data_config, 'rb'))

	if data_config['time_seperator'] == 'min':
		lineups = pd.read_csv('%s/%s' % (CONFIG.data.lineups.dir, 'lineups-min.csv'))
	else:
		lineups = pd.read_csv('%s/%s' % (CONFIG.data.lineups.dir, 'lineups-sec.csv'))

	matchups = _matchups(data_config, lineups)
	matchups.to_csv('%s/%s' % (CONFIG.data.matchups.dir, 'lineups-min.csv'), index=False)