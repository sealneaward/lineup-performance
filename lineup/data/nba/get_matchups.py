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
from tqdm import tqdm

import lineup.config as CONFIG
from lineup.data.utils import parse_nba_play, roster

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

def _performance_vector(team_matchup_pbp, team):
	"""
	Get performance vector
	"""
	fga = len(team_matchup_pbp.loc[team_matchup_pbp.is_fga == True, :])
	fta = len(team_matchup_pbp.loc[team_matchup_pbp.is_fta == True, :])
	fgm = len(team_matchup_pbp.loc[team_matchup_pbp.is_fgm == True, :])
	fga_2 = len(team_matchup_pbp.loc[(team_matchup_pbp.is_fga == True) & (team_matchup_pbp.is_three == False), :])
	fgm_2 = len(team_matchup_pbp.loc[(team_matchup_pbp.is_fgm == True) & (team_matchup_pbp.is_three == False), :])
	fga_3 = len(team_matchup_pbp.loc[(team_matchup_pbp.is_fga == True) & (team_matchup_pbp.is_three == True), :])
	fgm_3 = len(team_matchup_pbp.loc[(team_matchup_pbp.is_fgm == True) & (team_matchup_pbp.is_three == True), :])
	ast = len(team_matchup_pbp.loc[team_matchup_pbp.is_assist == True, :])
	blk = len(team_matchup_pbp.loc[team_matchup_pbp.is_block == True, :])
	pf = len(team_matchup_pbp.loc[team_matchup_pbp.is_pf == True, :])
	reb = len(team_matchup_pbp.loc[team_matchup_pbp.is_reb == True, :])
	dreb = len(team_matchup_pbp.loc[team_matchup_pbp.is_dreb == True, :])
	oreb = len(team_matchup_pbp.loc[team_matchup_pbp.is_oreb == True, :])
	to = len(team_matchup_pbp.loc[team_matchup_pbp.is_to == True, :])
	pts = fgm_2 * 2 + fgm_3 * 3
	if fga > 0:
		pct = (1.0 * fgm)/fga
	else:
		pct = 0.0
	if fga_2 > 0:
		pct_2 = (1.0 * fgm_2) / fga_2
	else:
		pct_2 = 0.0
	if fga_3 > 0:
		pct_3 = (1.0 * fgm_3) / fga_3
	else:
		pct_3 = 0.0

	cols = ['fga', 'fta', 'fgm', 'fga_2', 'fgm_2', 'fga_3', 'fgm_3', 'ast', 'blk', 'pf', 'reb', 'dreb', 'oreb', 'to', 'pts', 'pct', 'pct_2', 'pct_3']
	cols = ['%s_%s' % (col, team) for col in cols]
	data = [fga, fta, fgm, fga_2, fgm_2, fga_3, fgm_3, ast, blk, pf, reb, dreb, oreb, to, pts, pct, pct_2, pct_3]

	performance = pd.DataFrame(data=[data], columns=cols)
	return performance

def _performance(matchup, pbp):
	"""
	Get performance for single matchup
	"""
	starting_min = matchup['starting_min']
	end_min = matchup['end_min']

	matchup_pbp = pbp.loc[(pbp.minute >= starting_min) & (pbp.minute <= end_min), :]

	# get totals for home
	team_matchup_pbp = matchup_pbp.loc[matchup_pbp.home == True, :]
	performance_home = _performance_vector(team_matchup_pbp, 'home')

	# get totals for visitor
	team_matchup_pbp = matchup_pbp.loc[matchup_pbp.home == False, :]
	performance_away = _performance_vector(team_matchup_pbp, 'visitor')

	performance = pd.concat([performance_home, performance_away], axis=1)

	return performance


def _matchup_performances(matchups, pbp):
	"""
	Create performance vectors for each of the matchups

	Parameters
	----------
	matchups: pandas.DataFrame
		time in/out of lineup matchups
	pbp: pandas.DataFrame
		events in game with timestamps

	Returns
	-------
	matchups_performance: pandas.DataFrame
		performance vectors
	"""
	performances = pd.DataFrame()

	for ind, matchup in matchups.iterrows():
		performance = _performance(matchup, pbp)
		if not performance.empty:
			if (int(performance['pts_home']) - int(performance['pts_visitor'])) > 0:
				performance['outcome'] = 1
			elif (int(performance['pts_home']) - int(performance['pts_visitor'])) <= 0:
				performance['outcome'] = -1
			performances = performances.append(performance)

	performances = pd.concat([matchups, performances], axis=1)
	return performances

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

	hm_roster = roster(game)

	plays = []

	for ind, play in pbp.iterrows():
		play = parse_nba_play(play, hm_roster)
		if play is None:
			continue
		else:
			plays.append(play)

	plays = pd.DataFrame(plays)
	return plays


def _game_matchups(data_config, lineups, game, season, cols):
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
	try:
		current_matchup = _matchup(game=game, season=season, lineups=starting_lineups, cols=cols, time_start=time_start, time_end=time_end)
	except MatchupException:
		return pd.DataFrame()

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
	# if data_config['gameids'] is not None:
	# 	gameids = lineups.loc[lineups.game.isin(data_config['gameids']), 'game'].drop_duplicates(inplace=False).values
	# else:
	# 	gameids = lineups.loc[:, 'game'].drop_duplicates(inplace=False).values

	gameids = lineups.loc[:, 'game'].drop_duplicates(inplace=False).values

	for game in tqdm(gameids):
		pbp = _pbp(game)
		game_lineups = lineups.loc[lineups.game == game, :]
		game_matchups = _game_matchups(lineups=game_lineups, cols=cols, game=game, season=season)
		if game_matchups.empty:
			continue
		game_matchups = _matchup_performances(matchups=game_matchups, pbp=pbp)
		if game_matchups.empty:
			continue
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
		lineups = pd.read_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'lineups-min.csv'))
	else:
		lineups = pd.read_csv('%s/%s' % (CONFIG.data.nba.lineups.dir, 'lineups-sec.csv'))

	matchups = _matchups(data_config, lineups)
	matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups.csv'), index=False)
