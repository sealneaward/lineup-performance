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
from lineup.data.utils import parse_nhl_play

class MatchupException(Exception):
	pass

def _cols(data_config):
	"""
	Get column names
	"""
	away_ids = ["away_%s" % id for id in list(range(6))]
	home_ids = ["home_%s" % id for id in list(range(6))]
	cols = ['game', 'season', 'home_team', 'away_team', 'starting_sec', 'end_sec']

	cols.extend(home_ids)
	cols.extend(away_ids)

	return cols

def _performance_vector(team_matchup_pbp, team):
	"""
	Get performance vector
	"""
	# TODO inspect, eveything is fucked
	shots = len(team_matchup_pbp.loc[team_matchup_pbp.is_shot == True, :])
	shots_on_goal = len(team_matchup_pbp.loc[team_matchup_pbp.is_shot_on_goal == True, :])
	goals = len(team_matchup_pbp.loc[team_matchup_pbp.is_goal == True, :])
	ast = len(team_matchup_pbp.loc[team_matchup_pbp.is_assist == True, :])
	blk = len(team_matchup_pbp.loc[team_matchup_pbp.is_block == True, :])

	cols = ['ast', 'blk', 'shots', 'shots_on_goal', 'goals']
	cols = ['%s_%s' % (col, team) for col in cols]
	data = [ast, blk, shots, shots_on_goal, goals]

	performance = pd.DataFrame(data=[data], columns=cols)
	return performance

def _performance(matchup, pbp):
	"""
	Get performance for single matchup
	"""
	starting_sec = matchup['starting_sec']
	end_sec = matchup['end_sec']

	matchup_pbp = pbp.loc[(pbp.second >= starting_sec) & (pbp.second <= end_sec), :]

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
			performances = performances.append(performance)

	performances = pd.concat([matchups, performances], axis=1)
	return performances

def _matchup(lineups, game, season, cols, time_start, time_end, home, away):
	"""
	Get lineup at time t
	"""
	lineup_ids = list(map(str, list(range(6))))
	if lineups.empty:
		raise MatchupException('no lineups at time')
	if not len(lineups) == 2:
		raise MatchupException('too many lineups at time')

	start_time = lineups.loc[:, time_start].max()
	end_time = lineups.loc[:, time_end].min()

	for ind, lineup in lineups.iterrows():
		if lineup['team'] == home:
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


def _pbp(game, pbp):
	"""
	Get hockey play-by-play by game id

	Parameters
	----------
	game: str
		hockey reference gameID
	pbp: pandas.DataFrame
		information on all plays
	Returns
	-------
	game_pbp: pandas.DataFrame
		information on all game plays
	"""
	pbp = pbp.drop(columns=['Unnamed: 0'], axis=1, inplace=False)
	pbp = pbp.loc[pbp.Game_Id == game, :]
	plays = []

	for ind, play in pbp.iterrows():
		play = parse_nhl_play(play)
		if play is None:
			continue
		else:
			plays.append(play)

	plays = pd.DataFrame(plays)
	home = pbp['Home_Team'].drop_duplicates(inplace=False).values[0]
	away = pbp['Away_Team'].drop_duplicates(inplace=False).values[0]
	return plays, home, away


def _game_matchups(lineups, game, season, cols, home, away):
	"""
	Get matchups for game
	"""

	time_start = 'starting_sec'
	time_end = 'end_sec'
	time_range = range(3600)

	matchups = pd.DataFrame()
	starting_lineups = lineups.loc[(lineups[time_start] <= 0) & (lineups[time_end] >= 0), :]
	try:
		current_matchup = _matchup(
			game=game,
			season=season,
			lineups=starting_lineups,
			cols=cols,
			time_start=time_start,
			time_end=time_end,
			home=home,
			away=away
		)
	except MatchupException:
		return pd.DataFrame()

	for time in time_range:
		try:
			time_lineups = lineups.loc[(lineups[time_start] <= time) & (lineups[time_end] >= time), :]
			matchup = _matchup(
				game=game,
				season=season,
				lineups=time_lineups,
				cols=cols,
				time_start=time_start,
				time_end=time_end,
				home=home,
				away=away
			)
			if not matchup.equals(current_matchup):
				# lineup change detected
				# record previous matchup
				matchups = matchups.append(current_matchup)
				current_matchup = matchup
		except MatchupException:
			continue

	matchups = matchups.append(current_matchup)

	return matchups


def _matchups(data_config, lineups, pbp):
	"""
	Form lineup matches for embedding purposes.
	For each minute form ten man lineup consisting of
	team A and team B at time T

	Parameters
	----------
	data_config: dict
		additional config setting
	lineups: pandas.DataFrame
		information on single team lineup at time T
	pbp: pandas.DataFrame
		information on all plays
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
		pbp_game, home, away = _pbp(game, pbp)
		game_lineups = lineups.loc[lineups.game == game, :]
		game_matchups = _game_matchups(lineups=game_lineups, cols=cols, game=game, season=season, home=home, away=away)
		if game_matchups.empty:
			continue
		game_matchups = _matchup_performances(matchups=game_matchups, pbp=pbp_game)
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

	lineups = pd.read_csv('%s/%s' % (CONFIG.data.nhl.lineups.dir, 'lineups-sec.csv'))
	pbp = pd.read_csv('%s/%s' % (CONFIG.data.nhl.lineups.dir, 'pbp.csv'))

	matchups = _matchups(data_config, lineups, pbp)
	matchups.to_csv('%s/%s' % (CONFIG.data.nhl.matchups.dir, 'matchups.csv'), index=False)
