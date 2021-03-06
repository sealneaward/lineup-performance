import pandas as pd
import re
import math
import urllib2
import numpy as np
import lxml.html as LH
from bs4 import BeautifulSoup

import lineup.config as CONFIG

PLAYER_RE = r'([A-Z].\s)\w+'

def _even_split(x, y):
	pos_y = y[y == 1]
	neg_y = y[y == -1]
	pos_x = x[y == 1]
	neg_x = x[y == -1]

	if len(pos_y) > len(neg_y):
		pos_y = pos_y[:len(neg_y) + 1]
		pos_x = pos_x[:len(neg_y) + 1]
	elif len(neg_y) > len(pos_y):
		neg_y = neg_y[:len(pos_y) + 1]
		neg_x = neg_x[:len(pos_y) + 1]

	# pos_inds = range(len(pos_y))
	# neg_inds = range(len(neg_y))
	# np.random.shuffle(pos_inds)
	# np.random.shuffle(neg_inds)
    #
	# pos_y = pos_y[pos_inds]
	# neg_y = neg_y[neg_inds]
	# pos_x = pos_x[pos_inds]
	# neg_x = neg_x[neg_inds]

	y = np.concatenate([pos_y, neg_y])
	x = np.concatenate([pos_x, neg_x])

	return x, y


def shuffle_2_array(x, y):
	randomize = np.arange(len(x))
	np.random.shuffle(randomize)
	x = x[randomize]
	y = y[randomize]
	return x, y

def _minute(play):
	"""
	Get int of minute (out of 48) that it occurs in
	"""
	minute = (12 - int(play['TIME'].split(':')[0])) + ((int(play['QUARTER']) - 1) * 12) - 1
	return minute

def _player_info(year):
	"""
	Get info on players for totals in single season

	Parameters
	----------
	year: str
		year id info

	Returns
	-------
	player_data: panda.DataFrame
		information on all members of home team

	"""
	# get home team id from game id
	url = ('https://www.basketball-reference.com/leagues/NBA_{YEAR}_totals.html').format(YEAR=year)
	player_data = pd.read_html(url)[0]
	# delete unecessary rows
	player_data = player_data.loc[player_data.Rk != 'Rk', :]
	# get soup table, then input to pandas.read_html function
	return player_data
	# else:
	#     # get player info from basketball_value folders
	#     player_data = pd.read_csv('%s/%s/%s' % (CONFIG.data.nba.matchups.bv.dir, year, 'players.csv'))
	#     player_data['Player'] = ''
	#
	#     # this data is really dirty and needs to delete duplicates and leave names with . in it
	#     duplicate_players = player_data[player_data.duplicated(subset=['PlayerID'], keep=False)]
	#     duplicate_players = duplicate_players.groupby(by=['PlayerID'])
	#     for x, duplicate_player in duplicate_players:
	#         for y, player in duplicate_player.iterrows():
	#             if '.' in player['PlayerName']:
	#                 # found the optimal name
	#                 break
	#             else:
	#                 continue
	#         player_data.loc[player_data.PlayerID == player['PlayerID'], 'PlayerName'] = player['PlayerName']
	#         player_data.loc[player_data.PlayerID == player['PlayerID'], 'Player'] = player['PlayerFirstName'] + ' ' + player['PlayerLastName']
	#
	#     player_data.drop_duplicates(subset=['PlayerID', 'PlayerName'], inplace=True)
	#     player_data['Player'] = player_data['PlayerFirstName'] + ' ' + player_data['PlayerLastName']
	#     return player_data

def roster(game):
	"""
	Parse team urls in game data to get home roster information in
	F. LastName format

	Parameters
	----------
	game: str
		game id info

	Returns
	-------
	home_roster: str list
		information on all members of home team

	"""
	# get home team id from game id
	team = game[-3:]
	url = ('https://www.basketball-reference.com/teams/{TEAM}/2016.html').format(TEAM=team)
	roster_info = pd.read_html(url)[0]
	home_roster = []

	for ind, player in roster_info.iterrows():
		try:
			first_name = str(player['Player']).split(' ')[0]
			last_name = str(player['Player']).split(' ')[1:]
			last_name = ' '.join(last_name)
			name = '%s. %s' % (first_name[0], last_name)
			home_roster.append(name)

		except Exception:
			continue

	return home_roster


def _game_id(game):
	"""
	Format game id for basketball reference format
	"""
	home = game[-3:]
	date = '%s0' % game[:-6]
	game = '%s%s' % (date, home)
	return game


def parse_nba_play(play, hm_roster):
	"""Parse play details from a play-by-play string describing a play.
	Assuming valid input, this function returns structured data in a dictionary
	describing the play. If the play detail string was invalid, this function
	returns None.
	:param details: detail string for the play
	:param is_hm: bool indicating whether the offense is at home
	:param returns: dictionary of play attributes or None if invalid
	:rtype: dictionary or None

	SOURCE: https://github.com/MikeRa1979/SportsScrape/blob/master/sportsref/nba/pbp.py
	"""
	if pd.isnull(play['SCORE']):
		return None
	elif pd.isnull(play['HOMEDESCRIPTION']):
		aw = True
		hm = False
		is_hm = False
		details = play['VISITORDESCRIPTION']
	else:
		hm = True
		aw = False
		is_hm = True
		details = play['HOMEDESCRIPTION']

	# if input isn't a string, return None
	if not details:
		return None


	p = {}
	p['detail'] = details
	p['home'] = hm
	p['away'] = aw
	p['is_home_play'] = is_hm
	p['minute'] = _minute(play)
	p['is_fga'] = None
	p['is_fgm'] = None
	p['is_three'] = None
	p['shot_dist'] = None
	p['is_assist'] = None
	p['assister'] = None
	p['is_block'] = None
	p['blocker'] = None
	p['is_reb'] = None
	p['is_oreb'] = None
	p['is_dreb'] = None
	p['rebounder'] = None
	p['is_fta'] = None
	p['is_ftm'] = None
	p['ft_shooter'] = None
	p['is_steal'] = None
	p['to_by'] = None
	p['is_steal'] = None
	p['stealer'] = None
	p['is_pf'] = None
	p['fouler'] = None

	# home roster
	# hm_roster = ['put team','roster', 'here']

	# parsing field goal attempts
	shotRE = (r'(?P<shooter>{0}) (?P<is_fgm>makes|misses) '
			  '(?P<is_three>2|3)\-pt shot').format(PLAYER_RE)
	distRE = r' (?:from (?P<shot_dist>\d+) ft|at rim)'
	assistRE = r' \(assist by (?P<assister>{0})\)'.format(PLAYER_RE)
	blockRE = r' \(block by (?P<blocker>{0})\)'.format(PLAYER_RE)
	shotRE = r'{0}{1}(?:{2}|{3})?'.format(shotRE, distRE, assistRE, blockRE)
	m = re.match(shotRE, details, re.IGNORECASE)
	if m:
		p['is_fga'] = True
		p.update(m.groupdict())
		p['is_fgm'] = p['is_fgm'] == 'makes'
		p['is_three'] = p['is_three'] == '3'
		p['is_assist'] = pd.notnull(p.get('assister'))
		p['is_block'] = pd.notnull(p.get('blocker'))

		return p

	# parsing rebounds
	rebRE = (r'(?P<is_oreb>Offensive|Defensive) rebound')
	rebounderRE = (r' (by (?P<rebounder>{0}))').format(PLAYER_RE)
	rebRE = r'{0}(?:{1})?'.format(rebRE, rebounderRE)
	m = re.match(rebRE, details, re.I)
	if m:
		p['is_reb'] = True
		p.update(m.groupdict())
		p['is_oreb'] = p['is_oreb'].lower() == 'offensive'
		p['is_dreb'] = not p['is_oreb']

		return p

	# parsing free throws
	ftRE = (r'(?P<ft_shooter>{0}) (?P<is_ftm>makes|misses) ').format(PLAYER_RE)
	m = re.match(ftRE, details, re.I)
	if m:
		p['is_fta'] = True
		p.update(m.groupdict())
		p['is_ftm'] = p['is_ftm'] == 'makes'
		return p

	# parsing turnovers
	toReasons = (r'(?P<to_type>[^;]+)(?:; steal by '
				 r'(?P<stealer>{0}))?').format(PLAYER_RE)
	toRE = (r'Turnover by (?P<to_by>{}|Team) '
			r'\((?:{})\)').format(PLAYER_RE, toReasons)
	m = re.match(toRE, details, re.I)
	if m:
		p['is_to'] = True
		p.update(m.groupdict())
		p['is_steal'] = p['stealer'] is not None
		if p['to_by'] == 'Team':
			p['off_team'] = hm if is_hm else aw
			p['def_team'] = aw if is_hm else hm
		else:
			to_home = p['to_by'] in hm_roster
			p['off_team'] = hm if to_home else aw
			p['def_team'] = aw if to_home else hm
		return p

	# parsing shooting fouls
	shotFoulRE = (r'Shooting(?P<is_pf> block)? foul by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(shotFoulRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing offensive fouls
	offFoulRE = (r'Offensive(?P<is_pf> charge)? foul '
				 r'by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(offFoulRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing personal fouls
	foulRE = (r'Personal foul by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(foulRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing loose ball fouls
	looseBallRE = (r'Loose ball foul by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(looseBallRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing away from play fouls
	awayFromBallRE = ((r'Away from play foul by (?P<fouler>{0})?').format(PLAYER_RE))
	m = re.match(awayFromBallRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing inbound fouls
	inboundRE = (r'Inbound foul by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(inboundRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing flagrant fouls
	flagrantRE = (r'Flagrant foul type (1|2) by (?P<fouler>{0})?').format(PLAYER_RE)
	m = re.match(flagrantRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	# parsing clear path fouls
	clearPathRE = (r'Clear path foul by (?P<fouler>{0})').format(PLAYER_RE)
	m = re.match(clearPathRE, details, re.I)
	if m:
		p['is_pf'] = True
		p.update(m.groupdict())
		return p

	return None


def parse_nhl_play(play):
	"""
	Parse play details from a play-by-play dataframe

	Parameters
	----------
	play: pandas.DataFrame
		play information

	Returns
	-------
	play: pandas.DataFrame
		parsed information specific to our problem
	"""

	if play['Strength'] != '5x5':
		return None
	elif play['Ev_Team'] == play['Away_Team']:
		aw = True
		hm = False
		is_hm = False
	elif play['Ev_Team'] == play['Home_Team']:
		hm = True
		aw = False
		is_hm = True
	else:
		return None

	# if input isn't a string, return None
	details = play['Description']
	if not details:
		return None


	p = {}
	p['detail'] = details
	p['home'] = hm
	p['away'] = aw
	p['is_home_play'] = is_hm
	p['second'] = play['Seconds_Elapsed']
	p['is_shot'] = None
	p['is_shot_on_goal'] = None
	p['is_goal'] = None
	p['is_assist'] = None
	p['is_block'] = None

	if play['Event'] == 'SHOT':
		p['is_shot'] = True
		p['is_shot_on_goal'] = True
		return p
	elif play['Event'] == 'MISS' or play['Event'] == 'BLOCK':
		p['is_shot'] = True
		p['is_shot_on_goal'] = False
		return p
	elif play['Event'] == 'GOAL':
		p['is_goal'] = True
		if 'Assists' in str(play['Description']):
			p['is_assist'] = True
		else:
			p['is_assist'] = False
		return p


	return None