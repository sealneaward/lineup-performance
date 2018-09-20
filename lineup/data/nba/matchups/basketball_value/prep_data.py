"""prep_data.py
Usage:
	prep_data.py <f_data_config>

Arguments:
	<f_data_config>  example ''lineups.yaml''

Example:
	prep_data.py lineups.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml
from tqdm import tqdm
import os

import lineup.config as CONFIG
from lineup.data.utils import parse_nba_play, roster, _game_id

import warnings
warnings.filterwarnings('ignore')

class MatchupException(Exception):
    pass


def _clean_matchups(data_config, matchups):
    """
    Format matchup columns to follow those that are previously
    used in older scrips that use data from basketball-reference.
    """
    matchups = matchups[data_config['basketball_value'].keys()].rename(columns=data_config['basketball_value'])
    return matchups


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
    starting_min = matchup['starting_minute']
    end_min = matchup['end_minute']

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

    performances.set_index(matchups.index.values, inplace=True)
    performance_columns = list(performances.columns.values)
    matchup_columns = list(matchups.columns.values)
    columns = matchup_columns + performance_columns

    performances = pd.concat([matchups, performances], axis=1, ignore_index=True)
    performances.columns = columns

    return performances


def _pbp(game):
    """
    Format game_pbp to follow basketball-reference format
    """
    game = _game_id(game)

    try:
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
    except Exception as err:
        return pd.DataFrame()


def _game_matchups(data_config, matchups, game, season, cols):
    """
    Get matchups for game
    """
    time_start = 'starting_minute'
    time_end = 'end_minute'
    time_range = range(48)

    start_times = []
    end_times = []

    # change starting time and end time to minute format by rounding
    matchups.reset_index(drop=True)
    for ind, matchup in matchups.iterrows():
        start_time = matchup[time_start].split(':')[1:]
        start_time = 48 - int(round(float('%s.%s' % (start_time[0], start_time[1])), 0))

        end_time = matchup[time_end].split(':')[1:]
        end_time = 48 - int(round(float('%s.%s' % (end_time[0], end_time[1])), 0))

        start_times.append(start_time)
        end_times.append(end_time)

    matchups[time_start] = pd.Series(start_times).values
    matchups[time_end] = pd.Series(end_times).values

    return matchups


def _matchups(data_config, matchups, season):
    """
    Form lineup matches for embedding purposes.
    For each minute form ten man lineup consisting of team A and team B at time T

    Parameters
    ----------
    data_config: dict
        additional config setting
    matchups: pandas.DataFrame
        information on matchups for teams at time T
    """
    total_matchups = pd.DataFrame()
    cols = _cols(data_config)
    matchups = _clean_matchups(data_config, matchups)

    gameids = matchups.loc[:, 'game'].drop_duplicates(inplace=False).values

    missed_games = 0
    for game in tqdm(gameids):
        game_pbp = _pbp(game)
        if game_pbp.empty:
            missed_games += 1
            continue
        game_matchups = matchups.loc[matchups.game == game, :]
        game_matchups = _game_matchups(data_config=data_config, matchups=game_matchups, cols=cols, game=game, season=season)
        if game_matchups.empty:
            missed_games += 1
            continue
        game_matchups = _matchup_performances(matchups=game_matchups, pbp=game_pbp)
        if game_matchups.empty:
            missed_games += 1
            continue
        total_matchups = total_matchups.append(game_matchups)

    print('Missed %s games for season \n' % missed_games)

    total_matchups['home_team'] = total_matchups['game'].str[-3:]
    total_matchups['away_team'] = total_matchups['game'].str[-6:-3]
    return total_matchups

def clean_teams(year):
    matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-%s.csv' % year))
    matchups['home_team'] = matchups['game'].str[-3:]
    matchups['away_team'] = matchups['game'].str[-6:-3]
    matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-%s.csv' % year), index=False)

def eval_bv(matchups, year):
    performances = []
    matchups['Performance'] = 0
    for ind, matchup in matchups.iterrows():
        performance = {}
        if (int(matchup['PointsScoredHome']) - int(matchup['PointsScoredAway'])) > 0:
            performance['outcome'] = 1
        elif (int(matchup['PointsScoredHome']) - int(matchup['PointsScoredAway'])) <= 0:
            performance['outcome'] = -1
        performances.append(performance)

    performances = pd.DataFrame.from_records(performances)
    n_pos = len(performances.loc[performances.outcome == 1, :])
    n_neg = len(performances.loc[performances.outcome == -1, :])

    print('Percent of positives in %s: %s' % (year, str(float((n_pos * 1.0) / (n_pos + n_neg)))))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    bv_dir = '%s/%s' % (CONFIG.data.nba.matchups.dir, 'basketball_value')
    years = [x[0] for x in os.walk(bv_dir)][1:]
    years = [year.split('/')[-1] for year in years]
    for year in years:
        matchups = pd.read_csv('%s/%s/matchups.csv' % (bv_dir, year), sep='\t')
        # eval_bv(matchups, year)

        # # pbp = pd.read_csv('%s/%s/playbyplay.csv' % (bv_dir, year), sep='\t')
        matchups = _matchups(data_config, matchups, year)
        matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-%s.csv' % year), index=False)

        clean_teams(year)