"""get_abilities.py
Usage:
    get_abilities.py <f_data_config>

Arguments:
    <f_data_config>  example ''lineups.yaml''

Example:
    get_abilities.py lineups.yaml
"""

from __future__ import print_function

import pandas as pd
from docopt import docopt
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import lineup.config as CONFIG

def _abilities(lineup, matchups, type):
    """
    Get ratings for lineup
    """
    matchups = matchups.loc[matchups['%s_lineup' % type] == lineup]
    matchups['time_played'] = matchups['end_minute'] - matchups['starting_minute'] + 1
    team = matchups['%s_team' % (type)].values[0]
    n_games = len(matchups.loc[matchups['%s_team' % (type)] == team, 'game'].drop_duplicates(inplace=False))
    lineup = matchups[['%s_%s' % (type, ind) for ind in range(5)]].drop_duplicates(inplace=False)
    lineup = lineup.reset_index(drop=True)

    if type == 'away':
        ability_type = 'visitor'
    else:
        ability_type = 'home'

    time = matchups['time_played'].sum()
    fga = matchups['fga_%s' % ability_type].sum() / time * 100
    fgm = matchups['fgm_%s' % ability_type].sum() / time * 100
    fga_2 = matchups['fga_2_%s' % ability_type].sum() / time * 100
    fgm_2 = matchups['fgm_2_%s' % ability_type].sum() / time * 100
    fga_3 = matchups['fga_3_%s' % ability_type].sum() / time * 100
    fgm_3 = matchups['fgm_3_%s' % ability_type].sum() / time * 100
    ast = matchups['ast_%s' % ability_type].sum() / time * 100
    reb = matchups['reb_%s' % ability_type].sum() / time * 100
    dreb = matchups['dreb_%s' % ability_type].sum() / time * 100
    oreb = matchups['oreb_%s' % ability_type].sum() / time * 100
    pts = matchups['pts_%s' % ability_type].sum() / time * 100
    time = matchups['time_played'].sum() / n_games


    cols = ['fga', 'fgm', 'fga_2', 'fgm_2', 'fga_3', 'fgm_3', 'ast', 'reb', 'dreb', 'oreb', 'pts', 'time', 'n_games']
    cols = ['%s_%s' % (col, type) for col in cols]
    abilities = [fga, fgm, fga_2, fgm_2, fga_3, fgm_3, ast, reb, dreb, oreb, pts, time, n_games]

    abilities = pd.DataFrame(columns=cols, data=[abilities])
    abilities = abilities.reset_index(drop=True)

    lineup_abilities = pd.concat([lineup, abilities], axis=1)
    if lineup_abilities.isnull().values.any():
        return pd.DataFrame()

    return lineup_abilities


def _lineups(matchups):
    """
    Parse lineups for both home and away nodes
    """

    matchups['home_lineup'] = matchups['home_0'] + '-' + \
                   matchups['home_1'] + '-' + \
                   matchups['home_2'] + '-' + \
                   matchups['home_3'] + '-' + \
                   matchups['home_4']

    matchups['away_lineup'] = matchups['away_0'] + '-' + \
                   matchups['away_1'] + '-' + \
                   matchups['away_2'] + '-' + \
                   matchups['away_3'] + '-' + \
                   matchups['away_4']

    home_lineups = matchups['home_lineup'].drop_duplicates(inplace=False).values
    away_lineups = matchups['away_lineup'].drop_duplicates(inplace=False).values

    return home_lineups, away_lineups, matchups


def _matchup_abilities(matchups):
    """
    Using a set of defined abilities (off reb rate, offensive rate),
    create table of lineups with adjacent abilities for each matchup

    Parameters
    ----------
    matchups: pandas.DataFrame
        matchup information

    Returns
    -------
    abilities: pandas.DataFrame
        each individual lineup's abilities
    """
    home_abilities, away_abilities = pd.DataFrame(), pd.DataFrame()
    home_lineups, away_lineups, matchups = _lineups(matchups)

    for lineup in tqdm(home_lineups):
        lineup_abilities = _abilities(lineup, matchups, 'home')
        if lineup_abilities.isnull().values.any() or lineup_abilities.empty:
            continue
        home_abilities = home_abilities.append(lineup_abilities)

    for lineup in tqdm(away_lineups):
        lineup_abilities = _abilities(lineup, matchups, 'away')
        if lineup_abilities.isnull().values.any() or lineup_abilities.empty:
            continue
        away_abilities = away_abilities.append(lineup_abilities)

    return home_abilities, away_abilities

def clean():
    home_abilities = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'home_abilities.csv'))
    away_abilities = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'away_abilities.csv'))

    home_abilities = home_abilities.dropna(inplace=False)
    away_abilities = away_abilities.dropna(inplace=False)

    home_abilities.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'home_abilities.csv'), index=False)
    away_abilities.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'away_abilities.csv'), index=False)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    years = data_config['years']
    for year in years:
        matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-%s.csv' % year))
        home_abilities, away_abilities = _matchup_abilities(matchups)
        home_abilities.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'home_abilities-%s.csv' % year), index=False)
        away_abilities.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'away_abilities-%s.csv' % year), index=False)

    # clean()