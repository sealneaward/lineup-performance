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

class LineupFormationException(Exception):
    pass


def _second_ranges(player):
    seconds_count = [0.0] * 3600
    for ind, r in player.iterrows():
        for i in range(int(r['Start'] + ((r['Period'] - 1) * 1200)), int(r['End'] + ((r['Period'] - 1) * 1200))):
            seconds_count[i] += 1.0
            if seconds_count[i] > 1:
                print('Stop')
    return seconds_count


def _form_lineup(lineups, lineup, team, game, season, starting_second, end_second, cols):
    try:
        assert len(lineup) == 6
    except Exception:
        raise LineupFormationException('Incorrect number of people in lineup - probably penalty kill')
    lineup = lineup.loc[:, 'name'].values
    data = [team, game, season, starting_second, end_second]
    data.extend(lineup)
    lineup = pd.DataFrame(data=[data], columns=cols)
    lineups = lineups.append(lineup)

    return lineups


def _lineups_game_sec(on_ice, game, team, season):
    team_on_ice = pd.DataFrame()

    # minute range
    seconds = map(str, range(3600))
    on_ice = on_ice.groupby('Player')
    for name, player in on_ice:
        range_player = _second_ranges(player)
        data = [name, team, game, season] + range_player
        cols = ['name', 'team', 'game', 'season'] + seconds
        player = pd.DataFrame(data=[data], columns=cols)
        team_on_ice = team_on_ice.append(player)

    # define 6 player lineups
    # for single team for a game
    names = map(str, range(6))
    cols = ['team', 'game', 'season', 'starting_sec', 'end_sec'] + names
    lineups = pd.DataFrame(columns=cols)
    current_lineup = team_on_ice.loc[team_on_ice[str(0)] == 1, :]
    starting_sec = 0
    for second in range(2880):
        second_lineup = team_on_ice.loc[team_on_ice[str(second)] == 1, :]
        if not current_lineup.equals(second_lineup):
            # add old lineup
            end_sec = second
            try:
                lineups = _form_lineup(lineups,current_lineup,team,game,season,starting_sec,end_sec,cols)
            except LineupFormationException:
                current_lineup = team_on_ice.loc[team_on_ice[str(second - 1)] == 1, :]
                continue
            # start the time for new lineup
            starting_sec = second + 1
            current_lineup = second_lineup

    # add last lineup
    if not starting_sec == 3600:
        end_sec = 3599
        try:
            lineups = _form_lineup(lineups, current_lineup, team, game, season, starting_sec, end_sec, cols)
        except LineupFormationException:
            return lineups

    return lineups


def _lineups(on_ice, data_config):
    """
    Use the minute ranges to find lineup changes in games
    """
    lineups = pd.DataFrame()

    # debugging purposes
    if data_config['gameids'] is not None and data_config['limit']:
        gameids = on_ice.loc[on_ice.game.isin(data_config['gameids']), 'game'].drop_duplicates(inplace=False).values
    else:
        gameids = on_ice.loc[:, 'Game_Id'].drop_duplicates(inplace=False).values

    for gameid in tqdm(gameids):
        try:
            on_ice_game = on_ice.loc[on_ice.Game_Id == gameid]
            season = data_config['years'][0]
            teams = on_ice_game.loc[:, 'Team'].drop_duplicates(inplace=False).values

            for team in teams:
                on_ice_team = on_ice_game.loc[on_ice_game.Team == team, :]
                game_lineups = _lineups_game_sec(on_ice_team, gameid, team, season)
                lineups = lineups.append(game_lineups)
        except Exception as err:
            continue
            # print('Something went wrong in game: %s' % (gameid)) investigate later

    return lineups

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print("...Docopt... ")
    print(arguments)
    print("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    on_ice = pd.read_csv('%s/%s' % (CONFIG.data.nhl.lineups.dir, 'on_ice_players.csv'))
    on_ice = on_ice.drop(columns=['Unnamed: 0'], axis=1)
    lineups = _lineups(on_ice, data_config)
    lineups.to_csv('%s/%s' % (CONFIG.data.nhl.lineups.dir, 'lineups-sec.csv'), index=False)
