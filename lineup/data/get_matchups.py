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

class LineupFormationException(Exception):
    pass

def _minute_ranges(player):
    minutes_count = [0.0] * 48
    for ind, r in player.iterrows():
        for i in range(r['start_min'], r['end_min']):
           minutes_count[i] += 1.0
    return minutes_count


def _form_lineup(lineups, minute_lineup, team, game, season, starting_minute, end_minute, cols):
    try:
        assert len(minute_lineup) == 5
    except Exception:
        raise LineupFormationException('Incorrect number of people in lineup')
    lineup = minute_lineup.loc[:, 'name'].values
    data = [team, game, season, starting_minute, end_minute]
    data.extend(lineup)
    lineup = pd.DataFrame(data=[data], columns=cols)
    lineups = lineups.append(lineup)

    return lineups

def _lineups_game(on_court, game, team, season):
    team_on_court = pd.DataFrame()

    # minute range
    minutes = map(str, range(48))
    on_court = on_court.groupby('player')
    for name, player in on_court:
        range_player = _minute_ranges(player)
        data = [name, team, game, season] + range_player
        cols = ['name', 'team', 'game', 'season'] + minutes
        player = pd.DataFrame(data=[data], columns=cols)
        team_on_court = team_on_court.append(player)

    # define 5 player lineups
    # for single team for a game
    names = map(str, range(5))
    cols = ['team', 'game', 'season', 'starting_minute', 'end_minute'] + names
    lineups = pd.DataFrame(columns=cols)
    current_lineup = team_on_court.loc[team_on_court[str(0)] == 1, :]
    starting_minute = 0
    for minute in range(48):
        minute_lineup = team_on_court.loc[team_on_court[str(minute)] == 1, :]
        if not current_lineup.equals(minute_lineup):
            # add old lineup
            end_minute = minute
            try:
                lineups = _form_lineup(lineups,minute_lineup,team,game,season,starting_minute,end_minute,cols)
            except LineupFormationException:
                print('Something wrong in game lineup')
            # start the time for new lineup
            starting_minute = minute + 1
            current_lineup = minute_lineup

    # add last lineup
    end_minute = 47
    try:
        lineups = _form_lineup(lineups, minute_lineup, team, game, season, starting_minute, end_minute, cols)
    except LineupFormationException:
        print('Something wrong in end lineup')

    return lineups


def _lineups(on_court, data_config):
    """
    Use the minute ranges to find lineup changes in games
    """
    gameids = on_court.loc[:, 'game'].drop_duplicates(inplace=False).values
    lineups = pd.DataFrame()

    for gameid in gameids:
        try:
            on_court_game = on_court.loc[on_court.game == gameid]
            season = on_court_game.loc[:, 'season'].drop_duplicates(inplace=False).values[0]
            teams = on_court_game.loc[:, 'team'].drop_duplicates(inplace=False).values

            for team in teams:
                on_court_team = on_court_game.loc[on_court_game.team == team, :]
                game_lineups = _lineups_game(on_court_team, gameid, team, season)
                lineups = lineups.append(game_lineups)
        except Exception as err:
             print('Something went wrong in game: %s' % (gameid))

    return lineups

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))

    on_court = pd.read_csv('%s/%s' % (CONFIG.data.lineups.dir, 'on_court_players.csv'))
    lineups = _lineups(on_court, data_config)
    lineups.to_csv('%s/%s' % (CONFIG.data.lineups.dir, 'lineups.csv'), index=False)