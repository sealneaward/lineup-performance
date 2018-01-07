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

class LineupFormationException(Exception):
    pass

def _minute_ranges(player):
    minutes_count = [0.0] * 48
    for ind, r in player.iterrows():
        for i in range(r['start_min'], r['end_min']):
           minutes_count[i] += 1.0
    return minutes_count

def _second_ranges(player):
    minutes_count = [0.0] * 2880
    for ind, r in player.iterrows():
        for i in range(r['start_sec'], r['end_sec']):
           minutes_count[i] += 1.0
    return minutes_count


def _form_lineup(lineups, lineup, team, game, season, starting_second, end_second, cols):
    try:
        assert len(lineup) == 5
    except Exception:
        raise LineupFormationException('Incorrect number of people in lineup')
    lineup = lineup.loc[:, 'name'].values
    data = [team, game, season, starting_second, end_second]
    data.extend(lineup)
    lineup = pd.DataFrame(data=[data], columns=cols)
    lineups = lineups.append(lineup)

    return lineups

def _lineups_game_min(on_court, game, team, season):
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
                lineups = _form_lineup(lineups,current_lineup,team,game,season,starting_minute,end_minute,cols)
            except LineupFormationException:
                print('Something wrong in game lineup')
            # start the time for new lineup
            starting_minute = minute + 1
            current_lineup = minute_lineup

    # add last lineup
    if not starting_minute == 48:
        end_minute = 47
        try:
            lineups = _form_lineup(lineups, current_lineup, team, game, season, starting_minute, end_minute, cols)
        except LineupFormationException:
            print('Something wrong in end lineup')

    return lineups


def _lineups_game_sec(on_court, game, team, season):
    team_on_court = pd.DataFrame()

    # minute range
    seconds = map(str, range(2880))
    on_court = on_court.groupby('player')
    for name, player in on_court:
        range_player = _second_ranges(player)
        data = [name, team, game, season] + range_player
        cols = ['name', 'team', 'game', 'season'] + seconds
        player = pd.DataFrame(data=[data], columns=cols)
        team_on_court = team_on_court.append(player)

    # define 5 player lineups
    # for single team for a game
    names = map(str, range(5))
    cols = ['team', 'game', 'season', 'starting_sec', 'end_sec'] + names
    lineups = pd.DataFrame(columns=cols)
    current_lineup = team_on_court.loc[team_on_court[str(0)] == 1, :]
    starting_sec = 0
    for second in range(2880):
        second_lineup = team_on_court.loc[team_on_court[str(second)] == 1, :]
        if not current_lineup.equals(second_lineup):
            # add old lineup
            end_sec = second
            try:
                lineups = _form_lineup(lineups,current_lineup,team,game,season,starting_sec,end_sec,cols)
            except LineupFormationException:
                print('Something wrong in game lineup')
            # start the time for new lineup
            starting_sec = second + 1
            current_lineup = second_lineup

    # add last lineup
    if not starting_sec == 2880:
        end_sec = 2879
        try:
            lineups = _form_lineup(lineups, current_lineup, team, game, season, starting_sec, end_sec, cols)
        except LineupFormationException:
            print('Something wrong in end lineup')

    return lineups


def _lineups(on_court, data_config):
    """
    Use the minute ranges to find lineup changes in games
    """
    gameids = on_court.loc[:, 'game'].drop_duplicates(inplace=False).values
    lineups = pd.DataFrame()

    # debugging purposes
    if data_config['gameid'] is not None:
        gameids = [on_court.loc[on_court.game == data_config['gameid'], 'game'].values[0], '']


    for gameid in gameids:
        try:
            on_court_game = on_court.loc[on_court.game == gameid]
            season = on_court_game.loc[:, 'season'].drop_duplicates(inplace=False).values[0]
            teams = on_court_game.loc[:, 'team'].drop_duplicates(inplace=False).values

            for team in teams:
                on_court_team = on_court_game.loc[on_court_game.team == team, :]
                if data_config['time_seperator'] == 'min':
                    game_lineups = _lineups_game_min(on_court_team, gameid, team, season)
                else:
                    game_lineups = _lineups_game_sec(on_court_team, gameid, team, season)
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
    if data_config['time_seperator'] == 'min':
        lineups.to_csv('%s/%s' % (CONFIG.data.lineups.dir, 'lineups-min.csv'), index=False)
    else:
        lineups.to_csv('%s/%s' % (CONFIG.data.lineups.dir, 'lineups-sec.csv'), index=False)