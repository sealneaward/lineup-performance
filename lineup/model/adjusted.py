import importlib
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import signal
from sklearn.model_selection import train_test_split
from copy import copy

import lineup.config as CONFIG
from lineup.model.utils import *
from lineup.data.nba.get_matchups import _pbp, _cols, _game_matchups, _performance_vector, _matchup_performances, MatchupException
from lineup.data.utils import _player_info

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class Adjusted:
    """
    Use adjusted plus minus data as input for model
    """
    def __init__(self, data_config, model_config, data):
        self.data = data
        self.model_config = model_config
        self.data_config = data_config
        self.model = getattr(importlib.import_module(self.model_config['sklearn']['module']), self.model_config['sklearn']['model'])()

    def prep_data(self):
        self.lineups = self.data
        self.matchups = self._matchups()
        self.matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-previous.csv'), index=False)

    def train(self):
        self.matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-previous.csv'))

        # clean
        self.matchups = clean(self.config, self.matchups, 'abilities')

        # split to train and test split
        Y = self.matchups['outcome']
        self.matchups.drop(['outcome'], axis=1, inplace=True)
        X = self.matchups
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(X, Y, test_size=self.config['split'])

        self.model.fit(self.train_x, self.train_y)


    def _matchups(self):
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
        season = '2016'
        matchups = pd.DataFrame()
        player_info = _player_info(season)

        gameids = self.lineups.loc[:, 'game'].drop_duplicates(inplace=False).values
        for game in tqdm(gameids):
            try:
                with time_limit(30):
                    pbp = _pbp(game)
                    game_lineups = self.lineups.loc[self.lineups.game == game, :]
                    game_matchups = _game_matchups(data_config=self.data_config, lineups=game_lineups, game=game, season=season, cols=_cols(self.data_config))
                    if game_matchups.empty:
                        continue

                    game_matchups = _matchup_performances(matchups=game_matchups, pbp=pbp)
                    if game_matchups.empty:
                        continue

                    home_possessions = []
                    away_possessions = []
                    for ind, matchup in game_matchups.iterrows():
                        matchup_home_possessions = self._possessions(matchup=matchup, possession_type='home')
                        matchup_away_possessions = self._possessions(matchup=matchup, possession_type='visitor')

                        home_possessions_matchup = copy(matchup_home_possessions)
                        away_possessions_matchup = copy(matchup_away_possessions)

                        # TODO get time played for each player on rosters
                        # TODO get score per possession for each matchup

                        home_possessions.append(home_possessions_matchup)
                        away_possessions.append(away_possessions_matchup)

                    game_matchups['poss_home'] = pd.Series(home_possessions).values
                    game_matchups['poss_visitor'] = pd.Series(away_possessions).values

                    matchups = matchups.append(game_matchups)

            except TimeoutException as e:
                print("Game sequencing too slow for %s - skipping" % (game))
                continue

        return matchups

    def _possessions(self, matchup, possession_type):
        possessions = 0
        try:
            if possession_type == 'home':
                possessions = 0.5 * (
                    (
                        matchup['fga_home']
                        + 0.4 * matchup['fta_home']
                        - 1.07 * (matchup['oreb_home'] / (matchup['oreb_home'] + matchup['dreb_visitor']))
                        * (matchup['fga_home'] - matchup['fgm_home'])
                        + matchup['to_home']
                    )
                    +
                    (
                        matchup['fga_visitor']
                        + 0.4 * matchup['fta_visitor']
                        - 1.07 * (matchup['oreb_visitor'] / (matchup['oreb_visitor'] + matchup['dreb_home']))
                        * (matchup['fga_visitor'] - matchup['fgm_visitor'])
                        + matchup['to_visitor']
                    )
                )
            elif possession_type == 'visitor':
                possessions = 0.5 * (
                    (
                        matchup['fga_visitor']
                        + 0.4 * matchup['fta_visitor']
                        - 1.07 * (matchup['oreb_visitor'] / (matchup['oreb_visitor'] + matchup['dreb_home']))
                        * (matchup['fga_visitor'] - matchup['fgm_visitor'])
                        + matchup['to_visitor']
                    )
                    +
                    (
                        matchup['fga_home']
                        + 0.4 * matchup['fta_home']
                        - 1.07 * (matchup['oreb_home'] / (matchup['oreb_home'] + matchup['dreb_visitor']))
                        * (matchup['fga_home'] - matchup['fgm_home'])
                        + matchup['to_home']
                    )
                )
        except Exception:
            return possessions

        return possessions