import importlib
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import signal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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
        season = '2016'
        player_info = _player_info(season)
        self.player_names = player_info['Player'].values

        # self.matchups = self._matchups()
        # self.matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-adjusted.csv'), index=False)
        self.matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-adjusted.csv'))
        self.fit_regression()
        self.matchups_ridge.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-adjusted-regressed.csv'), index=False)
        self.matchups = self.matchups_ridge

    def fit_regression(self):
        """
        Use ridge regression to create unique ratings for players.
        """
        self.matchups = self.matchups.dropna()
        self.matchups_ridge = []

        X = self.matchups[self.player_names]
        Y = self.matchups['margin']

        clf = Ridge(alpha=1.0)
        clf.fit(X, Y)

        parameters = clf.coef_
        self.player_params = dict(zip(self.player_names, parameters))

        # once player coef values are established,
        # fit the values to the players in the matchups
        for ind, matchup in self.matchups.iterrows():
            matchup_ridge = {}
            for player_col in self.data_config['players']:
                try:
                    player = matchup[player_col]
                    matchup_ridge[player_col] = player
                    matchup_ridge['%s_apm' % player_col] = self.player_params[player]
                except Exception:
                    continue

            matchup_ridge['outcome'] = self.matchups.loc[ind, 'outcome']
            self.matchups_ridge.append(matchup_ridge)

        self.matchups_ridge = pd.DataFrame.from_records(self.matchups_ridge)
        self.matchups_ridge['lineup_apm'] = self.matchups_ridge[self.data_config['players_apm']].sum(axis=1)
        self.matchups_ridge['lineup_apm'] = self.matchups_ridge['lineup_apm'].values / 5

    def train(self):
        self.matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-adjusted-regressed.csv'))

        # clean
        self.matchups = clean(self.data_config, self.matchups, 'adjusted')
        self.matchups = self.matchups.dropna()

        # split to train and test split
        Y = self.matchups['outcome']
        self.matchups.drop(['outcome'], axis=1, inplace=True)
        X = self.matchups
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(X, Y, test_size=self.data_config['split'])

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

                        home_possessions.append(home_possessions_matchup)
                        away_possessions.append(away_possessions_matchup)

                    game_matchups['poss_home'] = pd.Series(home_possessions).values
                    game_matchups['poss_visitor'] = pd.Series(away_possessions).values

                    # calculate margin
                    game_matchups['margin'] = self._margins(game_matchups)
                    # get one hot representations of players on court
                    game_matchups = self._one_hot_player(game_matchups, player_info)

                    matchups = matchups.append(game_matchups)

            except TimeoutException as e:
                print("Game sequencing too slow for %s - skipping" % (game))
                continue

        return matchups

    def _possessions(self, matchup, possession_type):
        """
        Calculate number of possessions during matchup
        """
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

    def _margins(self, game_matchups):
        """
        Find points per possession margin for each matchup
        """
        margins = []
        game_matchups['margin'] = 0
        home_game_avg = game_matchups['pts_home'].sum() / game_matchups['poss_home'].sum()
        away_game_avg = game_matchups['pts_visitor'].sum() / game_matchups['poss_visitor'].sum()

        for ind, matchup in game_matchups.iterrows():
            if matchup['poss_home'] == 0 and matchup['poss_visitor'] == 0:
                margins.append(np.nan)
                continue
            if matchup['poss_home'] > 0:
                hv = matchup['pts_home'] / matchup['poss_home']
            else:
                hv = home_game_avg
            if matchup['poss_visitor'] > 0:
                av = matchup['pts_visitor'] / matchup['poss_visitor']
            else:
                av = away_game_avg

            margins.append(100*(hv-av))

        return pd.Series(margins).values

    def _one_hot_player(self, game_matchups, player_info, threshold=388):
        """
        Get associated player ids for each matchup
        based on whether they meet a threshold for minutes played.

        Then represent the players in the matchup with 1,-1,0
        as encoded representation of home, away, not present on-court(or not meet threshold)
        """
        player_info = player_info.loc[player_info['MP'] >= threshold, :]
        self.player_names = player_info['Player'].values

        for player in self.player_names:
            game_matchups[player] = 0

        for ind, matchup in game_matchups.iterrows():
            home_players = matchup[self.data_config['home_team']]
            visitor_players = matchup[self.data_config['away_team']]

            for home_player in home_players:
                try:
                    game_matchups.iloc[ind, game_matchups.columns.get_loc(home_player)] = 1
                except Exception:
                    continue

            for visitor_player in visitor_players:
                try:
                    game_matchups.iloc[ind, game_matchups.columns.get_loc(visitor_player)] = -1
                except Exception:
                    continue

        return game_matchups