import importlib
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import signal
from sklearn.model_selection import train_test_split

import lineup.config as CONFIG
from lineup.model.utils import *
from lineup.data.nba.get_matchups import _pbp, _cols, _game_matchups, _performance_vector, MatchupException

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


class Abilities:
    """
    Use previous lineup/matchup data as input for model
    """
    def __init__(self, data_config, model_config, data):
        self.data = data
        self.model_config = model_config
        self.data_config = data_config
        self.model = getattr(importlib.import_module(self.model_config['sklearn']['module']), self.model_config['sklearn']['model'])()

    def prep_data(self):
        self.lineups = self.data
        self.home_abilities = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'home_abilities.csv'))
        self.away_abilities = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'away_abilities.csv'))
        self.matchups = self._matchups()
        self.matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-abilities.csv'), index=False)

    def train(self):
        self.matchups = pd.read_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-abilities.csv'))

        # clean
        self.matchups = clean(self.data_config, self.matchups, 'abilities')

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
        Each matchup contains the abilites of the lineups.

        Parameters
        ----------
        data_config: dict
            additional config setting
        lineups: pandas.DataFrame
            information on single team lineup at time T
        """
        matchups = pd.DataFrame()
        cols = _cols(self.data_config)

        # debugging purposes
        season = '2016'

        gameids = self.lineups.loc[:, 'game'].drop_duplicates(inplace=False).values
        for game in tqdm(gameids):
            try:
                with time_limit(30):
                    pbp = _pbp(game)
                    game_lineups = self.lineups.loc[self.lineups.game == game, :]
                    game_matchups = _game_matchups(data_config=self.data_config, lineups=game_lineups, cols=cols, game=game, season=season)
                    if game_matchups.empty:
                        continue
                    game_matchups = self._matchup_performances(matchups=game_matchups, lineups=game_lineups, pbp=pbp)
                    if game_matchups.empty:
                        continue
                    matchups = matchups.append(game_matchups)

            except TimeoutException as e:
                print("Game sequencing too slow for %s - skipping" % (game))
                continue

        return matchups

    def _matchup_performances(self, matchups, lineups, pbp):
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

        i = 0
        for ind, matchup in matchups.iterrows():
            performance = self._performance(matchup, pbp)
            if not performance.empty:
                try:
                    matchup = self._abilities(matchup)
                except MatchupException:
                    print('Abilities not found for home or away lineup')
                    continue
                if (int(performance['pts_home']) - int(performance['pts_visitor'])) > 0:
                    matchup['outcome'] = 1
                elif (int(performance['pts_home']) - int(performance['pts_visitor'])) <= 0:
                    matchup['outcome'] = -1
                performances = performances.append(matchup)

        return performances

    def _abilities(self, matchup):
        """
        Get abilities for single matchup
        """
        abilities = pd.DataFrame()
        home_ability = pd.DataFrame()
        away_ability = pd.DataFrame()

        home_ability = self.home_abilities.loc[
           (self.home_abilities['home_0'] == matchup['home_0']) &
           (self.home_abilities['home_1'] == matchup['home_1']) &
           (self.home_abilities['home_2'] == matchup['home_2']) &
           (self.home_abilities['home_3'] == matchup['home_3']) &
           (self.home_abilities['home_4'] == matchup['home_4'])
        ,:
        ]
        away_ability = self.away_abilities.loc[
           (self.away_abilities['away_0'] == matchup['away_0']) &
           (self.away_abilities['away_1'] == matchup['away_1']) &
           (self.away_abilities['away_2'] == matchup['away_2']) &
           (self.away_abilities['away_3'] == matchup['away_3']) &
           (self.away_abilities['away_4'] == matchup['away_4'])
        , :
        ]

        if home_ability.empty or away_ability.empty:
            raise MatchupException

        home_data = home_ability.values[0]
        home_cols = home_ability.columns.values
        away_data = away_ability.values[0]
        away_cols = away_ability.columns.values

        data = []
        data.extend(home_data)
        data.extend(away_data)

        cols = []
        cols.extend(home_cols)
        cols.extend(away_cols)

        abilities = pd.DataFrame(data=[data], columns=cols)
        return abilities


    def _performance(self, matchup, pbp):
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