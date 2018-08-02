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
    def __init__(self, config):
        self.config = config
        self.model = getattr(importlib.import_module(config['module']), config['model'])()

    def prep_data(self, data, home_abilities, away_abilities):
        self.lineups = data
        self.home_abilities = home_abilities
        self.away_abilities = away_abilities
        self.matchups = self._matchups()
        self.matchups.to_csv('%s/%s' % (CONFIG.data.nba.matchups.dir, 'matchups-previous.csv'), index=False)

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
        cols = _cols(self.config)

        # debugging purposes
        season = '2016'

        gameids = self.lineups.loc[:, 'game'].drop_duplicates(inplace=False).values
        for game in tqdm(gameids):
            try:
                with time_limit(30):
                    pbp = _pbp(game)
                    game_lineups = self.lineups.loc[self.lineups.game == game, :]
                    game_matchups = _game_matchups(data_config=self.config, lineups=game_lineups, cols=cols, game=game, season=season)
                    if game_matchups.empty:
                        continue
                    game_matchups = self._matchup_performances(matchups=game_matchups, lineups=game_lineups, pbp=pbp)

            except TimeoutException as e:
                print("Game sequencing too slow for %s - skipping" % (game))
                continue

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
        augmented_matchups = pd.DataFrame()
        previous_matchup = pd.DataFrame()

        i = 0
        for ind, matchup in matchups.iterrows():
            performance = self._performance(matchup, pbp)
            if not performance.empty:
                if (int(performance['pts_home']) - int(performance['pts_visitor'])) > 0:
                    performance['outcome'] = 1
                elif (int(performance['pts_home']) - int(performance['pts_visitor'])) <= 0:
                    performance['outcome'] = -1

                performances = performances.append(performance)


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