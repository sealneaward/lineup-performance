import importlib
import pandas as pd
from tqdm import tqdm

from lineup.data.nba.get_matchups import _pbp, _cols, _game_matchups, _performance_vector, MatchupException

class Previous:
    """
    Use previous lineup/matchup data as input for model
    """
    def __init__(self, config):
        self.config = config
        self.model = getattr(importlib.import_module(config['module']), config['model'])()

    def prep_data(self, data):
        self.lineups = data
        self.lineups = self._matchups()

    def train(self):
        self.model.fit(self.data)

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
        matchups = pd.DataFrame()
        cols = _cols(self.config)

        # debugging purposes
        season = '2016'

        gameids = self.lineups.loc[:, 'game'].drop_duplicates(inplace=False).values

        for game in tqdm(gameids):
            pbp = _pbp(game)
            game_lineups = self.lineups.loc[self.lineups.game == game, :]
            game_matchups = _game_matchups(data_config=self.config, lineups=game_lineups, cols=cols, game=game, season=season)
            if game_matchups.empty:
                continue
            game_matchups = self._matchup_performances(matchups=game_matchups, lineups=game_lineups, pbp=pbp)
            if game_matchups.empty:
                continue
            matchups = matchups.append(game_matchups)

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
        previous_matchup = pd.DataFrame()

        i = 0
        for ind, matchup in matchups.iterrows():
            if i > 0:
                performance = self._performance(matchup, previous_matchup, pbp)
                if not performance.empty:
                    performances = performances.append(performance)
            i += 1
            previous_matchup = matchup

        performances = pd.concat([matchups, performances], axis=1)
        return performances

    def _performance(self, matchup, previous_matchup, pbp):
        """
        Get performance for single matchup
        """
        starting_min = matchup['starting_min']
        end_min = matchup['end_min']
        previous_starting_min = previous_matchup['starting_min']
        previous_end_min = previous_matchup['end_min']

        matchup_pbp = pbp.loc[(pbp.minute >= starting_min) & (pbp.minute <= end_min), :]
        previous_matchup_pbp = pbp.loc[(pbp.minute >= previous_starting_min) & (pbp.minute <= previous_end_min), :]

        # get totals for home
        team_matchup_pbp = matchup_pbp.loc[matchup_pbp.home == True, :]
        performance_home = _performance_vector(team_matchup_pbp, 'home')
        team_matchup_pbp = previous_matchup_pbp.loc[previous_matchup.home == True, :]
        performance_home_previous = _performance_vector(team_matchup_pbp, 'home')

        # get totals for visitor
        team_matchup_pbp = matchup_pbp.loc[matchup_pbp.home == False, :]
        performance_away = _performance_vector(team_matchup_pbp, 'visitor')
        team_matchup_pbp = previous_matchup_pbp.loc[previous_matchup.home == False, :]
        performance_away_previous = _performance_vector(team_matchup_pbp, 'visitor')

        performance = pd.concat([performance_home, performance_away], axis=1)

        return performance