import importlib
import pandas as pd
from tqdm import tqdm

from lineup.data.nba.get_matchups import _pbp, _cols, _game_matchups, _matchup, _performance, MatchupException

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
            game_matchups = self._matchup_performances(matchups=game_matchups, pbp=pbp)
            if game_matchups.empty:
                continue
            matchups = matchups.append(game_matchups)

        return matchups


    def _game_matchups(self, game, season, cols):
        """
        Get matchups for game
        """

        if self.config['time_seperator'] == 'min':
            time_start = 'starting_minute'
            time_end = 'end_minute'
            time_range = range(48)
        else:
            time_start = 'starting_sec'
            time_end = 'end_sec'
            time_range = range(2880)

        matchups = pd.DataFrame()
        starting_lineups = self.lineups.loc[(self.lineups[time_start] <= 0) & (self.lineups[time_end] >= 0), :]
        try:
            current_matchup = _matchup(game=game, season=season, lineups=starting_lineups, cols=cols, time_start=time_start, time_end=time_end)
        except MatchupException:
            return pd.DataFrame()

        for time in time_range:
            try:
                time_lineups = self.lineups.loc[(self.lineups[time_start] <= time) & (self.lineups[time_end] >= time), :]
                matchup = _matchup(game=game, season=season, lineups=time_lineups, cols=cols, time_start=time_start, time_end=time_end)
                if not matchup.equals(current_matchup):
                    # lineup change detected
                    # record previous matchup
                    matchups = matchups.append(current_matchup)
                    current_matchup = matchup
            except MatchupException:
                continue

        matchups = matchups.append(current_matchup)

        return matchups

    def _matchup_performances(self, matchups, pbp):
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
                performances = performances.append(performance)

        performances = pd.concat([matchups, performances], axis=1)
        return performances