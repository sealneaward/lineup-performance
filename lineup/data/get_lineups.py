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
from datetime import datetime
import time
from bs4.element import Comment
from bs4 import BeautifulSoup
import requests
from docopt import docopt
import yaml
import re
from itertools import izip
import urllib2

import lineup.config as CONFIG


class Player:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.minutes_count = [0.0] * 48
        self.games_count = 0
        self.games_played = 0
        self.games_started = 0
        self.minutes_played = 0

    def set_games_data(self, games_played, games_started, minutes_played):
        self.games_played = games_played
        self.games_started = games_started
        self.minutes_played = minutes_played

    def add_minute_range(self, start_min, end_min):
        for i in range(start_min, end_min):
            if self.minutes_count[i] < self.games_count:
                self.minutes_count[i] += 1.0

    def get_position_val(self):
        if "PG" in self.position:
            return 1
        if "SG" in self.position:
            return 2
        if "SF" in self.position:
            return 3
        if "PF" in self.position:
            return 4
        if "C" in self.position:
            return 5

        print("Uh oh, no position for: " + self.name)
        return 0


def _quarter(q, on_court_width):
    """
    Get the quarter 1-4 based on the starting court width at the time
    """
    if on_court_width < 250:
        q = 1
    elif 250 <= on_court_width and on_court_width < 500:
        q = 2
    elif 500 <= on_court_width and on_court_width < 750:
        q = 3
    else:
        q = 4

    return q

def process_plus_minus(plus_minus_link, isHomeGame, num_overtimes, players, team_abr, game, season):
    on_court = []

    width_regex = re.compile("width:([0-9]+)px")
    try:
        response = urllib2.urlopen(urllib2.Request(plus_minus_link, headers={'User-Agent': 'Mozilla'})).read()
    except urllib2.HTTPError:
        return False
    pm_soup = BeautifulSoup(response, 'lxml')
    pm_div = pm_soup.find("div", {"class": "plusminus"})
    style_div =pm_div.find("div", recursive=False)

    total_width = int(width_regex.search(style_div['style']).group(1)) - 1
    team_table = style_div.findAll("div", recursive=False)[isHomeGame]
    rows = team_table.findAll("div", recursive=False)[1:]

    total_minutes = 48.0 + (5.0 * num_overtimes)
    minute_width = total_width / total_minutes
    for player_row, minutes_row in izip(*[iter(rows)] * 2):
        player_name = player_row.find('span').text
        player_obj = players[player_name]
        player_obj.games_count += 1
        curr_minute = 0.0
        for bar in minutes_row.findAll('div'):
            if round(curr_minute) < 48:
                classes = bar.get('class')
                width = int(width_regex.search(bar.get('style')).group(1)) + 1
                span_length = width / minute_width

                start_time = int(round(curr_minute))
                end_time = int(round(curr_minute + span_length))

                on_court_team_player_time = {}
                on_court_team_player_time['team'] = team_abr
                on_court_team_player_time['player'] = player_name
                on_court_team_player_time['game'] = game
                on_court_team_player_time['season'] = season
                on_court_team_player_time['start_min'] = start_time
                on_court_team_player_time['end_min'] = end_time
                curr_minute += span_length

                if classes is not None and ("plus" in classes or "minus" in classes or "even" in classes):
                    on_court.append(on_court_team_player_time)
    on_court = pd.DataFrame(on_court)
    return on_court

def generate_player_dictionary(team_page_link):
    player_dict = {}
    response = urllib2.urlopen("http://www.basketball-reference.com" + team_page_link).read()
    team_page = BeautifulSoup(response, 'lxml')
    roster_rows = team_page.find("table", {"id": "roster"}).find("tbody").findAll("tr")

    for player_row in roster_rows:
        player_name = player_row.find("td", {"data-stat": "player"}).find("a").text
        if player_name == "Glenn Robinson III":
            player_name = "Glenn Robinson"
        elif player_name == "Nene":
            player_name = "Nene Hilario"
        elif player_name == "Taurean Prince":
            player_name = "Taurean Waller-Prince"
        elif player_name == "Kelly Oubre Jr.":
            player_name = "Kelly Oubre"

        position = player_row.find("td", {"data-stat": "pos"}).text
        if player_name in player_dict:
            print('Uh oh, we found a duplicate: ' + player_name + " on " + team_page_link)
        else:
            p = Player(player_name, position)
            player_dict[player_name] = p

    comments = team_page.findAll(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment_string = re.split("(?:<!--)|(?:-->)", comment)[0]
        comment_soup = BeautifulSoup(comment_string, "lxml")
        totals_table = comment_soup.find("table", {"id": "totals"})
        if totals_table:
            totals_rows = totals_table.find("tbody").findAll("tr")
            for totals_row in totals_rows:
                cols = totals_row.findAll("td")
                player_name = cols[0].find("a").text

                if player_name not in player_dict:
                    player_dict[player_name] = Player(player_name, "N/A")
                    print("Adding ", player_name)

                p = player_dict[player_name]

                games_played = int(cols[2].find("a").text)
                games_started = int(cols[3].text)
                minutes_played = int(cols[4].text)

                p.set_games_data(games_played, games_started, minutes_played)

    return player_dict


def main_plus_minus(data_config):
    """
    Uses basketball-reference endpoints to get 5 player lineups
    when they get off and on the court for different teams
    from random import choice

    Writes all of the lineups for all games to pkl file.

    Parameters
    ----------
    data_config: yaml
        scraping config
    """
    today = datetime.now().date()
    years = data_config['']
    on_court = pd.DataFrame()

    years = years[:1]
    for year in years:
        print("DOING YEAR " + year)
        link = "http://www.basketball-reference.com/leagues/NBA_" + year + ".html"
        response = urllib2.urlopen(urllib2.Request(link, headers={'User-Agent': 'Mozilla'})).read()
        season_summary = BeautifulSoup(response, 'lxml')
        comments = season_summary.findAll(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment_string = re.split("(?:<!--)|(?:-->)", comment)[0]
            comment_soup = BeautifulSoup(comment_string, "lxml")
            team_stats = comment_soup.find("table", {"id": "team-stats-per_game"})
            if team_stats:
                team_names = team_stats.find("tbody").findAll("td", {"data-stat": "team_name"})
                team_names = team_names[:2]
                for team_name in team_names:
                    team_page_link = team_name.find("a")['href']
                    abr_regex = re.compile("^\/teams\/(.*)\/.*\.html")
                    team_abr = abr_regex.search(team_page_link).group(1)

                    players = generate_player_dictionary(team_page_link)
                    schedule_link = "http://www.basketball-reference.com/teams/" + team_abr + "/" + year + "_games.html"
                    response = urllib2.urlopen(urllib2.Request(schedule_link, headers={'User-Agent': 'Mozilla'})).read()
                    schedule_soup = BeautifulSoup(response, 'lxml')
                    game_rows = schedule_soup.find("table", {"id": "games"}).find("tbody").findAll("tr",
                                                                                                   {"class": None})
                    print("Working on " + team_abr)
                    gamesPlayed = 0.0
                    for game_row in game_rows:
                        gameDate = datetime.strptime(game_row.find("td", {"data-stat": "date_game"})['csk'],
                                                     "%Y-%m-%d").date()
                        if gameDate >= today:
                            print("Breaking due to date")
                            break
                        else:
                            game_link = game_row.find("td", {"data-stat": "box_score_text"}).find("a")['href']
                            gameID_regex = re.compile('^/boxscores/([^.]+).html')
                            gameID = gameID_regex.search(game_link).group(1)

                            isHomeGame = not game_row.find("td", {"data-stat": "game_location"}).text == "@"

                            overtime_string = game_row.find("td", {"data-stat": "overtimes"}).text
                            num_overtimes = 0
                            if overtime_string:
                                if overtime_string == "OT":
                                    num_overtimes = 1
                                else:
                                    num_overtimes = int(overtime_string[0])
                            plus_minus_link = "http://www.basketball-reference.com/boxscores/plus-minus/" + gameID + ".html"

                            team_on_off_game = process_plus_minus(
                                plus_minus_link,
                                isHomeGame,
                                num_overtimes,
                                players,
                                team_abr,
                                gameID,
                                year
                            )
                            if team_on_off_game.empty:
                                print("Empty response")
                                continue
                            else:
                                # process the return
                                on_court = on_court.append(team_on_off_game)
                            gamesPlayed += 1.0

    on_court.to_csv('%s/%s' % (CONFIG.data.lineups.dir, 'on_court_players.csv'), index=False)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))
    main_plus_minus(data_config)