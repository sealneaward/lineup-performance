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
from bs4 import BeautifulSoup
import requests
from docopt import docopt
import yaml
import re

import lineup.config as CONFIG

def _teams(soup):
    """
    Get the teams from the game
    """
    teams = []

    team_divs = soup.find_all('h3')
    for team_div in team_divs:
        teams.append(str(team_div.contents[0]))

    return teams

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

def _players(team, soup):
    """
    Get the players from soup for the specified team
    """
    team_divs = soup.find_all('h3')
    players = []

    team_div = team_divs[0]
    team_name = team_div.contents[0]
    for tag in team_div.next_siblings:
        name = tag.name
        tag_type = type(tag)
        if tag.name == 'h3':
            team_name = tag.contents[0]
        elif tag.name is None:
            continue
        elif tag.attrs['class'][0] == 'player':
            player = {}
            player_div=tag
            player['name'] = str(tag.contents[0].contents[0])
            player['team'] = str(team_name)
            player['on-court'] = []
            on_court_width = 0
            q = 1

            del tag
            for tag in player_div.next_siblings:
                if tag.name is None:
                    continue
                elif tag.attrs['class'][0] == 'player-plusminus':
                    on_court_div = tag.contents
                    del tag
                    for tag in on_court_div:
                        if tag.name is None:
                            continue
                        else:
                            on_court = {}
                            q = _quarter(q, on_court_width)

                            if tag.next_sibling.name is None:
                                # starter
                                start_time = 720
                                end_width = int(re.findall(r'\d+', tag.attrs['style'])[0])
                                end_time = 720 - 720.0 * (end_width / 250.0)
                                on_court['quarter'] = q
                                on_court['start_time'] = start_time
                                on_court['end_time'] = end_time
                                on_court_width += end_width
                            else:
                                on_court_width += int(re.findall(r'\d+', tag.attrs['style'])[0])
                                start_time = 720 - 720.0 * (on_court_width - (q - 1) * 250.0) / 250.0
                                end_width = int(re.findall(r'\d+', tag.next_sibling.attrs['style'])[0])
                                end_time = 720 - 720.0 * (end_width / 250.0)
                                on_court['quarter'] = q
                                on_court['start_time'] = start_time
                                on_court['end_time'] = end_time
                                on_court_width += end_width

                            player['on-court'].append(on_court)
                    break
                else:
                    continue


            players.append(player)
        else:
            continue

    team_div = team_divs[1]
    for tag in team_div.next_siblings:
        if tag.name == 'h3':
            break
        else:
            print(None)

    print(None)


def _player(player, soup):
    """
    Using the plauyer, get the time on court as a single vector for player in soup
    """

def lineups(data_config):
    """
    Uses basketball-reference endpoints to get 5 player lineups
    when they get off and on the court for different teams
from random import choice

    Writes all of the lineups for all games to pkl file.

    Parameters
    ----------
    data_config: yaml
        endpoints
    """
    url = data_config['url'] % data_config['game']

    response = requests.get(url, headers=data_config['headers'], cookies=data_config['cookies'], verify=False)
    while response.status_code != 200:
        response = requests.get(url)
    soup = BeautifulSoup(response.text)

    teams = _teams(soup)
    for team in teams:
        players = _players(team, soup)
        for player in players:
            player_time = _player(player, soup)

    print(None)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print ("...Docopt... ")
    print(arguments)
    print ("............\n")

    f_data_config = '%s/%s' % (CONFIG.data.config.dir, arguments['<f_data_config>'])
    data_config = yaml.load(open(f_data_config, 'rb'))
    lineups(data_config)
