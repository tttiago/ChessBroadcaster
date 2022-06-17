"""Scrape chess-results page to get the round pairings and fill the PGN metadata."""

import re

import requests
from bs4 import BeautifulSoup

input_file = "initial_games.pgn"
tournament_id = 633851
tournament_place = "Clube dos Galitos"
round = 4
n_players = 5
url = f"https://chess-results.com/tnr{tournament_id}.aspx?lan=10&art=2&rd={round}&prt=1"

page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

divs = soup.find_all("div", class_="defaultDialog")
tournament_name = divs[0].find("h2").get_text().split("|")[0].strip()
round_info = divs[1].find("h3").get_text()
round_date = re.search(r"\d{4}[/.-]\d{2}[/.-]\d{2}", round_info).group()

table = soup.find("table", {"class": "CRs1"})
players, ratings = [], []
for row in table.findAll("tr")[1 : n_players + 1]:
    for name_row in row.select("td:has(a)"):
        players.append(name_row.get_text())
    for rating_row in row.select(".CRr"):
        ratings.append(rating_row.get_text())


players = [", ".join(player.split()[:2]).title() for player in players]

with open(input_file, "w") as in_file:
    for board in range(1, n_players + 1):
        in_file.write(f'[Event "{tournament_name}"]\n')
        in_file.write(f'[Site "{tournament_place}"]\n')
        in_file.write(f'[Round "{round}.{board}"]\n')
        in_file.write(f'[White "{players[board*2-2]}"]\n')
        in_file.write(f'[WhiteElo "{ratings[board*2-2]}"]\n')
        in_file.write(f'[Black "{players[board*2-1]}"]\n')
        in_file.write(f'[BlackElo "{ratings[board*2-1]}"]\n')
        in_file.write(f'[Date "{round_date}"]\n')
        if board != n_players:
            in_file.write("\n\n")
