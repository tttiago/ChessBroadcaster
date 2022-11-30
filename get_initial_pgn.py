"""Scrape chess-results page to get the round pairings and fill the PGN metadata."""

# TODO: catch exceptions (page can't be found, number of boards...)

import argparse
import re

import requests
from bs4 import BeautifulSoup


def create_parser():
    parser = argparse.ArgumentParser(
        description="Scrape chess-results to get pairings metadata.")
    parser.add_argument(
        "-t",
        "--tournament-id",
        metavar="TI",
        type=int,
        nargs="?",
        default=633851,
        help="ID of the tournament at chess-results.",
    )
    parser.add_argument(
        "-r",
        "--round",
        metavar="R",
        type=int,
        help="Tournament round.",
    )
    parser.add_argument(
        "-n",
        "--num-boards",
        metavar="N_BOARDS",
        type=int,
        nargs="?",
        default=5,
        help="Number of boards to be initialized.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    tournament_id = args.tournament_id
    round = args.round
    n_boards = args.num_boards
    input_file = "initial_games.pgn"
    tournament_place = "Clube dos Galitos"
    url = f"https://chess-results.com/tnr{tournament_id}.aspx?lan=10&art=2&rd={round}&prt=1"

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    divs = soup.find_all("div", class_="defaultDialog")
    tournament_name = divs[0].find("h2").get_text().split("|")[0].strip()
    round_info = divs[1].find("h3").get_text()
    round_date = re.search(r"\d{4}[/.-]\d{2}[/.-]\d{2}", round_info).group()

    table = soup.find("table", {"class": "CRs1"})
    players, ratings = [], []
    for row in table.findAll("tr")[1: n_boards + 1]:
        for name_row in row.select("td:has(a)"):
            players.append(name_row.get_text())
        for rating_row in row.select(".CRr"):
            ratings.append(rating_row.get_text())

    players = [", ".join(player.split()[:2]).title() for player in players]

    with open(input_file, "w") as in_file:
        for board in range(1, n_boards + 1):
            in_file.write(f'[Event "{tournament_name}"]\n')
            in_file.write(f'[Site "{tournament_place}"]\n')
            in_file.write(f'[Round "{round}.{board}"]\n')
            in_file.write(f'[White "{players[board*2-2]}"]\n')
            in_file.write(f'[WhiteElo "{ratings[board*2-2]}"]\n')
            in_file.write(f'[Black "{players[board*2-1]}"]\n')
            in_file.write(f'[BlackElo "{ratings[board*2-1]}"]\n')
            in_file.write(f'[Date "{round_date}"]\n')
            if board != n_boards:
                in_file.write("\n\n")

    print(f"{input_file} successfully updated with round {round} info.")
