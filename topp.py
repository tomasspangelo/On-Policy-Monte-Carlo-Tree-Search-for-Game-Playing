import tensorflow as tf

from statemanager import StateManager
from actor import ActorNetwork
from itertools import permutations
from collections import OrderedDict
import sys
import matplotlib.pyplot as plt
import numpy as np


class Tournament:

    def __init__(self, path_list, ngames, epsilon):
        """
        Initializes all variables and loads ANETs.
        :param path_list: List of paths to ANETs.
        :param ngames: Integer indicating number of games between each actor.
        :param epsilon: Float indicating percentage of randomness.
        """
        self.actors = {}
        for path in path_list:
            model = tf.keras.models.load_model(path, compile=False)
            self.board_size = int((model.layers[0].input_shape[1] - 1) ** 0.5)
            actor = ActorNetwork(model, StateManager(board_size=self.board_size), epsilon, 1)
            if sys.platform == "win32":
                self.actors[int(path.split('\\')[-1].strip())] = actor
            else:
                self.actors[int(path.split('/')[-1].strip())] = actor

        self.nModels = len(self.actors)

        self.wins = [0] * self.nModels
        self.p1wins = [0] * self.nModels
        self.p2wins = [0] * self.nModels
        self.ngames = ngames

        self.actors = OrderedDict(sorted(self.actors.items()))

        self.state_manager = StateManager(board_size=self.board_size)

    def play_game(self, actor1, actor2, starting_player=1):
        """
        Play a single game between two actors.
        :param actor1: ActorNetwork object, player 1.
        :param actor2: ActorNetwork object, player 2.
        :param starting_player: Player that starts.
        :return:
        """
        current_actors = (actor1, actor2)
        self.state_manager.reset_game(player=starting_player)
        statemanager = self.state_manager
        while not statemanager.is_finished():
            player = statemanager.current_player
            action = current_actors[player - 1].get_action(statemanager.get_grid(), player, True)
            statemanager.perform_action(action)

        winner = 1 if statemanager.has_won(1) else 2
        looser = 2 if statemanager.has_won(1) else 1

        key_list = list(self.actors.keys())
        value_list = list(self.actors.values())
        pos_win = value_list.index(current_actors[winner - 1])
        pos_los = value_list.index(current_actors[looser - 1])
        print(key_list[pos_win], " beat ", key_list[pos_los], " as player ", winner,
              "with starting player", starting_player)
        self.wins[pos_win] += 1
        if winner == 1:
            self.p1wins[pos_win] += 1
        else:
            self.p2wins[pos_win] += 1

    def match_up(self, actor1, actor2):
        """
        Match up between actor1 and actor2.
        :param actor1: ActorNetwork object, player 1.
        :param actor2: ActorNetwork object, player 2.
        :return: None
        """
        for i in range(self.ngames):
            self.play_game(actor1, actor2, starting_player=i % 2 + 1)  # Can add both ways

    def play_tournament(self):
        """
        Plays the tournament.
        :return: None
        """
        matchups = list(permutations(list(self.actors), 2))
        for actor1, actor2 in matchups:
            self.match_up(self.actors[actor1], self.actors[actor2])
        self.wins = [float(i) / (self.ngames * 2 * (len(self.wins) - 1)) for i in self.wins]
        self.plot()
        self.plot(split=True)
        # Everything bellow plays the tournament two more times, removing losers
        '''
        for _ in range(2):
            print("New tournament")
            new_actors = OrderedDict()
            keys = list(self.actors.keys())
            for i in range(len(self.wins)):
                if self.wins[i] >= 0.5:
                    key = keys[i]
                    new_actors[key] = self.actors[key]

            self.actors = new_actors
            self.wins = [0] * len(self.actors)
            self.p1wins = [0] * len(self.actors)
            self.p2wins = [0] * len(self.actors)
            matchups = list(permutations(list(self.actors), 2))
            for actor1, actor2 in matchups:
                self.match_up(self.actors[actor1], self.actors[actor2])
            self.wins = [float(i) / (self.ngames * 2 * (len(self.wins) - 1)) for i in self.wins]

            self.plot()
            self.plot(True)
        '''

    def plot(self, split=False):
        """
        Plots the results from the tournament.
        :param split: True if plot should be split between player 1 and 2.
        :return: None
        """
        x = np.arange(len(self.actors))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        if split:
            rects1 = ax.bar(x - width / 2, self.p1wins, width, label='As P1')
            rects2 = ax.bar(x + width / 2, self.p2wins, width, label='As P2')
        else:
            rects = ax.bar(x - width / 2, self.wins, width)

        ax.set_ylabel('Number of wins')
        ax.set_title('Results')
        ax.set_xticks(x)
        ax.set_xticklabels(self.actors.keys())
        ax.legend() if split else None
        ax.yaxis.grid()
        fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    pass
