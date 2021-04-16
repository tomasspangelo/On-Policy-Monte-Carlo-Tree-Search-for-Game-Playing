import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product


class StateManager:
    """Class for a state manager ("SimWorld") in the game of HEX."""

    def __init__(self, board_size=4, r_win=1, r_loss=-1, visualize=False, state=None):
        """
        Initializes all variables for the state manager.
        :param board_size: Integer k indicating the k x k board size.
        :param r_win: Float indicating the reward for winning (i.e. player 1 wins)
        :param r_loss: Float indicating the reward for loosing (i.e. player 1 looses)
        :param visualize: True if game should be visualized, otherwise False.
        :param state: The state of the game, if not provided a new kxk board with zeros will be initialized.
        """
        self.board_size = board_size
        self.r_win = r_win
        self.r_loss = r_loss

        self.current_player = 1
        if state is None:
            self.state = np.zeros((board_size, board_size), int)
        else:
            self.state = state
        self.finished = False

        self.visualize = visualize
        self.frame_delay = 0
        self.graph = nx.Graph()
        self.fig = plt.figure()

    def reset_game(self, visualize=False, frame_delay=1, player=1):
        """
        Resets the game.
        :param visualize: True if game should be visualized real-time, False otherwise.
        :param frame_delay: Float indicating amounts of seconds to delay real-time game plot.
        :param player: Starting player (1 or 2)
        :return: None
        """
        self.visualize = visualize
        self.frame_delay = frame_delay

        self.state = np.zeros((self.board_size, self.board_size), int)
        self.current_player = player
        self.finished = False

        if visualize:
            self.graph = nx.Graph()
            self.fig = plt.figure()
            self.visualize_state()

    def flatten_state(self, state=None):
        """
        Flattens the numpy array representing the state.
        :param state: Optional, if provided will flatten the provided state otherwise state of game.
        :return: 1-dimensional numpy array.
        """
        if state is None:
            state = self.state
        return state.flatten()

    def get_legal_actions(self, state=None):
        """
        Calculates legal options given state.
        :param state:  Optional, if provided will use this instead of game state.
        :return: List containing legal actions of form (row, column)
        """
        if state is None:
            state = self.state
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i, j] == 0:
                    legal_actions.append((i, j))

        return legal_actions

    def perform_action(self, action):
        """
        Performs action if it is legal, which changes game state.
        Player is inferred by the state of the StateManager.
        :param action: Tuple (row, column)
        :return:
        """
        if self.state[action] != 0:
            raise ValueError('Illegal move')
        else:
            self.state[action] = self.current_player

        if self.visualize:
            self.visualize_state(action)

        self.current_player = 2 if self.current_player == 1 else 1

    @staticmethod
    def get_state_from_action(player, state, action):
        """
        Static method for getting new state after a player performs
        a particular move.
        :param player: Player to perform action (1 or 2).
        :param state: Numpy array representing game state.
        :param action: Tuple (row, column) representing the action.
        :return: Numpy array with new state, Integer representing next player
        """
        if state[action] != 0:
            raise ValueError('Illegal move')
        new_state = state.copy()
        new_state[action] = player
        next_player = 2 if player == 1 else 1
        return new_state, next_player

    def generate_successor_states(self, player, state):
        """
        Generates all successor states.
        :param player: Current player (1 or 2).
        :param state: Numpy array representing state pf the game.
        :return: Next player (1 or 2), List with tuples of (state, action)
        """
        legal_actions = self.get_legal_actions(state)
        states = []
        state = state.copy()

        for action in legal_actions:
            states.append((self.get_state_from_action(player, state, action)[0], action))

        next_player = 2 if player == 1 else 1

        return next_player, states

    def is_finished(self, state=None):
        """
        Checks if the game is finished.
        :param state: Optional, if provided will see if state is finished.
        :return: Boolean
        """
        if state is None:
            state = self.state
        if self.has_won(1, state) or self.has_won(2, state):
            return True
        else:
            return False

    def has_won(self, player, state=None):
        """
        Checks if a particular player has won.
        Uses Depth First Search.
        :param player: Integer (1 or 2)
        :param state: Optional, if provided will see if player has won in state.
        :return: Boolean
        """
        if state is None:
            state = self.state
        edge = []
        visited = []
        size = len(state)
        if player == 1:
            for i in range(size):
                if state[0, i] == 1:
                    edge.append((0, i))
        else:
            for i in range(size):
                if state[i, 0] == 2:
                    edge.append((i, 0))

        while len(edge) != 0:
            node = edge.pop()
            visited.append(node)
            if player == 1 and node[0] == size - 1:
                return True
            elif player == 2 and node[1] == size - 1:
                return True

            adjacents = {(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)}

            for dy, dx in adjacents:
                if 0 <= node[0] + dy < size and 0 <= node[1] + dx < size \
                        and (node[0] + dy, node[1] + dx) not in (visited + edge) \
                        and state[(node[0] + dy, node[1] + dx)] == player:
                    edge.append((node[0] + dy, node[1] + dx))
        return False

    def get_reward(self, state=None):
        """
        Returns the reward if game is finished.
        :param state: Optional, if provided will give reward for state.
        :return: Reward (float)
        """
        if state is None:
            state = self.state
        if self.has_won(1, state):
            return self.r_win
        elif self.has_won(2, state):
            return self.r_loss
        else:
            raise ValueError('Not end state')

    def get_all_possible_actions(self):
        """
        :return: Two dimensional list containing all possible actions.
        """
        return list(product(range(self.board_size), range(self.board_size)))

    @staticmethod
    def action_is_legal(state, action):
        """
        Checks if action is legal given state.
        :param state: Numpy array representing state.
        :param action: Tuple (row, column) representing action.
        :return: Boolean
        """
        return state[action] == 0

    def get_grid(self):
        """
        :return: Numpy array representing state.
        """
        return self.state

    def visualize_state(self, action=None):
        """
        Method for visualizing the game state.
        :param action: Action to be performed.
        :return: None
        """
        graph = self.graph
        state = self.state
        board_size = self.board_size
        graph.clear()
        plt.clf()

        pos = {}
        total_height = 1 + 2 * board_size
        total_width = total_height
        color_map = []
        size_map = []

        added_node = action if action else (-1, -1)

        # Add nodes to graph (+position, color and size)
        c = 0
        for i in range(board_size):
            h = total_height - i
            w = np.ceil(total_width / 2) - i
            for j in range(board_size):
                node_value = state[i, j]
                graph.add_node(c)
                pos[c] = (w, h)
                size_map.append(1000)
                if node_value == 0:
                    color_map.append('lightgrey')
                if node_value == 1:
                    if (i, j) == added_node:
                        color_map.append('blue')
                    else:
                        color_map.append('blue')
                if node_value == 2:
                    if (i, j) == added_node:
                        color_map.append('red')
                    else:
                        color_map.append('red')

                h -= 1
                w += 1
                c += 1

        # Add edges to graph
        adjacents = {(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)}
        c = 0
        for i in range(board_size):
            for j in range(board_size):
                for dy, dx in adjacents:
                    if 0 <= i + dy < board_size and 0 <= j + dx < board_size \
                            and c != board_size * (i + dy) + (j + dx):
                        graph.add_edge(c, board_size * (i + dy) + (j + dx))
                c += 1

        nx.draw(graph, pos, node_color=color_map, node_size=size_map, with_labels=False,
                font_weight='bold')
        plt.pause(self.frame_delay)
        # plt.draw() #With this it only plots states 2 times, compared to 3 without

    def __str__(self):
        """
        :return: String representation of StateManager instance.
        """
        return str(self.state)


if __name__ == "__main__":
    pass
