from collections import defaultdict
import numpy as np


class MonteCarloTree:
    """Class for the Montecarlo tree"""

    def __init__(self, c, actor, simworld):
        """
        Initializes the MonteCarloTree
        :param c: Exploration bonus weight
        :param actor: The actor
        :param simworld: The StateManager that handles the game
        :return: None
        """
        self.root = Node(
            state=simworld.get_grid(),
            parent=None,
            player=simworld.current_player)
        self.c = c
        self.actor = actor
        self.simworld = simworld

    def update_root(self, action):
        """
        Changes the root from the current root to the node/state resulting from the chosen action
        :param action: The chosen action
        :return: None
        """
        new_root = self.root.get_child(action)
        if not new_root:
            self.root = Node(
                state=self.simworld.get_grid(),
                parent=None,
                player=self.simworld.current_player)
        self.root = new_root

    def search(self):
        """
        Traverses down the MonteCarloTree, choosing the action with the highest evaluation, until a leaf node is found
        :return: The leaf node to expand
        """
        current_node = self.root

        player = current_node.player
        sign = 1 if player == 1 else -1

        while current_node.get_child_count() > 0:

            best = -sign * float('inf')
            best_action = None
            for action in current_node.children:
                a = current_node.get_q(action) + sign * current_node.u(action, self.c)
                if a > best and sign == 1:
                    best = a
                    best_action = action
                elif a < best and sign == -1:
                    best = a
                    best_action = action

            current_node = current_node.get_child(best_action)
            sign *= -1

        return current_node

    def rollout(self, start_node):
        """
        Performs the rollout. Starts at start_node, gets an action from Actor and moves to the next state/node.
        Continues this until a final state is reached
        :return: The final node and reward of the goal state
        """
        state = start_node.state
        player = start_node.player

        is_final_state = self.simworld.is_finished(state)

        parent = start_node
        while not is_final_state:
            action = self.actor.get_action(state, player)
            state, player = self.simworld.get_state_from_action(player, state, action)
            node = parent.get_child(action, rollout=True)
            if not node:
                node = Node(state, parent, player)
                parent.add_child(node, action, rollout=True)
            parent = node
            is_final_state = self.simworld.is_finished(state)

        final_state = parent.state
        reward = self.simworld.get_reward(final_state)
        return parent, reward

    def expand(self, node):
        """
        Generates all successor nodes of node, and assigns them as children to node
        :param node: The node to expand
        :return: A list of the successor nodes
        """
        state = node.state
        player = node.player

        player, state_action_list = self.simworld.generate_successor_states(player, state)

        successors = []
        for state, action in state_action_list:
            successor = node.get_child(action, rollout=True)
            node.remove_child(action, rollout=True)
            if not successor:
                successor = Node(state, node, player)
            node.add_child(successor, action)
            successors.append(successor)
        return successors

    def backpropagate(self, leaf, reward):
        """
        Backpropagates the reward along all visited nodes during the MCT search (all visited nodes outside of rollout)
        :param leaf: The leaf (last visited node in the MCT search)
        :param reward: The reward from the final state
        :return: None
        """
        node = leaf

        while node != self.root:
            node.count += 1
            node.eval += reward

            node = node.parent

        self.root.count += 1
        self.root.eval += reward

    def mcts(self):
        """
        The MonteCarlo tree search. Gets a leaf node, expands the leaf node,
        and begins a rollout from a random successor. If a final state is reached, it starts backpropagation.
        :return: None
        """
        if self.simworld.is_finished(self.root.state):
            return

        leaf = self.search()

        if self.simworld.is_finished(leaf.state):
            reward = self.simworld.get_reward(leaf.state)
            self.backpropagate(leaf, reward)
            return

        successors = self.expand(leaf)

        successor = np.random.choice(successors)

        leaf, reward = self.rollout(successor)

        # If only nodes in MCT should be updated change leaf to successor
        self.backpropagate(successor, reward)

    def get_distribution(self):
        """
        Gets the distribution of visit counts of the children of the root
        :return: The normalized distribution
        """
        board_shape = self.simworld.get_grid().shape
        distribution = np.zeros(board_shape)
        for action in self.root.children:
            child = self.root.children[action]
            distribution[action] = child.count
        distribution = distribution.flatten()
        return distribution / np.sum(distribution)


class Node:
    """Class for a node in the MCT"""

    def __init__(self, state, parent, player):
        """
        Initializes a node
        :param state: The state of the node
        :param parent: The parent node
        :param player: The current player of the node
        :return: None
        """
        self.parent = parent
        self.children = defaultdict(lambda: None)
        self.rollout_children = defaultdict(lambda: None)
        self.state = state
        self.eval = 0
        self.count = 0
        self.player = player

    def add_child(self, child, action, rollout=False):
        """
        Adds a child node
        :param child: Child node to be added
        :param action: The action that moves the current state to the child's state
        :param rollout: Boolean of whether to add child as actual child or rollout child
        :return: None
        """
        if rollout:
            self.rollout_children[action] = child
            return
        self.children[action] = child

    def get_child(self, action, rollout=False):
        """
        Gets the child resulting from the chosen action
        :param action: The chosen action
        :param rollout: Boolean of whether to get an actual child or a rollout child
        :return: The child node
        """
        if rollout:
            return self.rollout_children[action]
        return self.children[action]

    def remove_child(self, action, rollout=False):
        """
        Removes the child resulting from the chosen action
        :param action: The chosen action
        :param rollout: Boolean of whether to remove the child node from actual children or rollout children
        :return: None
        """
        if rollout:
            self.rollout_children[action] = None
            return
        self.children[action] = None

    def get_action_count(self, action):
        """
        :param action: The chosen action
        :return: Visit count of child node resulting from the action
        """
        return self.children[action].count if self.children[action] else 0

    def get_child_count(self):
        """
        :return: The number of child nodes
        """
        return len(self.children)

    def get_parent(self):
        """
        :return: The parent node
        """
        return self.parent

    def get_q(self, action):
        """
        Calculates the q-value of the child node resulting from the action
        :param action: The action to evaluate
        :return: The q-value
        """
        if self.get_action_count(action) == 0:
            return 0
        return self.get_child(action).eval / (self.get_action_count(action))

    def u(self, action, c):
        """
        Calculates the u-value of the child node resulting from the action
        :param action: The action to evaluate
        :param c: Exploration weight
        :return: The u-value
        """
        return c * np.sqrt(np.log(self.count) / (1 + self.get_action_count(action)))
