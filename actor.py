import numpy as np
from utils import upload_local_directory_to_gcs
from gcloud import storage


class ActorNetwork:
    """Class for an actor that produces moves based on a state in matrix form"""

    def __init__(self, model, simworld, epsilon, epsilon_decay):
        """
        Initializes all variables for the actor.
        :param model: Keras model for producing probability distributions of actions from states.
        :param simworld: StateManager object for controlling the game
        :param epsilon: Float determining the probability of stochastically choosing an action from the probability
                        distribution rather than the highest value
        :param epsilon_decay: Float that is multiplied with the epsilon value to decay
        :return: None
        """
        self.model = model
        self.simworld = simworld
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def summary(self):
        return self.model.summary()

    def update_epsilon(self, zero=False):
        """
        Updates epsilon, either decay or set to 0
        :param zero: True if epsilon should be set to 0
        :return: None
        """
        self.epsilon *= self.epsilon_decay
        if zero:
            self.epsilon = 0

    def get_policy(self, state, player):
        """
        Gets a probability distribution of actions from model
        :param state: Numpy array of state
        :param player: Int of current player
        :return: Array of probabilities for actions
        """
        x = np.concatenate(([player], state), axis=None)
        x = x.reshape((1,) + x.shape)
        policy = self.model(x)
        policy = policy.numpy()
        return policy.reshape(policy.shape[-1])

    def update_policy(self, x, y, epochs=1):
        """
        Trains the model on single inmput and target
        :param x: Input, state of board
        :param y: Target, distribution of visit counts
        :param epochs: How many epochs to run the fit over the model
        :return: None
        """
        self.model.fit(x=x,
                       y=y,
                       epochs=epochs)

    def remove_illegal(self, prob_dist, state):
        """
        Removes illegal moves from the probability distribution
        :param prob_dist: Probability distribution of actions
        :param state: Current state
        :return: prob_dist with probabilities of illegal actions set to 0, normalized to sum to 1
        """
        for i in range(len(prob_dist)):
            action = np.unravel_index(i, self.simworld.get_grid().shape)
            action_is_legal = self.simworld.action_is_legal(state, action)
            if not action_is_legal:
                prob_dist[i] = 0
                prob_dist = prob_dist / prob_dist.sum()

        return prob_dist

    def get_action(self, state, player, random_possible=True):
        """
        Decides an action, chosen either from the highest probability or by likelihood epsilon chosen stochastically
        from the ANET prob dist.
        :param state: Current state
        :param player: Current player
        :param random_possible: Boolean of whether action can be chosen stochastically from the ANET prob dist.
        :return: An action
        """
        prob_dist = self.get_policy(state, player)
        prob_dist = self.remove_illegal(prob_dist, state)

        best_action = np.unravel_index(np.argmax(prob_dist), self.simworld.get_grid().shape)

        if np.random.random() < self.epsilon and random_possible:
            '''
            legal_moves = np.count_nonzero(prob_dist)
            legal_prob = np.where(prob_dist == 0, prob_dist, [1 / legal_moves for _ in range(len(prob_dist))])
            '''
            # The choice is now probabilistic over ANET prob dist., change to p=legal_prob if uniform.
            i = np.random.choice(a=np.arange(len(prob_dist)), size=None, p=prob_dist)
            best_action = np.unravel_index(i, self.simworld.get_grid().shape)

        return best_action

    def get_prob_action(self, state, player):
        """
        Decides an action stochastically from the probability distribution of the ANET
        :param state: Current state
        :param player: Current player
        :return: An action
        """
        prob_dist = self.get_policy(state, player)
        prob_dist = self.remove_illegal(prob_dist, state)

        i = np.random.choice(a=np.arange(len(prob_dist)), size=None, p=prob_dist)
        action = np.unravel_index(i, self.simworld.get_grid().shape)

        return action

    def perform_action(self, action):
        """
        Tells state manager to perform an action
        :param action: Action to perform
        :return: Next player
        """
        return self.simworld.perform_action(action)

    def save_net(self, episode):
        """
        Saves the keras model locally, with name episode
        :param episode: The current episode, will be name of folder
        :return: None
        """
        self.model.save("./anets/{episode}".format(episode=episode))
        # self.save_net_in_cloud(episode)  # Uncomment this to store in Google Cloud Storage

    def save_net_in_cloud(self, episode):
        """
        Saves the keras model in Google cloud
        :param episode: The current episode, will be name of folder
        :return: None
        """
        client = storage.Client()
        bucket = client.get_bucket('anets')
        upload_local_directory_to_gcs(local_path="./anets/{episode}".format(episode=episode),
                                      bucket=bucket,
                                      gcs_path="{episode}".format(episode=episode)
                                      )
