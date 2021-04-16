from mcts import MonteCarloTree
import numpy as np
import time
from utils import start_progress, progress, end_progress


class ReinforcementLearningSystem:

    def __init__(self, actor, simworld):
        """
        Initializes all variables for the RLS.
        :param actor: ActorNetwork object.
        :param simworld: StateManager object.
        """
        self.actor = actor
        self.simworld = simworld

    def learn(self, episodes, num_search_games, epochs,
              batch_size, save_interval, c=1, exp_weight=0, start_episode=0, max_time=2,
              visualize_last_game=False, visualize_final_state=False):
        """
        Implementation of the learning algorithm for
        the reinforcement learning system.
        :param episodes: Integer indicating number of episodes.
        :param num_search_games: Integer indicating number of search game per actual move.
        :param epochs: Integer number of epochs during ANET training.
        :param batch_size: Integer number of samples to use during ANET training.
        :param save_interval: Integer indicating how often to save ANET.
        :param c: Constant used in exploration bonus for MCTS.
        :param exp_weight: Weight to use in prob. dist. for RBUF.
        :param start_episode: Integer indicating starting number of episode.
        :param max_time: Float indicating max time to perform search games.
        :param visualize_last_game: True if last episode should be visualized in real-time.
        :param visualize_final_state: True if last final state of each episode should be visualized.
        :return: None
        """

        replay_buffer = []
        simworld = self.simworld
        actor = self.actor

        # Save "random" ANET
        actor.save_net(0)

        for episode in range(start_episode, episodes):
            current_player = episode % 2 + 1  # Switch which player starts

            print("Episode {episode}/{episodes}".format(episode=episode + 1, episodes=episodes))
            if episode == episodes - 1 and visualize_last_game:
                simworld.reset_game(visualize=True, player=current_player)
            else:
                simworld.reset_game(visualize=False, player=current_player)

            # Initialize MCT
            monte_carlo_tree = MonteCarloTree(c, actor, simworld)

            while not simworld.is_finished():

                # Perform search games
                start_time = time.perf_counter()
                timeout = False
                start_progress("Search game")
                for search_game in range(num_search_games):
                    monte_carlo_tree.mcts()
                    progress(search_game + 1, num_search_games)
                    if time.perf_counter() - start_time > max_time:
                        timeout = True
                        break
                end_progress(timeout)
                print("Timeout") if timeout else None

                # Get distribution from MCTS, add to buffer
                distribution = monte_carlo_tree.get_distribution()
                player = simworld.current_player
                state = simworld.get_grid()
                case = (np.concatenate(([player], state.flatten()), axis=None), distribution)
                replay_buffer.append(case)

                # Make actual move using ANET
                # action = actor.get_action(state, player, random_possible=False)
                #legal_distribution = actor.remove_illegal(distribution, state)
                action = np.unravel_index(np.argmax(distribution), simworld.get_grid().shape)

                '''
                if np.random.random() < actor.epsilon:
                    
                    #legal_moves = np.count_nonzero(prob_dist)
                    #legal_prob = np.where(prob_dist == 0, prob_dist, [1 / legal_moves for _ in range(len(prob_dist))])
                    
                    # The choice is now probabilistic over prob dist., change to p=legal_prob if uniform.
                    i = np.random.choice(a=np.arange(len(legal_distribution)), size=None, p=legal_distribution)
                    action = np.unravel_index(i, simworld.get_grid().shape)
                '''
                actor.perform_action(action)
                print("Player {player} chose action {action}".format(player=player, action=action))

                monte_carlo_tree.update_root(action)

            winner = 1 if simworld.has_won(1) else 2
            print("Actual game finished, player {winner} won.".format(winner=winner))

            # Creates mini_batch, distributed chance
            weights = [i ** exp_weight + 1e-10 for i in range(len(replay_buffer))]
            weights = weights / np.sum(weights)
            if batch_size >= 1:
                rnd_indices = np.random.choice(len(replay_buffer),
                                               size=batch_size if batch_size <= len(replay_buffer) else len(
                                                   replay_buffer),
                                               p=weights,
                                               replace=False)
            else:
                rnd_indices = np.random.choice(len(replay_buffer),
                                               size=int(len(replay_buffer) * batch_size),
                                               p=weights,
                                               replace=False)
            mini_batch = np.array(replay_buffer, dtype=tuple)[rnd_indices.astype(int)]
            x, y = zip(*mini_batch)

            # Train ANET
            actor.update_policy(x=np.array(x),
                                y=np.array(y),
                                epochs=epochs)

            # Update epsilon for Actor
            actor.update_epsilon()

            # Save net according to interval policy
            if (episode + 1) % save_interval == 0:
                actor.save_net(episode + 1)

            if visualize_final_state:
                self.simworld.visualize_state()


if __name__ == "__main__":
    pass
