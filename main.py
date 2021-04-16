from rls import ReinforcementLearningSystem
from actor import ActorNetwork
from statemanager import StateManager
from topp import Tournament
from utils import custom_cross_entropy
import numpy as np
import tensorflow as tf

import os

import sys
from configparser import ConfigParser
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_actor(state_manager, actor_config):
    """
    Initializes an actor, with parameters set by the config file and a new keras model
    :param state_manager: StateManager object actor will use
    :param actor_config: Config parser for the actor
    :return: The actor
    """
    learning_rate = float(actor_config['learning_rate'])
    activations = json.loads(actor_config['activations'])
    neurons = json.loads(actor_config['neurons'])
    optimizer_name = actor_config['optimizer']

    board_size = state_manager.board_size

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(board_size ** 2 + 1,)))

    for activation, units in zip(activations, neurons):
        model.add(tf.keras.layers.Dense(units=units,
                                        activation=activation))
    model.add(tf.keras.layers.Dense(units=board_size ** 2, activation="softmax"))

    if optimizer_name == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=custom_cross_entropy)

    epsilon = float(actor_config['epsilon'])
    epsilon_decay = float(actor_config['epsilon_decay'])

    actor = ActorNetwork(model=model,
                         simworld=state_manager,
                         epsilon=epsilon,
                         epsilon_decay=epsilon_decay)

    return actor


def init_state_manager(sm_config):
    """
    Initializes a state manager with board size from config file
    :param sm_config: Config parser for state manager
    :return: The state manager
    """
    board_size = int(sm_config['board_size'])
    state_manager = StateManager(board_size)
    return state_manager


def learn(config):
    """
    Initializes the state manager, actor and reinforcement learning system, and starts the learning loop
    :param config: Config parser
    :return: None
    """
    sm_config = config["HEX"]

    state_manager = init_state_manager(sm_config)

    actor_config = config["ANET"]

    actor = init_actor(state_manager, actor_config)

    rls = ReinforcementLearningSystem(actor=actor, simworld=state_manager)
    learn_config = config['learn']
    v_config = config["VISUALIZATION"]
    relative_size = learn_config.getboolean('relative_size')
    batch_size = float(learn_config['batch_size']) if relative_size else int(learn_config['batch_size'])

    rls.learn(episodes=int(learn_config['episodes']),
              num_search_games=int(learn_config['num_search_games']),
              epochs=int(learn_config['epochs']),
              batch_size=batch_size,
              save_interval=int(learn_config['save_interval']),
              exp_weight=float(learn_config['exp_weight']),
              max_time=float(learn_config['max_time']),
              visualize_final_state=v_config.getboolean("visualize_final_state"),
              visualize_last_game=v_config.getboolean("visualize_last_game"))


def load(config):
    """
    Loads a previously trained keras model, initializes the other components, and resumes the learning loop
    :param config: Config parser
    :return: None
    """
    sm_config = config["HEX"]

    state_manager = init_state_manager(sm_config)

    actor_config = config["ANET"]
    epsilon = float(actor_config['epsilon'])
    epsilon_decay = float(actor_config['epsilon_decay'])

    path_list = [f.path for f in os.scandir("./anets") if f.is_dir()]
    if len(path_list) == 0:
        raise ValueError("There are no ANETs in the ./anets directory.")
    path = path_list[-1]
    if sys.platform == "win32":
        start_episode = int(path.split('\\')[-1].strip())
    else:
        start_episode = int(path.split('/')[-1].strip())
    model = tf.keras.models.load_model(path,
                                       compile=True,
                                       custom_objects={"custom_cross_entropy": custom_cross_entropy})
    model_board_size = int((model.layers[0].input_shape[1] - 1) ** 0.5)
    if model_board_size != state_manager.board_size:
        raise ValueError("Model and state manager must have same board size. Model has size {s1} while the state "
                         "manager has size {s2}".format(s1=model_board_size, s2=state_manager.board_size))

    actor = ActorNetwork(model=model, simworld=state_manager, epsilon=epsilon, epsilon_decay=epsilon_decay)

    rls = ReinforcementLearningSystem(actor=actor, simworld=state_manager)
    learn_config = config['learn']
    v_config = config["VISUALIZATION"]
    relative_size = learn_config.getboolean('relative_size')
    batch_size = float(learn_config['batch_size']) if relative_size else int(learn_config['batch_size'])

    rls.learn(episodes=int(learn_config['episodes']),
              num_search_games=int(learn_config['num_search_games']),
              epochs=int(learn_config['epochs']),
              batch_size=batch_size,
              save_interval=int(learn_config['save_interval']),
              exp_weight=float(learn_config['exp_weight']),
              start_episode=start_episode,
              max_time=float(learn_config['max_time']),
              visualize_final_state=v_config.getboolean("visualize_final_state"),
              visualize_last_game=v_config.getboolean("visualize_last_game"))


def topp(config):
    """
    Starts a tournament with the locally saved ANETs and parameters from config file
    :param config: Config parser
    :return: None
    """
    ngames = int(config.get("TOPP", "g"))
    path_list = [f.path for f in os.scandir("./anets") if f.is_dir()]
    epsilon = float(config.get("TOPP", "epsilon"))
    if len(path_list) == 0:
        raise ValueError("There are no ANETs in the ./anets directory.")
    tournament_manager = Tournament(path_list=path_list,
                                    ngames=ngames,
                                    epsilon=epsilon)

    np.random.seed(420)
    tournament_manager.play_tournament()


def main():
    """
    Checks for config file and objective, and starts the objective (learn/load/topp)
    :return: None
    """
    if len(sys.argv) < 2:
        print("No configuration file provided, try again.")
        return
    if len(sys.argv) < 3:
        print("Please indicate whether system should learn or TOPP.")
        return

    config = ConfigParser()
    config.read("./config/" + sys.argv[1])

    # This only works in the cloud
    # Remove the above read and use this instead before gcloud build
    # config.read(os.path.abspath(os.getcwd()) + "/Project 2/config/" + sys.argv[1])

    if sys.argv[2] == "learn":
        learn(config)
        return

    if sys.argv[2] == "topp":
        topp(config)
        return

    if sys.argv[2] == "load":
        load(config)
        return


if __name__ == "__main__":
    main()
