#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import time
import yaml

import ray
from ray.tune.config_parser import make_parser
import numpy as np
from custom_trainer import get_agent_class
from ray import tune

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    ./run_evaluation -f src/ray/experiments/cartpole/ray_dqn_cpu_cp1.yml
"""


class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_summary(self, average_reward_train, num_episodes_train, average_hreward_train, average_reward_eval, num_episodes_eval, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Train/AverageHiddenReturns',
                             simple_value=average_hreward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)
        ])
        self.writer.add_summary(summary, iteration)
        self.writer.flush()


def create_parser(parser_creator=None):
    """ Creates argument parser."""
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser

def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["hidden_reward"] = []

def on_episode_step(info):
    episode = info["episode"]
    if episode.last_info_for()!=None and "hidden_reward" in episode.last_info_for():
      episode.user_data["hidden_reward"].append(episode.last_info_for()["hidden_reward"])

def on_episode_end(info):
    episode = info["episode"]
    hidden_reward = np.mean(episode.user_data["hidden_reward"])
    episode.custom_metrics["hidden_reward"] = hidden_reward

def run(args, parser):
    # Load configuration file
    with open(args.config_file) as f:
        experiments = yaml.load(f)

    # extract info about experiment
    experiment_name = list(experiments.keys())[0]
    experiment_info = list(experiments.values())[0]

    agent_name = experiment_info["run"]
    env_name = experiment_info["env"]
    results_dir = experiment_info['local_dir']
    checkpoint_freq = experiment_info["checkpoint_freq"]
    checkpoint_at_end = experiment_info["checkpoint_at_end"]
    checkpoint_dir = os.path.join(results_dir, experiment_name)
    num_iterations = experiment_info["stop"]["training_iteration"]
    config = experiment_info["config"]
    config["callbacks"]= {
        "on_episode_start": tune.function(on_episode_start),
                "on_episode_step": tune.function(on_episode_step),
                "on_episode_end": tune.function(on_episode_end),
    }
   # training_steps = experiment_info["agent_training_steps"]
   # evaluation_steps = experiment_info["agent_evaluation_steps"]

       # init training agent
    ray.init()
    agent_class = get_agent_class(agent_name)
    agent = agent_class(env=env_name, config=config)
    if agent_name == 'APPO':
        agent.set_timesteps_per_iteration(
            experiment_info["appo_timesteps_per_iteration"])
    average_reward_train, train_episodes = [], []
    average_reward_eval, eval_episodes = [], []
    timesteps_history = []

    # log results to tensorboard
    tensorboard = Tensorboard(os.path.join(results_dir, experiment_name))
    start_time = time.time()
    for iteration in range(num_iterations):
            # train agent
        train_result = agent.train()
        timesteps_history.append(train_result["timesteps_total"])
        average_reward_train.append(train_result["episode_reward_mean"])
        train_episodes.append(train_result["episodes_this_iter"])

        # evaluate agent
        eval_result = agent._evaluate()
        average_reward_eval.append(
            eval_result["evaluation"]["episode_reward_mean"])
        eval_episodes.append(eval_result["evaluation"]["episodes_this_iter"])

        # checkpoint agent's state
        if iteration % checkpoint_freq == 0:
            agent.save(checkpoint_dir)

        tensorboard.log_summary(
            train_result["episode_reward_mean"], train_result["episodes_this_iter"], train_result["custom_metrics"]["hidden_reward_mean"], eval_result["evaluation"]["episode_reward_mean"], eval_result["evaluation"]["episodes_this_iter"], iteration)

    # checkpoint agent's last state
    print("ABOUT TO CHECK POINT")
    if checkpoint_at_end and num_iterations>0:
        agent.save(checkpoint_dir)
        print("Saved in"+checkpoint_dir)
    else:
        print("NO CHECKPOINTS")
    end_time = time.time()

    
#    for i in range(len(average_reward_eval)):
#        tensorboard.log_summary(
#            average_reward_train[i], train_episodes[i], average_reward_eval[i], eval_episodes[i], i)


    # save runtime
    runtime_file = os.path.join(results_dir, 'runtime', 'runtime.csv')
    f = open(runtime_file, 'a+')
    f.write(experiment_name + ', ' +
            str(end_time - start_time) + '\n')
    f.close()
    if int(experiment_info["inference_steps"]>0):
    # inference testing
        try:
            inference_steps = experiment_info["inference_steps"]
            print("--- STARTING RAY CARTPOLE INFERENCE EXPERIMENT ---")
            eval_result = agent._evaluate()
            average_reward_eval=[]
            eval_episodes=[]
            average_reward_eval.append(
            eval_result["evaluation"]["episode_reward_mean"])
            eval_episodes.append(eval_result["evaluation"]["episodes_this_iter"])
            tensorboard.log_summary(
            0, 0, 0, eval_result["evaluation"]["episode_reward_mean"], eval_result["evaluation"]["episodes_this_iter"], 1001)
            print("--- RAY CARTPOLE INFERENCE EXPERIMENT COMPLETED ---")
        except KeyError:
            pass

    tensorboard.close()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
