
import gin
import tensorflow as tf
import time
import sys

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics


@gin.configurable
class CheckpointRunner(run_experiment.Runner):
    """
    Object that handles running Dopamine experiments.
    Extends Dopamine runner and allows to control the frequency of checkpointing.
    """

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 create_environment_fn=atari_lib.create_atari_environment,
                 checkpoint_file_prefix='ckpt',
                 checkpoint_freq=1,
                 logging_file_prefix='log',
                 log_every_n=1,
                 num_iterations=200,
                 training_steps=250000,
                 evaluation_steps=125000,
                 max_steps_per_episode=27000,
                 inference_steps=None):
        super(CheckpointRunner, self).__init__(base_dir,
                                               create_agent_fn,
                                               create_environment_fn,
                                               checkpoint_file_prefix,
                                               logging_file_prefix,
                                               log_every_n,
                                               num_iterations,
                                               training_steps,
                                               evaluation_steps,
                                               max_steps_per_episode)
        self.checkpoint_freq = checkpoint_freq
        self.current_checkpoint = 0
        self.inference_steps = inference_steps

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        # init checkpoint number
        self.current_checkpoint = 0
        if self._num_iterations <= self._start_iteration:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            # checkpoint with given frequency and after last iteration
            if self.checkpoint_freq != 0 and ((iteration + 1) % self.checkpoint_freq == 0 or (iteration + 1) == self._num_iterations):
                self._checkpoint_experiment(iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data. Overwrite parent method to better handle checkpointing frequency.
        Args:
        iteration: int, iteration number for checkpointing.
        """
        experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                            iteration)
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            experiment_data['logs'] = self._logger.data
            self._checkpointer.save_checkpoint(
                self.current_checkpoint, experiment_data)
            self.current_checkpoint = self.current_checkpoint + 1

    def run_inference_test(self):
        statistics = iteration_statistics.IterationStatistics()
        _ = self._run_one_phase(
            self.inference_steps, statistics, 'eval')

    def _run_one_iteration(self, iteration):
        print("one iteration")
        """Runs one iteration of agent/environment interaction.
        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).
        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.
        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        num_episodes_train, average_reward_train, avg_perf_train = self._run_train_phase(
            statistics)
        num_episodes_eval, average_reward_eval, avg_perf_eval = self._run_eval_phase(
            statistics)

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train, num_episodes_eval,
                                         average_reward_eval, avg_perf_train, avg_perf_eval)
        return statistics.data_lists

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    num_episodes_eval,
                                    average_reward_eval,
                                    avg_perf_train,
                                    avg_perf_eval):
        """Save statistics as tensorboard summaries.
        Args:
          iteration: int, The current iteration number.
          num_episodes_train: int, number of training episodes run.
          average_reward_train: float, The average training reward.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval),
            tf.Summary.Value(tag='Train/avg_perf_train',
                             simple_value=avg_perf_train),
            tf.Summary.Value(tag='Eval/avg_perf_eval',
                             simple_value=avg_perf_eval)
        ])

        self._summary_writer.add_summary(summary, iteration)


    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.
        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.
        Args:
          min_steps: int, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str: str, describes the run mode for this agent.
        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.
        sum_perf= 0
        while step_count < min_steps:
           episode_length, episode_return = self._run_one_episode()
           statistics.append({
            '{}_episode_lengths'.format(run_mode_str): episode_length,
            '{}_episode_returns'.format(run_mode_str): episode_return
                         })
           episode_perf= self._environment.hidden_reward
           sum_perf += episode_perf
           step_count += episode_length
           sum_returns += episode_return
           num_episodes += 1
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
        sys.stdout.flush()
        return step_count, sum_returns, num_episodes, sum_perf

    def _run_train_phase(self, statistics):
        print("train phase")
        """Run training phase.
        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.
        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: The average reward generated in this phase.
        """
        # Perform the training phase, during which the agent learns.
        self._agent.eval_mode = False
        start_time = time.time()
        number_steps, sum_returns, num_episodes, sum_perf = self._run_one_phase(
           self._training_steps, statistics, 'train')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        average_perf = sum_perf / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        time_delta = time.time() - start_time
        tf.logging.info('Average undiscounted return per training episode: %.2f',
                     average_return)
        tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
        return num_episodes, average_return, average_perf

    def _run_eval_phase(self, statistics):
        print("eval phase")
        """Run evaluation phase.
        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.
        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes, sum_perf = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        average_perf = sum_perf / num_episodes if num_episodes > 0 else 0.0
        tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                     average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return, average_perf

def create_runner(base_dir):
    """Creates an experiment CheckpointRunner.
    Args:
        base_dir: str, base directory for hosting all subdirectories.
    Returns:
        runner: A `CheckpointRunner` like object.
    """
    return CheckpointRunner(base_dir, run_experiment.create_agent)

