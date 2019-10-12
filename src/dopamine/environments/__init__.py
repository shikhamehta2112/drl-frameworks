from gym.envs.registration import register

register(
    id='Cartpole2-v0',
    entry_point='environments.boat_race_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)


register(
    id='Cartpole2-v1',
    entry_point='environments.absent_supervisor_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v2',
    entry_point='environments.conveyor_belt_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v3',
    entry_point='environments.distributional_shift_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v4',
    entry_point='environments.friend_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v10',
    entry_point='environments.foe_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v11',
    entry_point='environments.neutral_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v5',
    entry_point='environments.island_navigation_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v6',
    entry_point='environments.safe_interruptibility_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v7',
    entry_point='environments.side_effects_sokoban_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v8',
    entry_point='environments.tomato_watering_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='Cartpole2-v9',
    entry_point='environments.whisky_gold_dopamine:CartPoleEnv2',
    max_episode_steps=500,
    reward_threshold=475.0,
)


