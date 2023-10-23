import numpy as np

from cares_reinforcement_learning.util import helpers as hlp

def off_policy_train(env, agent, memory, record, algorithm_config):
    
    max_steps_training = algorithm_config['max_steps_training']
    max_steps_exploration = algorithm_config['max_steps_exploration']
    num_eps_evaluation = algorithm_config['evaluate_for_m_episodes']
    evaluate_every_n_steps = algorithm_config['evaluate_every_n_steps']

    batch_size = algorithm_config['batch_size']
    G = algorithm_config['g']

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    evaluate = False

    state, _ = env.reset()

    for step_counter in range(max_steps_training):
        episode_timesteps += 1

        if step_counter < max_steps_exploration:
            env.get_logger().info(f'Exploration Step #{step_counter} out of {max_steps_exploration}')
            action_env = np.random.uniform(env.MIN_ACTIONS, env.MAX_ACTIONS, env.ACTION_NUM)
            action = hlp.normalize(action_env, env.MAX_ACTIONS, env.MIN_ACTIONS)
        else:
            action = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)
        
        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if step_counter >= max_steps_exploration:
            for i in range(G):
                experience = memory.sample(batch_size)
                info = agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                ))
                memory.update_priorities(experience['indices'], info)
        
        if step_counter % evaluate_every_n_steps == 0:
            evaluate = True
        
        if done or truncated:
            env.get_logger().info(f'Episode #{episode_num} completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}')

            record.log_train(
                total_steps = step_counter + 1,
                episode = episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward = episode_reward,
                display = True
            )

            if evaluate:
                evaluate = False

                env.get_logger().info(f'*************--Begin Evaluation Loop--*************')
                off_policy_evaluate(env, agent, num_eps_evaluation, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')


            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def off_policy_evaluate(env, agent, eval_episodes, record, steps_counter):

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for episode_num in range(eval_episodes):
        state, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

            next_state, reward, done, truncated, _ = env.step(action_env)

            state = next_state
            episode_reward += reward

            if done or truncated:
                record.log_eval(
                    total_steps = steps_counter + 1,
                    episode = episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward = episode_reward,
                    display = True
                )

                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                break