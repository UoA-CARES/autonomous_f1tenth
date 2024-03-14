import numpy as np

from cares_reinforcement_learning.util import helpers as hlp

def off_policy_train(env, agent, memory, record, algorithm_config):
    
    max_steps_training = algorithm_config['max_steps_training']
    max_steps_exploration = algorithm_config['max_steps_exploration']
    number_eval_episodes = algorithm_config['number_eval_episodes']
    number_steps_per_evaluation = algorithm_config['number_steps_per_evaluation']

    batch_size = algorithm_config['batch_size']
    G = algorithm_config['g']

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    evaluate = False

    state, _ = env.reset()
    
    obs = env.parse_observation(state)
    env.get_logger().info('-----------------------------------')
    env.get_logger().info('\n' + obs)
    env.get_logger().info('-----------------------------------')

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
        memory.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if step_counter >= max_steps_exploration:
            for i in range(G):
                experience = memory.sample(batch_size)
                info = agent.train_policy(experience)
                #memory.update_priorities(experience['indices'], info)
        
        if (step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True
        
        if done or truncated:
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
                off_policy_evaluate(env, agent, number_eval_episodes, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')


            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def off_policy_evaluate(env, agent, eval_episodes, record=None, steps_counter=0):

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

                if record:
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



def ppo_train(env, agent, memory, record, algorithm_config):

    max_steps_training = algorithm_config['max_steps_training']
    max_steps_per_batch = algorithm_config['max_steps_per_batch']
    number_eval_episodes = algorithm_config['number_eval_episodes']
    number_steps_per_evaluation = algorithm_config['number_steps_per_evaluation']

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0

    evaluate = False

    state, _ = env.reset()

    obs = env.parse_observation(state)
    env.get_logger().info('-----------------------------------')
    env.get_logger().info('\n' + obs)
    env.get_logger().info('-----------------------------------')

    for step_counter in range(max_steps_training):
        episode_timesteps += 1

        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob)

        state = next_state
        episode_reward += reward

        if (step_counter+1) % max_steps_per_batch == 0:
            experience = memory.flush()
            info = agent.train_policy((
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done'],
                experience['log_prob']
            ))
        
        if (step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True
        
        if done or truncated:
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
                ppo_evaluate(env, agent, number_eval_episodes, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')


            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def ppo_evaluate(env, agent, eval_episodes, record=None, steps_counter=0):

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for episode_num in range(eval_episodes):
        state, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            action, _ = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

            next_state, reward, done, truncated, _ = env.step(action_env)

            state = next_state
            episode_reward += reward

            if done or truncated:

                if record:
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