import numpy as np
from typing import Dict, Literal, Optional, Tuple
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
    episode_info:Dict[str, list[Literal['avg','sum'],any]] = {}

    state, _ = env.reset()
    
    # obs = env.parse_observation(state)
    # env.get_logger().info('-----------------------------------')
    # env.get_logger().info('\n' + obs)
    # env.get_logger().info('-----------------------------------')

    for step_counter in range(max_steps_training):
        episode_timesteps += 1

        # select and action
        if step_counter < max_steps_exploration:
            # env.get_logger().info(f'Exploration Step #{step_counter} out of {max_steps_exploration}')
            action_env = np.random.uniform(env.MIN_ACTIONS, env.MAX_ACTIONS, env.ACTION_NUM)
            action = hlp.normalize(action_env, env.MAX_ACTIONS, env.MIN_ACTIONS)
        else:
            action = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)

        
        # perform action, step environment, and setup for next step
        next_state, reward, done, truncated, step_info = env.step(action_env)
        memory.add(state, action, reward, next_state, done)
        state = next_state

        # record log relevant information
        episode_reward += reward
        for info_key, info_content in step_info.items():
            # step info has form: { "info name": ["aggregate behavior type", value] }
            if info_key in episode_info:
                episode_info[info_key][1] += info_content[1]
            else:
                episode_info[info_key] = info_content
        
        # train agent
        if step_counter >= max_steps_exploration:
            for i in range(G):
                info = agent.train_policy(memory,batch_size)
        
        # handle if should evaluate at end of episode
        if (step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True
            record.save_agent(str(step_counter+1), "agent")
        
        # handle end of episode
        if done or truncated:
            # aggregate episode information
            # step info has form: { "info name": ("aggregate behavior type", value) }
            for info_key, info_content in episode_info.items():
                match info_content[0]:
                    case 'avg':
                        episode_info[info_key] = info_content[1]/(episode_timesteps+1)
                    case 'sum':
                        episode_info[info_key] = info_content[1]
            if episode_timesteps > 2:
                record.log_train(
                    total_steps = step_counter + 1,
                    episode = episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward = episode_reward,
                    display = True,
                    **episode_info
                )
                episode_num += 1

            if evaluate:
                evaluate = False
                env.get_logger().info(f'*************--Begin Evaluation Loop--*************')
                off_policy_evaluate(env, agent, number_eval_episodes, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            
            episode_info = {}
        

def off_policy_evaluate(env, agent, eval_episodes, record=None, steps_counter=0):

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_info:Dict[str, list[Literal['avg','sum'],any]] = {}

    # put environment in evaluation mode
    env.start_eval()
    
    for episode_num in range(eval_episodes):
        
        state, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1

            # select and perform action, setup for next step
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)
            with open("network_output.csv", 'a') as f:
                f.write(f"{episode_num},{episode_timesteps},{action[0]:.4f},{action[1]:.4f}\n")
            next_state, reward, done, truncated, step_info = env.step(action_env)
            state = next_state

            # record relevant log information
            episode_reward += reward
            for info_key, info_content in step_info.items():
                # step info has form: { "info name": ["aggregate behavior type", value] }
                if info_key in episode_info:
                    episode_info[info_key][1] += info_content[1]
                else:
                    episode_info[info_key] = info_content
            
            # handle end of eval episode
            if done or truncated:
                if record:
                    # aggregate episode information
                    # step info has form: { "info name": ("aggregate behavior type", value) }
                    for info_key, info_content in episode_info.items():
                        match info_content[0]:
                            case 'avg':
                                episode_info[info_key] = info_content[1]/(episode_timesteps+1)
                            case 'sum':
                                episode_info[info_key] = info_content[1]

                    record.log_eval(
                        total_steps = steps_counter + 1,
                        episode = episode_num + 1,
                        episode_steps=episode_timesteps,
                        episode_reward = episode_reward,
                        display = True,
                        **episode_info
                    )

                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                episode_info = {}
                break
    
    env.stop_eval()

def multi_off_policy_train(env, env2, agent, memory, record, algorithm_config):
    
    max_steps_training = algorithm_config['max_steps_training']
    max_steps_exploration = algorithm_config['max_steps_exploration']
    number_eval_episodes = algorithm_config['number_eval_episodes']
    number_steps_per_evaluation = algorithm_config['number_steps_per_evaluation']

    batch_size = algorithm_config['batch_size']
    G = algorithm_config['g']

    episode_timesteps = 0
    episode_reward1 = 0
    episode_reward2 = 0
    episode_num = 0
    evaluate = False
    episode_info:Dict[str, list[Literal['avg','sum'],any]] = {}

    state1, _ = env.reset()
    state2, _ = env2.reset()
    
    # obs = env.parse_observation(state)
    # env.get_logger().info('-----------------------------------')
    # env.get_logger().info('\n' + obs)
    # env.get_logger().info('-----------------------------------')

    for step_counter in range(max_steps_training):
        episode_timesteps += 1

        # select and action
        if step_counter < max_steps_exploration:
            # env.get_logger().info(f'Exploration Step #{step_counter} out of {max_steps_exploration}')
            action_env1 = np.random.uniform(env.MIN_ACTIONS, env.MAX_ACTIONS, env.ACTION_NUM)
            action1 = hlp.normalize(action_env1, env.MAX_ACTIONS, env.MIN_ACTIONS)
            action_env2 = np.random.uniform(env2.MIN_ACTIONS, env2.MAX_ACTIONS, env2.ACTION_NUM)
            action2 = hlp.normalize(action_env2, env2.MAX_ACTIONS, env2.MIN_ACTIONS)
        else:
            action1 = agent.select_action_from_policy(state1)
            action_env1 = hlp.denormalize(action1, env.MAX_ACTIONS, env.MIN_ACTIONS)
            action2 = agent.select_action_from_policy(state2)
            action_env2 = hlp.denormalize(action2, env2.MAX_ACTIONS, env2.MIN_ACTIONS)

        
        # perform action, step environment, and setup for next step
        next_state1, reward1, done1, truncated1, step_info1 = env.step(action_env1)
        next_state2, reward2, done2, truncated2, step_info2 = env2.step(action_env2)
        memory.add(state1, action1, reward1, next_state1, done1)
        memory.add(state2, action2, reward2, next_state2, done2)
        state1 = next_state1
        state2 = next_state2

        # record log relevant information
        episode_reward1 += reward1
        episode_reward2 += reward2
        for info_key, info_content in step_info1.items():
            # step info has form: { "info name": ["aggregate behavior type", value] }
            if info_key in episode_info:
                episode_info[info_key][1] += info_content[1]
            else:
                episode_info[info_key] = info_content
        for info_key, info_content in step_info2.items():
            # step info has form: { "info name": ["aggregate behavior type", value] }
            if info_key in episode_info:
                episode_info[info_key][1] += info_content[1]
            else:
                episode_info[info_key] = info_content
        # train agent
        if step_counter >= max_steps_exploration:
            for i in range(G):
                info = agent.train_policy(memory,batch_size)
        
        # handle if should evaluate at end of episode
        if (step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True
            record.save_agent(str(step_counter+1), "agent")
        
        # handle end of episode
        if done1 or truncated1:
            # aggregate episode information
            # step info has form: { "info name": ("aggregate behavior type", value) }
            for info_key, info_content in episode_info.items():
                match info_content[0]:
                    case 'avg':
                        episode_info[info_key] = info_content[1]/(episode_timesteps+1)
                    case 'sum':
                        episode_info[info_key] = info_content[1]
            if episode_timesteps > 2:
                record.log_train(
                    total_steps = step_counter + 1,
                    episode = episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward = episode_reward1,
                    display = True,
                    **episode_info
                )
                episode_num += 1

            if evaluate:
                evaluate = False
                env.get_logger().info(f'*************--Begin Evaluation Loop--*************')
                multi_off_policy_evaluate(env, env2, agent, number_eval_episodes, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')

            # Reset environment
            state1, _ = env.reset()
            episode_reward1 = 0
            episode_timesteps = 0
            
            episode_info = {}
        if done2 or truncated2:
            # aggregate episode information
            # step info has form: { "info name": ("aggregate behavior type", value) }
            for info_key, info_content in episode_info.items():
                match info_content[0]:
                    case 'avg':
                        episode_info[info_key] = info_content[1]/(episode_timesteps+1)
                    case 'sum':
                        episode_info[info_key] = info_content[1]
            if episode_timesteps > 2:
                record.log_train(
                    total_steps = step_counter + 1,
                    episode = episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward = episode_reward2,
                    display = True,
                    **episode_info
                )
                episode_num += 1

            if evaluate:
                evaluate = False
                env.get_logger().info(f'*************--Begin Evaluation Loop--*************')
                multi_off_policy_evaluate(env, env2, agent, number_eval_episodes, record, step_counter)
                env.get_logger().info(f'*************--End Evaluation Loop--*************')

            # Reset environment
            state2, _ = env2.reset()
            episode_reward2 = 0
            episode_timesteps = 0
            
            episode_info = {}
        

def multi_off_policy_evaluate(env, agent, eval_episodes, record=None, steps_counter=0):

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_info:Dict[str, list[Literal['avg','sum'],any]] = {}

    # put environment in evaluation mode
    env.start_eval()
    
    for episode_num in range(eval_episodes):
        
        state, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1

            # select and perform action, setup for next step
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(action, env.MAX_ACTIONS, env.MIN_ACTIONS)
            with open("network_output.csv", 'a') as f:
                f.write(f"{episode_num},{episode_timesteps},{action[0]:.4f},{action[1]:.4f}\n")
            next_state, reward, done, truncated, step_info = env.step(action_env)
            state = next_state

            # record relevant log information
            episode_reward += reward
            for info_key, info_content in step_info.items():
                # step info has form: { "info name": ["aggregate behavior type", value] }
                if info_key in episode_info:
                    episode_info[info_key][1] += info_content[1]
                else:
                    episode_info[info_key] = info_content
            
            # handle end of eval episode
            if done or truncated:
                if record:
                    # aggregate episode information
                    # step info has form: { "info name": ("aggregate behavior type", value) }
                    for info_key, info_content in episode_info.items():
                        match info_content[0]:
                            case 'avg':
                                episode_info[info_key] = info_content[1]/(episode_timesteps+1)
                            case 'sum':
                                episode_info[info_key] = info_content[1]

                    record.log_eval(
                        total_steps = steps_counter + 1,
                        episode = episode_num + 1,
                        episode_steps=episode_timesteps,
                        episode_reward = episode_reward,
                        display = True,
                        **episode_info
                    )

                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                episode_info = {}
                break
    
    env.stop_eval()    

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
        memory.add(state, action, reward, next_state, done, log_prob)

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