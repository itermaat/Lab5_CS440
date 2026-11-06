from agent_strategy4 import VacuumEnvironment, INITIAL_GRID
from qLearning_Agent import QLearningAgent

def trainVacuum(episodes=1000, living_reward = -0.04,
                learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    agent = QLearningAgent(learning_rate, discount_factor, epsilon)

    for episode in range(episodes):
        # Reset the environment for each training episode
        env = VacuumEnvironment(INITIAL_GRID)
        state = env.pos

        for step in range(100):
            validActions = agent.getActions(env)
            action = agent.chooseAction(state, validActions)

            # perform chosen action
            env.step(action)
            nextState = env.pos

            # calculate reward
            reward = living_reward
            cell_value = env.grid[nextState[0]][nextState[1]]

            if isinstance(cell_value, (int, float)):
                reward += cell_value

            # see if agent reached goal
            done = nextState == (3, 3)

            # update q table
            nextActions = (
                [] if done else agent.getActions(env)
            )

            agent.updateQVal(
                state, action, reward, nextState, nextActions
            )

            state = nextState

            if done:
                break

    return agent



