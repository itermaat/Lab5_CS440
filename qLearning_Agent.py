import random
from collections import defaultdict
from agent_strategy4 import ACTION_DELTAS

class QLearningAgent:
    # Set Paramaters
        # Learning Rate -> how much the agent changes Q values (0 means nothing changes, 1 means replacing with all new information)
        # Discount Factor -> future rewards vs immediate rewards (scale of 0 to 1)
        # Epsilon -> whether agent explores or follow what it's learned
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # store learned qvals
        self.qtable = defaultdict(float)

    def chooseAction(self, state, validActions):
        # chose random action
        if random.random() < self.epsilon:
            return random.choice(validActions)

        # chose action with highest qval
        bestVal = max(self.qtable[(state, action)] for action in validActions)

        bestActions = [
            action for action in validActions
            if self.qtable[(state, action)] == bestVal
        ]

        return random.choice(bestActions)
    
    def updateQVal(self, state, action, reward, nextState, nextActions):
        # get current qval
        oldQ = self.qtable[(state, action)]

        # find max future
        if nextActions:
            maxFutureQ = max(
                self.qtable[(nextState, i)] for i in nextActions
            )
        else:
            maxFutureQ = 0

        # apply q learning formula
        newQ = oldQ + self.alpha * (
            reward + self.gamma * maxFutureQ - oldQ
        )

        # Save updated qval
        self.qtable[(state, action)] = newQ


    def getActions(self, environment):
        
        row, column = environment.pos
        validActions = []

        for action, (dr, dc) in ACTION_DELTAS.items():
            nextRow = row + dr
            nextColumn = column + dc

            if (
                0 <= nextRow < len(environment.grid)
                and 0 <= nextColumn < len(environment.grid[nextRow])
                and environment.grid[nextRow][nextColumn] != "X"
            ):
                validActions.append(action)

        return validActions
    


