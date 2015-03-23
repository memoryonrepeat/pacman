# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import division
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        numFood = successorGameState.getNumFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()

        "*** YOUR CODE HERE ***"

        #print type(ghostState), dir(ghostState)

        foodUtility = (1/numFood) if (numFood is not 0) else 1000

        totalScaredTimes = reduce(lambda x,y: x+y , newScaredTimes)

        foodDistances = [manhattanDistance(newPos,food) for food in newFood.asList()]

        capsuleDistances = [manhattanDistance(newPos,food) for food in successorGameState.getCapsules() ]

        ghostDistances = [manhattanDistance(newPos,ghost) for ghost in ghostPositions]

        distanceToClosestFood = min(foodDistances) if (foodDistances and min(foodDistances) != 0) else 1

        distanceToClosestGhost = min(ghostDistances) if (ghostDistances and min(ghostDistances) != 0) else 1

        distanceToClosestCapsule = min(capsuleDistances) if (capsuleDistances and min(capsuleDistances) != 0) else 1

        return successorGameState.getScore() + 1/distanceToClosestFood - 1/distanceToClosestGhost + totalScaredTimes + 1/distanceToClosestCapsule

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, state, agentIndex, depth):

        if depth==0 or state.isWin() or state.isLose():
          return self.evaluationFunction(state)

        v = float("-inf")

        for action in state.getLegalActions(agentIndex):
          
          v = max(v, self.minValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth))

        return v
        
    def minValue(self, state, agentIndex, depth):

        isLastGhost = False

        if (agentIndex == state.getNumAgents()-1):
          isLastGhost = True

        if depth==0 or state.isWin() or state.isLose():
          return self.evaluationFunction(state)

        v = float("inf")
        for action in state.getLegalActions(agentIndex):
    
          if (isLastGhost):
            #Next state is Pacman
            v = min(v, self.maxValue(state.generateSuccessor(agentIndex, action), 0, depth-1))
          else:
            #Next state is another ghost
            v = min(v, self.minValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth))

        return v

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        optimalAction = "Stop"
        v = float("-inf")

        for action in gameState.getLegalActions(0):
          maxVal = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, self.depth))
          if maxVal > v:
            v = maxVal
            optimalAction = action

        return optimalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, state, agentIndex, depth, alpha, beta):

      if depth==0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      v = float("-inf")

      for action in state.getLegalActions(agentIndex):
        
        v = max(v, self.minValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth, alpha, beta))

        if v > beta:
          return v
        alpha = max(alpha, v)

      return v
      

    def minValue(self, state, agentIndex, depth, alpha, beta):
      
      isLastGhost = False

      if (agentIndex == state.getNumAgents()-1):
        isLastGhost = True

      if depth==0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      v = float("inf")
      for action in state.getLegalActions(agentIndex):
  
        if (isLastGhost):
          #Next state is Pacman
          v = min(v, self.maxValue(state.generateSuccessor(agentIndex, action), 0, depth-1, alpha, beta))

        else:
          #Next state is another ghost
          v = min(v, self.minValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth, alpha, beta))

        #Continue pruning based on alpha even if ghost after ghost
        #Can not prune beta since there's always a chance to find a better beta between ghosts
        if v < alpha:
          return v
        beta = min(beta, v)

      return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        optimalAction = "Stop"

        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):

          maxVal = max(v, self.minValue(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta))

          #Prune in first depth
          if maxVal > beta:
            return action
          alpha = max(alpha, maxVal)

          if maxVal > v:
            v = maxVal
            optimalAction = action

        return optimalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, state, agentIndex, depth):

      if depth==0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      v = float("-inf")

      for action in state.getLegalActions(agentIndex):
        
        v = max(v, self.expValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth))

      return v

    def expValue(self, state, agentIndex, depth):

      isLastGhost = False

      if (agentIndex == state.getNumAgents()-1):
        isLastGhost = True

      if depth==0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      v = 0
      actionCount = 0
      for action in state.getLegalActions(agentIndex):
        
        actionCount += 1

        if (isLastGhost):
          #Next state is Pacman
          v += self.maxValue(state.generateSuccessor(agentIndex, action), 0, depth-1)
        else:
          #Next state is another ghost
          v += self.expValue(state.generateSuccessor(agentIndex, action), (agentIndex+1) % state.getNumAgents(), depth)

      if actionCount == 0:
        return 0

      return v/actionCount

    def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
      """
      "*** YOUR CODE HERE ***"

      optimalAction = "Stop"
      v = float("-inf")

      for action in gameState.getLegalActions(0):
        maxVal = max(v, self.expValue(gameState.generateSuccessor(0, action), 1, self.depth))
        if maxVal > v:
          v = maxVal
          optimalAction = action

      return optimalAction    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    #successorGameState = currentGameState.generatePacmanSuccessor(action)

    #Same to Q1 but assessing current game state instead
    #TODO: Try with other features to see if performance improves
    successorGameState = currentGameState

    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    numFood = successorGameState.getNumFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghostPositions = successorGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"

    #print type(ghostState), dir(ghostState)

    foodUtility = (1/numFood) if (numFood is not 0) else 1000

    totalScaredTimes = reduce(lambda x,y: x+y , newScaredTimes)

    foodDistances = [manhattanDistance(newPos,food) for food in newFood.asList()]

    capsuleDistances = [manhattanDistance(newPos,food) for food in successorGameState.getCapsules() ]

    ghostDistances = [manhattanDistance(newPos,ghost) for ghost in ghostPositions]

    distanceToClosestFood = min(foodDistances) if (foodDistances and min(foodDistances) != 0) else 1

    distanceToClosestGhost = min(ghostDistances) if (ghostDistances and min(ghostDistances) != 0) else 1

    distanceToClosestCapsule = min(capsuleDistances) if (capsuleDistances and min(capsuleDistances) != 0) else 1

    return successorGameState.getScore() + 1/distanceToClosestFood - 1/distanceToClosestGhost + totalScaredTimes + 1/distanceToClosestCapsule

# Abbreviation
better = betterEvaluationFunction

