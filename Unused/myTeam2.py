# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

from math import sqrt, log

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ProcrastinateAgent', second = 'ProcrastinateAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ProcrastinateAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)
    ''' 
    Your initialization code goes here, if you need any.
    '''

    # Store team and enemy indices
    self.teamIndices = self.getTeam(gameState)
    self.enemyIndices = self.getOpponents(gameState)

    # Check how recently we were near the enemy to check if we've knocked him out
    self.nearEnemyCounter = 0

    # Set up particle filters to track enemy locations
    self.enemyLocFilters = {}
    for i in self.enemyIndices:
      self.enemyLocFilters[i] = (ParticleFilter(gameState, i,
                              gameState.getInitialAgentPosition(i)))

  def chooseAction(self, gameState):
    """
    Makes a random move only when it absolutely has to.
    Annoying, ain't it?
    """
    startTime = time.time()

    # Populate particle filters with most recent data
    actions = getLegalActionsNoStop(gameState, self.index)
    for i, f in self.enemyLocFilters.iteritems():
      exactPosition = gameState.getAgentPosition(i)
      # If we can see the enemy, cluster all particles on his position
      if exactPosition:
        f.clusterParticles(exactPosition)
        self.nearEnemyCounter = 2
      else: # Otherwise, run particle filtering calculations
        f.elapseTime(gameState)
        f.observe(gameState, gameState.getAgentPosition(self.index), self.nearEnemyCounter > 0)
        if (self.nearEnemyCounter > 0): self.nearEnemyCounter -= 1
    
    dist = util.Counter()
    l = 'Bottom' if self.isBottom else 'Top'
    for food in self.foodLists[l]: 
      dist[food] += 1
    dist.normalize()
    self.displayDistributionsOverPositions([dist])

    # Remove old food
    foodLocs = self.getFood(gameState).asList()
    for _, l in self.foodLists.iteritems():
      missingFood = [l1 for l1 in l if l1 not in foodLocs]
      for food in missingFood:
        l.remove(food)

    # """ For debugging purposes
    self.displayDistributionsOverPositions(
        [f.getBeliefDistribution() for i, f in self.enemyLocFilters.iteritems()])
    # """
    return self.runTopSecretAlgorithm(startTime, gameState)

  """
  The top secret algorithm is a heavy-playout Monte Carlo Search Tree using UCT.  Don't tell anyone.

  Structurally, this was informed somewhat by the mcts.ai implementation:
    http://mcts.ai/code/index.html
  However, their implementation is fairly general, and I was careful not to study it too closely.
  """
  def runTopSecretAlgorithm(self, startTime, gameState):
    currIndex = self.index


    # Copy current state and set all ghost positions to most likely values
    currState = gameState.deepCopy()
    for i in self.enemyIndices:
      mostLikelyPos = self.enemyLocFilters[i].getMostLikelyPos()
      conf = game.Configuration(mostLikelyPos, game.Directions.STOP)
      currState.data.agentStates[i] = game.AgentState(conf, False)

    startTimeLeft = currState.data.timeleft

    currNode = Node(None, None, currState, currIndex)  # Create root node

    currTime = time.time()
    while (currTime - startTime < 2.6):
      """
      Selection
      """
      # Select child nodes based on UCT while all children are explored
      while currNode.children and not currNode.unbornChildren:
        currNode = currNode.chooseChild()

      """
      Expansion
      """
      # If the node has an unexplored child, create it and go to that child
      if currNode.unbornChildren:
        currNode = currNode.birthChild(self.getNextIndex(gameState, currIndex))

      """
      Playout
      """
      currState = currNode.gameState
      currIndex = currNode.index
      while not currState.isOver() and currState.data.timeleft > startTimeLeft - 25:
        currState = self.executeNextMove(currState, currIndex)
        currIndex = self.getNextIndex(gameState, currIndex)

      finalScore = currState.getScore()
      foodLocations = currState.getBlueFood().asList()
      gameResult = 0.5 + 0.5 * (1 / self.getMinFoodDistance(currState.getAgentPosition(currIndex), foodLocations))
      if (finalScore > 0):
        gameResult = 1
      elif (finalScore < 0):
        gameResult = 0

      """
      Backpropagation
      """
      while currNode.parent:
        currNode.updateNode(gameResult)
        currNode = currNode.parent
      currNode.updateNode(gameResult)

      # Keep track of time
      currTime = time.time()
    bestAction = self.getBestAction(currNode)
    endTime = currTime - startTime
    if (endTime > 3):
      print(endTime)
    for child in currNode.children:
      print(child)
    print('---- bestAction: ' + bestAction)
    return bestAction

  def getBestAction(self, currNode):
    """
    Returns the action leading to the best-scoring child of the current node
    """
    bestScore = None
    bestAction = None
    for child in currNode.children:
      childScore = child.cumulativeScore
      if (bestAction is None or childScore > bestScore):
        bestScore = childScore
        bestAction = child.lastAction
    return bestAction

  def executeNextMove(self, gameState, currIndex):
    """
    Choose a move for the current agent semi-randomly based on the evaluateState() heuristic and 
    returns the resulting state
    """
    succScores = util.Counter()
    actions = getLegalActionsNoStop(gameState, currIndex)
    successors = [gameState.generateSuccessor(currIndex, a) for a in actions]
    for s in successors:
      succScores[s] = self.evaluateState(s, currIndex)
    chosenMoveState = max(succScores)
    return chosenMoveState

  def evaluateState(self, gameState, currIndex):
    """
    Given some state, return a number indicating how advantageous the state is for the current
    agent's team
    """
    score = gameState.getScore()
    scoreScore = score * 100
    # If current 
    if currIndex in gameState.getRedTeamIndices():
      foodLocations = gameState.getBlueFood().asList()
      foodScore = 900-(self.getMinFoodDistance(gameState.getAgentPosition(currIndex), foodLocations))
      scoreScore = score * 1000
    else:
      foodLocations = gameState.getRedFood().asList()
      foodScore = -(self.getMinFoodDistance(gameState.getAgentPosition(currIndex), foodLocations))
      scoreScore = -(score * 1000)
    return foodScore + scoreScore

  def getNextIndex (self, gameState, currIndex):
    """
    Utility function to get the index of the next agent whose turn it is
    """
    nextIndex = currIndex + 1
    if (nextIndex >= gameState.getNumAgents()):
      nextIndex = 0
    return nextIndex

  def getMinFoodDistance(self, pos, foodLocations):
    foodDistances = []
    for food in foodLocations:
      foodDistances.append(self.distancer.getDistance(pos, food))
    if len(foodDistances) is 0: # Can't take min of empty list
      nearestFoodDist = 0
    else:
      nearestFoodDist = min(foodDistances)
    if nearestFoodDist is 0:
      nearestFoodDist = 1
    return nearestFoodDist

def getLegalActionsNoStop(gameState, index):
  legalActions = gameState.getLegalActions(index)
  rev = Directions.REVERSE[gameState.getAgentState(index).configuration.direction]
  goodActions = [a for a in legalActions if a != Directions.STOP and a != rev]
  if not goodActions:
    goodActions = [rev]
  return goodActions

class Node(object):
  def __init__(self, action, parent, gameState, index):
    self.lastAction = action  # The action taken to get to this node. "None" if root
    self.parent = parent      # The node leading up to the current one. "None" if root
    self.children = []        # Explored nodes branching from the current one. Empty if leaf
    self.unbornChildren = []  # Actions not yet explored that would create successor states

    self.cumulativeScore = 0  # Cumulative score earned from all games played from the node
    self.maxScore = 0         # Maximum possible score from playing that many games

    self.index = index          # Index of the player whose turn it is currently
    self.gameState = gameState  # State in the game that the node represents

    # Create unborn list as soon as you create the node
    self.populateUnborn()

  def populateUnborn(self):
    """
    Populates self.unbornChildren with a list of actions that could one day lead to children
    """
    for action in getLegalActionsNoStop(self.gameState, self.index):
      self.unbornChildren.append(action)
    random.shuffle(self.unbornChildren)

  def birthChild(self, nextPlayerIndex):
    """
    Pop an action of the unborn list and make the resulting state a new child
    """
    newAction = self.unbornChildren.pop()
    succState = self.gameState.generateSuccessor(self.index, newAction)
    baby = Node(newAction, self, succState, nextPlayerIndex)
    self.children.append(baby)
    return baby

  def updateNode(self, score):
    self.cumulativeScore += score
    self.maxScore += 1

  def chooseChild(self):
    """
    Chooses an existing child based on the UCT algorithm:
      UCTValue(parent, n) = winrate + sqrt(c + (2 ln(parent.visits))/(n.nodevisits))

    Source: http://senseis.xmp.net/?UCT
    """
    bestChild = None
    bestValue = None
    expConst = 2      # Exploration constant
    for c in self.children:
      currScore = (c.cumulativeScore / c.maxScore)
      currValue = currScore + sqrt(expConst + (log(self.maxScore) / c.maxScore))
      if bestChild is None or currValue > bestValue:
        bestChild = c
        bestValue = currValue
    return bestChild

  def __repr__(self):
    return (' Unborn Children: ' + str(self.unbornChildren) + ' Action: ' + str(self.lastAction)
            + ' Score: ' + str(self.cumulativeScore) + ' Max Score: ' + str(self.maxScore) 
            + ' Child Count: ' + str(len(self.children)))




"""
Particle filter based on my work for Project 4
"""
class ParticleFilter(object):
  def __init__(self, gameState, index, startPos, numParticles=300):
    "Initializes a list of particles."
    self.legalPositions = gameState.getWalls().asList(False)
    self.numParticles = numParticles
    self.particles = [startPos for _ in range(0, self.numParticles)]
    self.index = index # Index of the tracked agent

  def clusterParticles(self, position):
    "Put all particles in one place, to be used when enemy is visible"
    self.particles = [position for _ in range(0, self.numParticles)]

  def resetParticles(self):
    "Scatter the particles randomly, if our estimates are too far off"
    self.particles = [random.choice(self.legalPositions) for _ in range(0, self.numParticles)]

  def observe(self, gameState, selfPosition, shouldClusterToInit):
    "Update beliefs based on the given distance observation."
    observation = gameState.getAgentDistances()[self.index]
    particleWeights = util.Counter()
    newParticleList = []
    beliefDist = self.getBeliefDistribution()
    cumulativeProb = 0
    # Assign weights to particles depending on how likely it is for that location to be
    # correct given the most recent observation
    for particle in self.particles:
      trueDistance = util.manhattanDistance(particle, selfPosition)
      distanceProb = gameState.getDistanceProb(observation, trueDistance)

      particleWeights[particle] = (distanceProb * beliefDist[particle])
      # If the probablity of all particles is 0, we're either way off or we've knocked out the
      # enemy.  We keep track of this, and either reset or cluster to init if it is 0.
      cumulativeProb += distanceProb

    if cumulativeProb != 0:
      # Resample based on new weights
      for _ in range(self.numParticles):
        newParticleList.append(util.sample(particleWeights))
      self.particles = newParticleList
    else:
      # Reset particles if we're too far off
      if shouldClusterToInit:
        self.clusterParticles(gameState.getInitialAgentPosition(self.index))
      else:
        self.resetParticles()
    
  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.
    """
    newParticleList = []
    # Pretend each particle is a ghost, and set its position semi-randomly based on how
    # likely the ghost is to move to that position
    for particle in self.particles:
      newPosDist = self.getPositionDistribution(gameState, particle)
      newParticleList.append(util.sample(newPosDist))
    self.particles = newParticleList

  def getPositionDistribution(self, gameState, position):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.
    """
    dist = util.Counter()
    conf = game.Configuration(position, game.Directions.STOP)
    newState = gameState.deepCopy()
    newState.data.agentStates[self.index] = game.AgentState(conf, False)

    for action in getLegalActionsNoStop(newState, self.index):
      successorPosition = newState.getSuccessor(position, action)
      if (action is game.Directions.STOP):
        dist[successorPosition] += .1
      else:
        dist[successorPosition] += 1
    return dist

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    # This essentially gives a point to a location for each particle there, then 
    # normalizes the point values so they add up to 1.
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

  def getMostLikelyPos(self):
    """
    Return the ghost position considered most likely by our current model.
    """
    mostLikelyPos = None
    mostLikelyProb = None
    beliefDist = self.getBeliefDistribution()
    for part in self.particles:
      currProb = beliefDist[part]
      if mostLikelyPos is None or currProb > mostLikelyProb:
        mostLikelyPos = part 
        mostLikelyProb = currProb
    return mostLikelyPos