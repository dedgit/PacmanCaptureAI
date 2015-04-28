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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SecretAgent', second = 'SecretAgent'):
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
class SecretAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

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


    # Dict of qValues with (state, action) tuples as keys
    self.qValues = util.Counter()

    self.modes = ['Offense', 'Defense']
    self.currMode = 'Offense'
    self.discount = 0.5
    self.learningRate = 0.5

    self.initializeWeights()

    test += 1
    print(test)

  def chooseAction(self, gameState):
    if (gameState.isOver()):
      print("TOE")
    self.updateParticleFilters(gameState)
    actions = getGoodLegalActions(gameState, self.index)
    return random.choice(actions)

  def updateParticleFilters(self, gameState):
    # Populate particle filters with most recent data
    actions = gameState.getLegalActions(self.index)
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
    
    # """ For debugging purposes
    self.displayDistributionsOverPositions(
        [f.getBeliefDistribution() for i, f in self.enemyLocFilters.iteritems()])
    # """

  def getFeatureValues(self, gameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    myPos = gameState.getAgentState(self.index).getPosition()

    if self.mode == 'Offense':
      minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minFoodDistance

      enemies = [gameState.getAgentState(i) for i in self.enemyIndices]
      minEnemyDistance = min([self.getMazeDistance(myPos, enemy) for enemy in enemies])
      features['distanceToEnemy'] = minEnemyDistance
    return features

  def getReward(self, gameState):
    pass

  def initializeWeights(self):
    self.weights = util.Counter()
    self.weights['Offense'] = {'distanceToFood': -1, 'distanceToEnemy': 1}

  def estimateQValues(self, action):
    features = features = self.getFeatures(gameState)
    weights = self.getWeights(self.agentMode)
    return features * weights

  def updateWeights(self, gameState, expectedQValue, realQValue):
    """
    Updates weight estimates using Q-learning.  Formula:
      weight = weight + discount * (newQ - oldQ) * featureValues
    """
    featureValues = getFeatureValues(gameState)
    for name in self.weights:
      self.weights[name] += self.learningRate * (realQValue - expectedQValue) * featureValues[name]

  def getWeights(self, agentMode):
    return self.weights[agentMode]


###################
# Particle Filter #
###################
class ParticleFilter(object):
  """
  Based on my work for Project 4
  """
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

    for action in newState.getLegalActions(self.index):
      successorPosition = game.Actions.getSuccessor(position, action)
      if (action is game.Directions.STOP or action is game):
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


####################
# Utility Funtions #
####################

def getGoodLegalActions(gameState, index):
  """
  Same as 'getLegalActions', sans Stop and Reverse
  """
  legalActions = gameState.getLegalActions(index)
  rev = Directions.REVERSE[gameState.getAgentState(index).configuration.direction]
  goodActions = [a for a in legalActions if a != Directions.STOP and a != rev]
  if not goodActions:
    goodActions = [rev]
  return goodActions