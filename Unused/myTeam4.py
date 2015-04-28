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

    # Store map dimensions
    self.mapWidth = gameState.data.layout.width
    self.mapHeight = gameState.data.layout.height


    # Check how recently we were near the enemy to check if we've knocked him out
    self.nearEnemyCounter = 0

    # Set up particle filters to track enemy locations
    self.enemyLocFilters = {}
    for i in self.enemyIndices:
      self.enemyLocFilters[i] = (ParticleFilter(gameState, i,
                              gameState.getInitialAgentPosition(i)))

    # Decide which Pac-man takes the bottom half
    self.isBottom = (self.index == min(self.teamIndices))

    # Divvy up food between the two Pac-men
    self.foodLists = self.distributeFood(False, False, gameState)
    self.foodList = self.foodLists['Bottom'] if self.isBottom else self.foodLists['Top']


    self.modes = ['Attack', 'Defend', 'Chase', 'Scatter']
    self.seenEnemies = []

    # Set up strategy for Minimax simulations
    self.strategies = {}
    for i in range(gameState.getNumAgents()):
      if i in self.teamIndices:
        self.strategies[i] = 'Attack'
      else:
        self.strategies[i] = 'Attack'

    self.startTime = None
    self.timeLimit = 0.9
    self.currentTarget = None
    self.closestEnemy = None
    #self.goOffensive = False

  def chooseAction(self, gameState):
    self.startTime = time.clock()
    self.updateParticleFilters(gameState)

    # Check for a death, setting defensive ghosts to attack mode if so
    #for i in self.enemyIndices:
    #  if (self.enemyLocFilters[i].getMostLikelyPos() == gameState.getInitialAgentPosition(i)
    #        and gameState.data.timeleft < 1147):
    #    print('EEOEE')

    # Check if the enemy is observable, changing strategy if necessary
    self.seenEnemies = []
    for i in self.enemyIndices:
      exactPosition = gameState.getAgentPosition(i)
      if exactPosition is not None:
        self.seenEnemies.append(i)


    if not self.seenEnemies:
      self.strategies[self.index] = "Attack"
    else:
      newStrat = None
      closestEnemyDist = None
      for e in self.seenEnemies:
        myPos = gameState.getAgentPosition(self.index)
        enemyPos = gameState.getAgentPosition(e)
        dist = self.distancer.getDistance(myPos, enemyPos)
        if not closestEnemyDist or dist < closestEnemyDist:
          self.closestEnemy = e
      newStrat = self.evalStratChange(gameState, e)
      if newStrat is not None:
        self.strategies[self.index] = newStrat
    
    # Remove old food
    foodLocs = self.getFood(gameState).asList()
    for _, l in self.foodLists.iteritems():
      missingFood = [l1 for l1 in l if l1 not in foodLocs]
      for food in missingFood:
        l.remove(food)

    ''' debugging    
    dist = util.Counter()
    l = 'Bottom' if self.isBottom else 'Top'
    for food in self.foodLists[l]: 
      dist[food] += 1
    dist.normalize()
    self.displayDistributionsOverPositions([dist])
    '''
    # Get current target if chasing
    if self.strategies[self.index] == 'Chase' and self.closestEnemy is not None:
      self.currentTarget = self.getTarget(gameState, self.closestEnemy)
      ''' debugging target
      dist = util.Counter()
      dist[self.currentTarget] += 1
      dist.normalize()
      self.displayDistributionsOverPositions([dist])
      '''
    '''
      minDistance = None
      closestEnemy = None
      for e in self.seenEnemies:
        distance = self.distancer.getDistance(gameState.getAgentPosition(self.index), 
                    gameState.getAgentPosition(e))
        if not closestEnemy or minDistance > distance:
          minDistance = distance
          closestEnemy = e
      if closestEnemy is not None:
        self.currentTarget = self.getTarget(gameState, closestEnemy)
    '''
    # Get list of actions, and consider more actions if it's urgent
    actions = self.getGoodLegalActions(gameState, self.index)

    # If there's only one action, just take it
    if len(actions) is 1:
      return actions[0]

    # Create simulated game state based on estimated enemy locations
    simState = gameState.deepCopy()
    for i in self.enemyIndices:
      if gameState.getAgentPosition(i) is None:
        mostLikelyPos = self.enemyLocFilters[i].getMostLikelyPos()
        conf = game.Configuration(mostLikelyPos, game.Directions.STOP)
        simState.data.agentStates[i] = game.AgentState(conf, False)


    bestAction = random.choice(actions)
    currBestAction = self.getBestAction(simState, 2, actions)
    bestAction = currBestAction
    return bestAction

  def evalStratChange(self, gameState, enemyIndex):
    selfState = gameState.getAgentState(self.index)
    enemyState = gameState.getAgentState(enemyIndex)
    distance = self.distancer.getDistance(gameState.getAgentPosition(self.index), 
                        gameState.getAgentPosition(enemyIndex))
    if ((enemyState.isPacman and selfState.scaredTimer < distance / 2) 
        or (distance / 2 < enemyState.scaredTimer)) \
        and distance < 8:
      return 'Chase'
    elif ((selfState.scaredTimer > distance) or selfState.isPacman) \
              and distance < 6 and len(self.getFood(gameState).asList()) > 4:
      return 'Scatter'
    return None

  def distributeFood(self, shouldRedistBottom, shouldRedistTop, gameState):
    # Distribute food in top and bottom rows if just starting off
    if not shouldRedistBottom and not shouldRedistTop:
      foodLists = self.createFoodLists(gameState)

      while len(foodLists['Neither']) > 0:
        # Assign unassigned food based on proximity and amount assigned already
        bottomFoodNum = len(foodLists['Bottom'])
        topFoodNum = len(foodLists['Top'])

        if bottomFoodNum < topFoodNum:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Bottom'])
          foodLists['Bottom'].append(newFood)
        else:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Top'])
          foodLists['Top'].append(newFood)
        foodLists['Neither'].remove(newFood)

      foodLists['Bottom'], foodLists['Top'] = self.fixOutlierPellets(foodLists['Bottom'], 
                                                                      foodLists['Top'])
    elif shouldRedistBottom:
      foodLists = dict(self.foodLists)
      foodLists['Neither'] = list(foodLists['Bottom'])
      foodLists['Bottom'] = []
      while len(foodLists['Neither']) > 0:
        bottomFoodNum = len(foodLists['Bottom'])
        if bottomFoodNum > len(foodLists['Neither']) / 2:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Bottom'])
          foodLists['Bottom'].append(newFood)
        else:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Top'])
          foodLists['Top'].append(newFood)
        foodLists['Neither'].remove(newFood)

      #foodLists['Bottom'], foodLists['Top'] = self.fixOutlierPellets(foodLists['Bottom'], 
      #                                                                foodLists['Top'])
    elif shouldRedistTop:
      foodLists = dict(self.foodLists)
      foodLists['Neither'] = list(foodLists['Top'])
      foodLists['Top'] = []
      #TODO: foodLists['Top'].append(self.getClosestFood(foodList))
      while len(foodLists['Neither']) > 0:
        topFoodNum = len(foodLists['Top'])
        if topFoodNum > len(foodLists['Neither']) / 2:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Top'])
          foodLists['Top'].append(newFood)
        else:
          newFood, _ = self.getClosestFoodFromLists(foodLists['Neither'], foodLists['Bottom'])
          foodLists['Bottom'].append(newFood)
        foodLists['Neither'].remove(newFood)

      #foodLists['Bottom'], foodLists['Top'] = self.fixOutlierPellets(foodLists['Bottom'], 
      #                                                                foodLists['Top'])
    return foodLists

  def fixOutlierPellets(self, topList, bottomList):
    # Do another pass on bordering pellets to reduce outliers
    bottomFoodNum = len(bottomList)
    topFoodNum = len(topList)
    if bottomFoodNum < topFoodNum:
      newFood, minDist = self.getClosestFoodFromLists(topList, bottomList)
      while (minDist < 2):
        bottomList.append(newFood)
        topList.remove(newFood)
        newFood, minDist = self.getClosestFoodFromLists(topList, bottomList)
    else:
      newFood, minDist = self.getClosestFoodFromLists(bottomList, topList)
      while (minDist < 2):
        topList.append(newFood)
        bottomList.remove(newFood)
        newFood, minDist = self.getClosestFoodFromLists(bottomList, topList)
    return topList, bottomList


  def createFoodLists(self, gameState):
    foodLists = {'Bottom':[], 'Top':[], 'Neither':[]}
    foodLocs = self.getFood(gameState).asList()
    for x, y in foodLocs:
      topLimit = self.mapHeight - 7
      if y < 6:
        foodLists['Bottom'].append((x,y))
      elif y > topLimit:
        foodLists['Top'].append((x,y))
      else:
        foodLists['Neither'].append((x,y))
    return foodLists

  def getClosestFoodFromLists(self, sourceList, destList):
    closestFoodDist = None
    closestFoodPos = None
    for food in destList:
      currFoodPos, currFoodDist = self.getClosestFood(sourceList, food)
      if closestFoodPos is None or currFoodDist < closestFoodDist:
        closestFoodPos = currFoodPos
        closestFoodDist = currFoodDist
    return (closestFoodPos, closestFoodDist)

  def getClosestFood(self, foodList, pos):
    closestFoodDist = None
    closestFoodPos = None
    for food in foodList:
      currFoodDist = self.distancer.getDistance(food, pos)
      if closestFoodPos is None or currFoodDist < closestFoodDist:
        closestFoodPos = food
        closestFoodDist = currFoodDist
    return (closestFoodPos, closestFoodDist)


  def getTarget(self, gameState, enemyIndex):
    myPos = gameState.getAgentPosition(self.index)
    enemyPos = gameState.getAgentPosition(enemyIndex)
    enemyDirection = gameState.getAgentState(enemyIndex).configuration.direction
    target = enemyPos

    if self.distancer.getDistance(myPos, enemyPos) > 2:
      if enemyDirection == Directions.NORTH:
        target = (enemyPos[0], enemyPos[1] + 1)
      elif enemyDirection == Directions.SOUTH:
        target = (enemyPos[0], enemyPos[1] - 1)
      elif enemyDirection == Directions.EAST:
        target = (enemyPos[0] + 1, enemyPos[1])
      elif enemyDirection == Directions.WEST:
        target = (enemyPos[0] - 1, enemyPos[1])
    else:
      if enemyPos[0] == myPos[0] and enemyPos[1] == myPos[1] + 1:
        target = (myPos[0], myPos[1] + 2)
      elif enemyPos[0] == myPos[0] and enemyPos[1] == myPos[1] + 2:
        target = (myPos[0], myPos[1] + 3)
      elif enemyPos[0] == myPos[0] and enemyPos[1] == myPos[1] - 1:
        target = (myPos[0], myPos[1] - 2)
      elif enemyPos[0] == myPos[0] and enemyPos[1] == myPos[1] - 2:
        target = (myPos[0], myPos[1] - 3)
      elif enemyPos[0] == myPos[0] - 1 and enemyPos[1] == myPos[1]:
        target = (myPos[0] - 2, myPos[1])
      elif enemyPos[0] == myPos[0] - 2 and enemyPos[1] == myPos[1]:
        target = (myPos[0] - 3, myPos[1])
      elif enemyPos[0] == myPos[0] + 1 and enemyPos[1] == myPos[1]:
        target = (myPos[0] + 2, myPos[1])
      elif enemyPos[0] == myPos[0] + 3 and enemyPos[1] == myPos[1]:
        target = (myPos[0] + 3, myPos[1])
    return target

  def getBestAction(self, gameState, depth, possibleActions):
    """
      Returns the minimax action using depth and self.evaluationFunction
    """

    # Run AlphaBeta for each initial action possibility to specified depth
    bestActions = []
    bestScore = None
    for action in possibleActions:
      newState = gameState.generateSuccessor(self.index, action)
      newScore = self.runAlphaBeta(newState, getNextIndex(gameState, self.index),
                    depth, float("-inf"), float("inf"))
      # If out of time, abort
      if newScore is None:
        return random.choice(possibleActions)
      #if self.strategies[self.index] != "Attack":
      #  print(self.strategies[self.index], newScore, action)

      if bestScore is None or newScore > bestScore:
        bestScore = newScore
        bestActions = [action]
      elif newScore == bestScore:
        bestActions.append(action)
    return random.choice(bestActions)

  # Returns score of going down a given path based on eval function
  def runAlphaBeta(self, gameState, currAgentNum, depthRemaining, alpha, beta):
    # Abort if we're running out of time
    currTime = time.clock()
    if (currTime - self.startTime > self.timeLimit):
      print('fail0')
      return None

    # If the current state is terminal according to current strategy, stop evaluation
    terminalScore = self.getScoreIfTerminal(gameState)
    if terminalScore:
      return terminalScore

    # Return if at leaf node
    if depthRemaining is 0:
      return self.evaluationFunction(gameState)

    nextAgentNum = getNextIndex(gameState, currAgentNum)  # Index of agent to eval next
    nextDepthRemaining = depthRemaining        # Remaining depth at next eval
    # If done with all agents, decrease depth and go to our agent
    if nextAgentNum == self.index:
      nextAgentNum = 0
      nextDepthRemaining -= 1

    # Get list of actions, and consider more actions if it's urgent
    actions = self.getGoodLegalActions(gameState, currAgentNum)

    bestScore = None
    newAlpha = alpha
    newBeta = beta
    if currAgentNum in self.teamIndices:   # Evaluate max
      for action in actions:
        successor = gameState.generateSuccessor(currAgentNum, action)
        newScore = self.runAlphaBeta(successor, nextAgentNum,
                                     nextDepthRemaining, newAlpha, newBeta)
        # If out of time, abort
        if newScore is None:
          return None
        # If new score is the best, set best
        if bestScore is None or newScore > bestScore:
          bestScore = newScore
        # If new score is more than alpha, change alpha
        if bestScore > newAlpha:
          newAlpha = bestScore
        # Stop searching nodes if not viable
        if newBeta <= newAlpha:
          break
    else:   # Evaluate min
      newBeta = beta
      for action in actions:
        successor = gameState.generateSuccessor(currAgentNum, action)
        newScore = self.runAlphaBeta(successor, nextAgentNum, 
                                    nextDepthRemaining, newAlpha, newBeta)
        # If out of time, abort
        if newScore is None:
          return None
        # If new score is the best, set best
        if bestScore is None or newScore < bestScore:
          bestScore = newScore
        # If new score is less than beta, change beta
        if bestScore < newBeta:
          newBeta = bestScore
        # Stop searching nodes if not viable
        if newBeta <= newAlpha and nextAgentNum is 0:
          break
    return bestScore

  def evaluationFunction(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    #if self.strategies[self.index] != "Attack":
    #  print(features, [gameState.getAgentPosition(i) for i in self.seenEnemies])
    return features * weights

  def getFeatures(self, gameState):
    features = util.Counter()
    myPos = gameState.getAgentState(self.index).getPosition()
    if self.strategies[self.index] == 'Attack':
      features['score'] = self.getScore(gameState)

      # Compute distance to the nearest food
      foodList = self.getFood(gameState).asList()
      closestFood = self.getClosestFood(self.foodList, myPos)[1]
      if closestFood:# and len(self.foodList) > 3:
        features['foodDist'] = self.getClosestFood(self.foodList, myPos)[1]
      else:
        features['foodDist'] = min([self.getMazeDistance(myPos, food) for food in foodList])

    elif self.strategies[self.index] == 'Chase':
      if 0 < self.currentTarget[0] < self.mapWidth or \
          0 < self.currentTarget[1] < self.mapHeight or \
          gameState.hasWall(self.currentTarget[0], self.currentTarget[1]):
        features['targetDist'] = util.manhattanDistance(myPos, self.currentTarget)
      else:
        features['targetDist'] = self.getMazeDistance(myPos, self.currentTarget)
      selfState = gameState.getAgentState(self.index)

    elif self.strategies[self.index] == 'Scatter':
      # Get distance to hunter
      #knownEnemyLocations = [gameState.getAgentPosition(i) for i in self.seenEnemies]
      features['hunterDist'] = self.getMazeDistance(myPos, 
                                  gameState.getAgentPosition(self.closestEnemy))

      # Get closest power pellet
      pelletLocations = self.getCapsules(gameState)
      if pelletLocations:
        features['powerPelletDist'] = min([self.getMazeDistance(myPos, l) for l in pelletLocations])
      else:
        features['powerPelletDist'] = 0
      # Get distance from ally
      for i in self.teamIndices:
        if i != self.index:
          allyPos = gameState.getAgentState(i).getPosition()
          allyDist = self.getMazeDistance(myPos, allyPos)
          features['allyDist'] = allyDist
    return features

  def getWeights(self, gameState):
    if self.strategies[self.index] == 'Attack':
      weights = {'score': 100, 'foodDist': -1}
    elif self.strategies[self.index] == 'Chase':
      weights = {'targetDist': -1}
    elif self.strategies[self.index] == 'Scatter':
      weights = {'hunterDist': 1000, 'powerPelletDist': -10, 'allyDist': 1}
    return weights

  def getScoreIfTerminal(self, gameState):
    """
    Returns infinity if the current state is terminal for our Pac-man -- i.e. a natural endpoint
    for the current strategy, either good or bad
    """
    if self.strategies[self.index] == 'Scatter':
      # Was killed in action

      myPos = gameState.getAgentState(self.index).getPosition()
      myStart = gameState.getInitialAgentPosition(self.index)
      if myPos == myStart:
        return float("-inf")
      # Is no longer prey
      selfState = gameState.getAgentState(self.index)
      enemyState = gameState.getAgentState(self.closestEnemy)
      if enemyState.scaredTimer > 0 or (enemyState.isPacman and not selfState.scaredTimer > 0):
        return float("inf")
    return None

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
        f.observe(gameState, gameState.getAgentPosition(self.index), 
                            self.nearEnemyCounter > 0)
        if (self.nearEnemyCounter > 0): self.nearEnemyCounter -= 1
    
    """ For debugging purposes
    self.displayDistributionsOverPositions(
        [f.getBeliefDistribution() for i, f in self.enemyLocFilters.iteritems()])
    """

  def getGoodLegalActions(self, gameState, index):
    """
    Same as 'getLegalActions', sans Stop (and Reverse if canReverse is False)
    """
    legalActions = gameState.getLegalActions(index)
    #closestFood = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food)
    #                    for food in self.getFood(gameState).asList()])
  
    if self.strategies[index] == 'Chase' or self.strategies[index] == 'Scatter':# or \
      #(gameState.getAgentState(self.index).isPacman and closestFood > 10):
      goodActions = [a for a in legalActions if a != Directions.STOP]
    else:
      rev = Directions.REVERSE[gameState.getAgentState(index).configuration.direction]
      goodActions = [a for a in legalActions if a != Directions.STOP and a != rev]
      if not goodActions:
        goodActions = [rev]
    return goodActions



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


#####################
# Utility Functions #
#####################
def getNextIndex(gameState, currIndex):
  """
  Utility function to get the index of the next agent whose turn it is
  """
  nextIndex = currIndex + 1
  if (nextIndex >= gameState.getNumAgents()):
    nextIndex = 0
  return nextIndex
