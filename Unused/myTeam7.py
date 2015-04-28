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
               first = 'DivideAndConquerAgent', second = 'DivideAndConquerAgent'):
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

class DivideAndConquerAgent(CaptureAgent):
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

    # Decide which Pac-man takes the bottom half
    self.isBottom = (self.index == min(self.teamIndices))

    # Divvy up food between the two Pac-men
    self.foodLists = self.distributeFood(False, False, gameState)
    self.foodList = self.foodLists['Bottom'] if self.isBottom else self.foodLists['Top']

    # Check how recently we were near the enemy to check if we've knocked him out
    self.nearEnemyCounter = 0

    # Set up particle filters to track enemy locations
    self.enemyLocFilters = {}
    for i in self.enemyIndices:
      self.enemyLocFilters[i] = (ParticleFilter(gameState, i,
                              gameState.getInitialAgentPosition(i)))

  def chooseAction(self, gameState):
    startTime = time.clock()

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

    goodActions = getGoodLegalActions(gameState, self.index)
    # If there's only one action, just take it
    if len(goodActions) is 1:
      return goodActions[0]

    bestDestination = self.runTopSecretAlgorithm(startTime, gameState)
    print(bestDestination)
    bestActions = []
    bestDist = None
    for action in goodActions:
      successor = gameState.generateSuccessor(self.index, action)
      currentDist = self.distancer.getDistance(bestDestination, 
                                  successor.getAgentPosition(self.index))
      if bestDist is None or currentDist < bestDist:
        bestDist = currentDist
        bestActions = [action]
      elif currentDist == bestDist:
        bestActions.append(action)
      print(currentDist, successor.getAgentPosition(self.index))
    return random.choice(bestActions)

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

      # Do another pass on bordering pellets to reduce outliers
      if bottomFoodNum < topFoodNum:
        newFood, minDist = self.getClosestFoodFromLists(foodLists['Top'], foodLists['Bottom'])
        while (minDist < 2):
          foodLists['Bottom'].append(newFood)
          foodLists['Top'].remove(newFood)
          newFood, minDist = self.getClosestFoodFromLists(foodLists['Top'], foodLists['Bottom'])
      else:
        newFood, minDist = self.getClosestFoodFromLists(foodLists['Bottom'], foodLists['Top'])
        while (minDist < 2):
          foodLists['Top'].append(newFood)
          foodLists['Bottom'].remove(newFood)
          newFood, minDist = self.getClosestFoodFromLists(foodLists['Bottom'], foodLists['Top'])

    elif shouldRedistBottom:
      foodLists = self.foodLists
      foodLists['Neither'] = foodLists['Bottom'].copy()
      foodLists['Bottom'] = []
    elif shouldRedistTop:
      foodLists = self.foodLists
      foodLists['Neither'] = foodLists['Top'].copy()
      foodLists['Top'] = []
    return foodLists

  def createFoodLists(self, gameState):
    foodLists = {'Bottom':[], 'Top':[], 'Neither':[]}
    foodLocs = self.getFood(gameState).asList()
    for x, y in foodLocs:
      if y < 6:
        foodLists['Bottom'].append((x,y))
      elif y > 11:
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

    

  """
  The top secret algorithm is a heavy-playout Monte Carlo Search Tree using UCT.  Don't tell anyone.

  Structurally, this was informed somewhat by the mcts.ai implementation:
    http://mcts.ai/code/index.html
  However, their implementation is fairly general, and I was careful not to study it too closely.
  """

  def runTopSecretAlgorithm(self, startTime, gameState):
    unexploredFood = self.foodList
    longestPossiblePath = len(gameState.getWalls().asList(False)) * len(self.foodList)
    # Create root node (at player's position)
    currNode = Node(None, gameState.getAgentPosition(self.index), unexploredFood)
    pathDistance = 0

    currTime = time.clock()
    while (currTime - startTime < 2.6):
      """
      Selection
      """
      # Select child nodes based on UCT while all children are explored
      while currNode.children and not currNode.unbornChildren:
        currNode = currNode.chooseChild()
        pathDistance += currNode.distance

      """
      Expansion
      """
      # If the node has an unexplored child, create it and go to that child
      if currNode.unbornChildren:
        currNode = currNode.birthChild()
        pathDistance += currNode.distance

      """
      Playout
      """
      foodRemaining = list(currNode.remainingFood)
      newPos = currNode.foodPos
      while foodRemaining:
        newPos, dist = self.executeNextMove(foodRemaining, newPos)
        foodRemaining.remove(newPos)
        pathDistance += dist

      finalDistance = pathDistance
      finalScore = (longestPossiblePath - pathDistance) / longestPossiblePath

      """
      Backpropagation
      """
      while currNode.parent:
        currNode.updateNode(finalScore)
        currNode = currNode.parent
      currNode.updateNode(finalScore)

      # Keep track of time
      currTime = time.clock()
    bestDestination = self.getBestDestination(currNode)
    endTime = currTime - startTime

    return bestDestination

  def getBestDestination(self, currNode):
    """
    Returns the location of the pellet that is part of the shortest discovered path
    """
    bestScore = None
    bestDestination = None
    for child in currNode.children:
      childScore = child.cumulativeScore
      if (bestDestination is None or childScore > bestScore):
        bestScore = childScore
        bestDestination = child.foodPos
    return bestDestination

  def executeNextMove(self, foodRemaining, currPos):
    """
    Choose a location for the next pellet semi-randomly based on the evaluateState() heuristic and
    returns the resulting state
    """
    chosenFood = random.choice(foodRemaining)
    dist = self.distancer.getDistance(currPos, chosenFood)

    return (chosenFood, dist)

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

#############
# MCST Node #
#############
class Node(object):
  def __init__(self, parent, foodPos, remainingFood):
    self.parent = parent      # The node leading up to the current one. "None" if root
    self.children = []        # Explored nodes branching from the current one. Empty if leaf
    self.unbornChildren = []  # Actions not yet explored that would create successor states

    self.cumulativeScore = 0  # Cumulative score earned from all games played from the node
    self.maxScore = 0         # Maximum possible score from playing that many games

    self.foodPos = foodPos    # Location of food that the node represents, or player loc if root
    self.distance = 0         # Distance between this node and its parent
    self.remainingFood = remainingFood

    # Create unborn list as soon as you create the node
    self.populateUnborn()

  def populateUnborn(self):
    """
    Populates self.unbornChildren with a list of pellets that could one day be children
    """
    for foodPos in self.remainingFood:
      self.unbornChildren.append(foodPos)
    random.shuffle(self.unbornChildren)

  def birthChild(self):
    """
    Pop an action of the unborn list and make the resulting state a new child
    """
    newFoodLoc = self.unbornChildren.pop()
    newRemainingFood = list(self.remainingFood)
    newRemainingFood.remove(newFoodLoc)

    baby = Node(self, newFoodLoc, newRemainingFood)
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
    return (' Unborn Children: ' + str(self.unbornChildren)
            + ' Score: ' + str(self.cumulativeScore) + ' Max Score: ' + str(self.maxScore) 
            + ' Child Count: ' + str(len(self.children)))


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