from vgdl.core import VGDLParser
from pygame.locals import *
# from examples.gridphysics.sokoban import so
from pybrain.rl.agents.agent import Agent
from pygame.surfarray import array3d as grab_pixels
from numpy import random
import numpy as np
from vgdl.ontology import colors, colorIndices
import pygame

keys = [None, K_SPACE, K_UP, K_DOWN, K_RIGHT, K_LEFT]
key2index = {k: i for i, k in enumerate(keys)}

color_items = colors.items()
color2index = {color: index for index, color in enumerate(colors.values())}

def colorIndex(color):
  color = tuple(color)
  if color not in color2index:
    color = "UNKNOWN"
  return color2index[color]

def indexColor(index):
  _, color = color_items[index]
  if isinstance(color, str):
    color = (0, 0, 0) # UNKNOWN or OFFSCREEN
  return color

# TODO: complete
effects = ['killSprite', 'collectResource', 'changeResource']

class DummyAgent(Agent):
  def getAction(self):
    return random.choice(keys)

class Player(object):
  def __init__(self, game_str, level_str, agent=None):
    self.game_str = game_str
    self.level_str = level_str
    
    self.agent = DummyAgent() if agent is None else agent
    
    self.reset()
  
  def reset(self):
    self.game = VGDLParser().parseGame(self.game_str)
    self.game.buildLevel(self.level_str, block_size=2)
    self.resetScreen()
  
  def resetScreen(self):
    # workaround pygame only having one global screen
    self.game._initScreen(self.game.screensize, headless=True)
    self.game._drawAll()
  
  def play(self, steps=None, movie=False):
    frames = []
    actions = []
    rewards = []
    events = []
    
    i = 0

    self.resetScreen()

    while True:
      pixels = grab_pixels(self.game.screen)
      frames.append(pixels)
      self.agent.integrateObservation(pixels)
      
      action = self.agent.getAction()
      actions.append(key2index[action])
      
      win, score, events_ = self.game.tick(action, headless=False)
      
      #agent.giveReward(score)
      #rewards.append(score)
      
      events_ = map(lambda (e, c1, c2): (e, colorIndices[c1], colorIndices[c2]), events_)
      events.append(events_)
      
      if movie:
        pygame.image.save(self.game.screen, 'movie/%d.png' % i)
      
      if win is not None:
        #self.game.reset()
        self.reset()
        
        if steps is None:
          break
      
      i += 1
      if i == steps:
        break
    
    for frame in frames:
      #assert(frame.dtype == np.uint8)
      assert(frame.shape == frames[0].shape)
    
    frames = np.array(frames)
    frames = np.apply_along_axis(colorIndex, 3, frames)
    
    return frames, actions, events

