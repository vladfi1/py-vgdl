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

color2index = {color: colorIndices[name] for name, color in colors.items()}

# TODO: complete
effects = ['killSprite', 'collectResource', 'changeResource']

class DummyAgent(Agent):
    def getAction(self):
        return random.choice(keys)

class Player(object):
  def __init__(self, game_str, level_str, agent=None):
    self.game_str = game_str
    self.level_str = level_str
    
    self.reset()
    
    self.agent = DummyAgent() if agent is None else agent
  
  def reset(self):
    self.game = VGDLParser().parseGame(self.game_str)
    self.game.buildLevel(self.level_str, block_size=2)
    self.game._initScreen(self.game.screensize, headless=True)
  
  def play(self, steps=None, movie=False):
    frames = []
    actions = []
    rewards = []
    events = []
    
    i = 0
    
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
    
    frames = np.array(frames)
    frames = np.apply_along_axis(lambda color: color2index[tuple(color)], 3, frames)
    
    return frames, actions, events

from examples.gridphysics.aliens import aliens_level, aliens_game

import next_frame

player = Player(aliens_game, aliens_level)

for _ in xrange(1000):
  frames, actions, events = player.play(50)
  next_frame.train(frames, actions)

