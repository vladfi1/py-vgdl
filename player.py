from vgdl.core import VGDLParser
from pygame.locals import *
# from examples.gridphysics.sokoban import so
from pybrain.rl.agents.agent import Agent
from pygame.surfarray import array3d as grab_pixels
from numpy import random
import numpy as np
from vgdl.ontology import colors, colorIndices

keys = [K_SPACE, K_UP, K_DOWN, K_RIGHT, K_LEFT]
key2index = {k: i for i, k in enumerate(keys)}

color2index = {color: colorIndices[name] for name, color in colors.items()}

# TODO: complete
effects = ['killSprite', 'collectResource', 'changeResource']

class DummyAgent(Agent):
    def getAction(self):
        return random.choice(keys)

# this might be implemented in pybrain.rl?
def play(game_str, level_str, agent=None, steps=None):
    g = VGDLParser().parseGame(game_str)
    g.buildLevel(level_str, block_size=2)
    g._initScreen(g.screensize,headless=True)
    
    frames = []
    actions = []
    rewards = []
    events = []
    
    i = 0
    
    if agent is None:
        agent = DummyAgent()
    
    while True:
        pixels = grab_pixels(g.screen)
        frames.append(pixels)
        agent.integrateObservation(pixels)
        
        action = agent.getAction()
        actions.append(key2index[action])
        
        win, score, events_ = g.tick(action, headless=False)
        
        #agent.giveReward(score)
        #rewards.append(score)
        
        events_ = map(lambda (e, c1, c2): (e, colorIndices[c1], colorIndices[c2]), events_)
        events.append(events_)
        
        #pygame.image.save(g.screen, 'movie/%d.png' % i)
        
        if win is not None:
            break
        
        i += 1
        if i == steps:
            break
    
    frames = np.array(frames)
    frames = np.apply_along_axis(lambda color: color2index[tuple(color)], 3, frames)
    
    return frames, actions, events

from examples.gridphysics.aliens import aliens_level, aliens_game
frames, actions, events = play(aliens_game, aliens_level, steps=50)


