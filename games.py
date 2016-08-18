from importlib import import_module

prefix = 'examples.gridphysics.'

def importGame(spec):
  module = None
  name = None
  
  if len(spec) == 1:
    module = spec[0]
    name = spec[0]
  elif len(spec) == 2:
    module, name = spec
  else:
    raise ValueError("Invalid spec " + str(spec))
  
  game_level = [name + '_game', name + '_level']
  
  # what is the point of fromlist here?
  # m = __import__(prefix + module, fromlist=game_level)
  m = import_module(prefix + module)
  
  return [getattr(m, k) for k in game_level]

gridphysics = [
  #('aliens',),
  #('dodge', 'bullet'),
  #('frogs', 'frog'),
  #('frogs_video', 'frog'),
  #('chase',),
  #('boulderdash',),
  #('butterflies', 'chase'), buggy :(
  #('missilecommand',),
  #('mrpacman', 'pacman'),
  ('portals', 'portal'),
  #('sokoban', 'push', 'box'),
  #('survivezombies', 'zombie'),
  ('zelda',),
]

games = map(importGame, gridphysics)


