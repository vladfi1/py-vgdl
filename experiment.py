from player import *

import next_frame

from games import games

players = [Player(*game) for game in games]

def colorize(pixels):
  f = np.vectorize(indexColor, otypes=3 * [np.uint8])
  return np.stack(f(pixels), axis=-1)

def writeVideo(path, video):
  from PIL import Image
  
  for i, image in enumerate(video):
    image = np.swapaxes(image, 0, 1)
    img = Image.fromarray(image, 'RGB')
    img.save(path + '%d.png' % i)

def run(iters=1000, steps=50):
  for i in range(iters):
    for _ in range(steps):
      player = random.choice(players)
      frames, actions, events = player.play(50)
      next_frame.train(frames, actions)
    
    next_frame.save('next_frame/')
    
    player = random.choice(players)
    frames, actions, events = player.play(50)
    predictions = next_frame.predict(frames, actions)
    frames = frames[-len(predictions):]

    # concat along y axis
    joined = np.concatenate([frames, predictions], 2)
    video = colorize(joined)
    
    path = 'predictions/%d/' % i
    import os
    if not os.path.exists(path):
      os.makedirs(path)
    writeVideo(path, video)

