import tensorflow as tf
import tf_lib as tfl

from vgdl.ontology import colors

colorspace = len(colors)

sequence_length = 50

# TODO: batched inputs
frames_nxy = tf.placeholder(tf.int64, [sequence_length, None, None])

actions_n = tf.placeholder(tf.int64, [sequence_length])
actions_na = tf.one_hot(actions_n, 6) # num_actions
actions_bna = tf.expand_dims(actions_na, 0)

image_dims = tf.shape(frames_nxy)[1:]
image_width = image_dims[0]
image_height = image_dims[1]

frames_nxyc = tf.one_hot(frames_nxy, colorspace)

# local info
# TODO: make this hierarchical somehow, using "deconvolutions"?
conv1 = tfl.convLayer(frames_nxyc, filter_size=5, filter_depth=32, pool_size=1)

# global info
summary = tf.reduce_max(conv1, [1, 2])
summary = tf.expand_dims(summary, 0) # batch size 1 for now

summary = tf.concat(2, [summary, actions_bna])

gru = tf.nn.rnn_cell.GRUCell(64)
outputs, _ = tf.nn.dynamic_rnn(gru, summary, dtype=tf.float32)
#outputs, _ = tf.nn.rnn(gru, tf.unpack(summary), dtype=tf.float32)

outputs = tf.squeeze(outputs, squeeze_dims=[0])

# tile across width and height
outputs = tf.expand_dims(outputs, 1)
outputs = tf.expand_dims(outputs, 1)
# need a recent tf for shape inference to get this
outputs = tf.tile(outputs, tf.concat(0, [[1], image_dims, [1]]))

# lots of intermediate reshaping here
# which could be done only once, at the beginning
features = tf.concat(3, [frames_nxyc, conv1, outputs])
fc1 = tfl.affineLayer(features, 64, tf.nn.relu)
logits = tfl.affineLayer(fc1, colorspace, bias=False)

burn_in = 10 # allow the rnn to learn for a bit

targets = frames_nxyc[1+burn_in:]
logits = logits[burn_in:sequence_length-1]

flat_targets = tf.reshape(targets, [-1, colorspace])
flat_logits = tf.reshape(logits, [-1, colorspace])

loss = tf.nn.softmax_cross_entropy_with_logits(flat_logits, flat_targets)
loss = tf.reduce_mean(loss)

update_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

mle = tf.argmax(logits, 3)
#predictions = tfl.softmax(logits)
#flat_predictions = tf.nn.softmax(flat_logits)
#predictions = tf.reshape(flat_predictions, tf.concat(0, [[sequence_length-burn_in-1], image_dims, [colorspace]]))

saver = tf.train.Saver(tf.all_variables())

sess = tf.Session()

def train(frames, actions):
  _, l = sess.run([update_op, loss], feed_dict = {frames_nxy:frames, actions_n:actions})
  print(l)

def predict(frames, actions):
  "Maximum-Likelihood estimates."
  return sess.run(mle, feed_dict = {frames_nxy:frames, actions_n:actions})

