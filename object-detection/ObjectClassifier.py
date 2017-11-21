import tensorflow as tf
import sys
import os
import shutil
from os import listdir
from os import mkdir
from shutil import copyfile
from os.path import isfile, join


class ObjectClassifier():

	"""
	Static references - so that we don't have to initialize them every time
	"""
	label_lines = ['tablet', 'mobile', 'car', 'watch', 'tv', 'bike', 'treadmill']

	with tf.gfile.FastGFile("object_model/graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	def __init__(self, image):
		self.image = image

	def getTags(self):
		with tf.Session() as sess:
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
			image_data =  tf.gfile.FastGFile(self.image, 'rb').read()       

			print ('Processing ... '+self.image)
			predictions = sess.run(softmax_tensor, \
								{'DecodeJpeg/contents:0': image_data})

			# Sort to show labels of first prediction in order of confidence
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			firstElt = top_k[0];
			return ObjectClassifier.label_lines[firstElt]
			
			# # Other predictions
			# for node_id in top_k:
			# 	human_string = label_lines[node_id]
			# 	score = predictions[0][node_id]
			# 	print('%s (score = %.5f)' % (human_string, score))

