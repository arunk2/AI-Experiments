import numpy as np
import tensorflow as tf


class EmotionClassifier():

    with tf.gfile.FastGFile('graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


    def __init__(self, image):
        self.image = image


    def getTags(self):
        answer = None

        if not tf.gfile.Exists(self.image):
            tf.logging.fatal('File does not exist %s', self.image)
            return answer

        image_data = tf.gfile.FastGFile(self.image, 'rb').read()

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            labels = ["sad", "disgust", "anger", "surprise", "fear", "happy"]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                # print('%s (score = %.5f)' % (human_string, score))

            answer = labels[top_k[0]]
            return answer


if __name__ == '__main__':
    classifier = EmotionClassifier('test_images/output20.jpg')
    print 'Emotion: '+classifier.getTags()
