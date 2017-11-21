In this experiment we will be using **Inception v3** architecture model for 'Object Detection' problem


## Steps to retrain the image classifier:

1. Clone the git repository - https://github.com/tensorflow/tensorflow.git
2. The retrain script is present at tensorflow/tensorflow/examples/image_retraining/ folder
3. To execute the script create a folder which contains the images (the images should be placed inside a subfolder with the name of the lable)
for eg. images/car/ folder should contain the images of car

**Example :**
```
python /home/dev/tensorflow/tensorflow/examples/image_retraining/retrain.py \
--image_dir='/home/dev/work/deep_learning/object_detection/images/' \   # the folder which contains the images
--output_graph='/home/dev/work/deep_learning/object_detection/graph.pb' \	# the path and name of graph to be stored
--output_labels='/home/dev/deep_learning/object_detection/labels.txt' \ # the output labels of the trained graph (auto generated file)
--summaries_dir='/home/dev/deep_learning/object_detection/retrain_logs/' \ # the summaries to view in tensorboard
--how_many_training_steps='4000' # the number of steps to train the network
```

4. Update the newly trained model ('output_graph'), in classification system.
