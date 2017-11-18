python /home/dev/Work/tensorflow/tensorflow/examples/image_retraining/retrain.py \
--image_dir='/home/dev/Work/tensorflow_Inception/human/processed_dataset' \
--output_graph='/home/dev/Work/tensorflow_Inception/human/graph.pb' \
--output_labels='/home/dev/Work/tensorflow_Inception/human/labels.txt' \
--summaries_dir='/home/dev/Work/tensorflow_Inception/human/retrain_logs/' \
--how_many_training_steps='30000' \
--bottleneck_dir='/home/dev/Work/tensorflow_Inception/human/bottleneck/'
