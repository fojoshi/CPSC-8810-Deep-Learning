Keep all the files in the same directory.
To run the training, please use the bleu_eval.py file provided.

For training, the directory must also contain the following:
1. Training data
2. Test Data
3. bleu_eval.py file



To run testing:
sh hw2_seq2seq.sh testing_data/feat op.txt

Please specify a directory path containing directory feat/ inside with all the .npy files as 2nd argument.

To run training:
python training.py

