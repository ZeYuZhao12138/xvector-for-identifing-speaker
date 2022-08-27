# xvector-for-identifing-speaker
input MFCC-feature

you need to import torch and librosa
this code is for training a model to identify speakers

the data has to be longer than 2s audios
and the structure of dataset file has to be:

audio_dataset
audio_dataset/speaker1name
audio_dataset/speaker2name
audio_dataset/speaker1name_test
audio_dataset/speaker2name_test

and the train audios are in file speaker1name & speaker2name
the test audios are in file speaker1name_test & speaker2name_test
