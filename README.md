# Pytorch implementation of 'Augmented Knowledge-Driven Speech-Based Method of Depression Detection Leveraging Vowel Information'
#### utils
It includes the trained vowel CNN model (vowel_CNN.pth), the model structure of vowel CNN (model.py), and the code that extracts the utterance-level embeddings from the data (extract_feature_spec.py)

To run extract_feature_spec.py, you will first need to download the processed feature files from this [Google drive link][https://drive.google.com/drive/folders/1c-dPcSp16-oKLESGyQ8Vv1MIPcc6mA5b?usp=sharing]. These files were not uploaded to GitHub due to the file size restriction. The output of this code is in 'processed_data' folder.

#### processed_data
It includes the feature embeddings from the vowel CNN (feature_fc1), the length (length) and saliency information (saliency). Some other files are directly derived from the original dataset.
#### others
The 'window_size_10', 'window_size_21', and 'window_size_42' folder include a sample running result and the model structure for the related experimental settings. The running environment is also specified in each jupyter notebook file.

For other information related to the training of vowel CNN, it is revised based on our previous BHI 2022 paper: [Toward Knowledge-Driven Speech-Based Models of Depression: Leveraging Spectrotemporal Variations in Speech Vowels][https://github.com/HUBBS-Lab-TAMU/2dCNN-LSTM-depression-identification]
