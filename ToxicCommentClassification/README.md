# Toxic Comment Classification Sentiment Analysis model using CNN
In this notebook, a simple CNN architecture is used for multi-label classification by use of fast text embeddings. CNNs can learn patterns in word embedding and as per the dataset of the Toxic Classification Challenge we can use the sub-word information. Here is a link to the <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data">dataset</a>.
#Fast Text Word Embeddings
This code uses pretrained word vectors of 294 languages, trained on Wikipedia using fastText. These vectors in dimension 300 were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters. You can find it <a href="https://fasttext.cc/docs/en/pretrained-vectors.html">here</a>.
