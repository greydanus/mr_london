#MrLondon
A LSTM recurrent neural network implemented in pure numpy. Web demo [here](https://mr-london.herokuapp.com/)

Description
-----------
I built this project to teach myself about how recurrent neural networks function. First, I made a dataset that consisted of several million words from Jack London novels.

Next, I trained a deep Long Short Term Memory (LSTM) recurrent neural network using Keras and saved the weights using python's pickle utility. Finally, I rewrote all the forward propagation code in pure numpy (as opposed to Theano or Tensorflow as in Keras). This lets me run the network as a demo via heroku.

The model runs a bit slow on the web app, but keep in mind that it had to learn English vocabulary, grammar, and punctuation at the character level!

Dependencies
--------
*All code is written in python 3.

Dependencies are packaged in the flask folder, so this app does not have any external depencies.

Examples
--------
Here are a few examples of text written by MrLondon:

![sample_1.png](https://github.com/greydanus/mr_london/blob/master/app/static/img/sample_1.png)
--------
![sample_2.png](https://github.com/greydanus/mr_london/blob/master/app/static/img/sample_2.png)
--------
![sample_3.png](https://github.com/greydanus/mr_london/blob/master/app/static/img/sample_3.png)
