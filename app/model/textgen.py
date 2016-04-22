import pickle
import numpy as np

class LiteTextGen:
	def __init__(self, fn=None):
		self.maxlen = 20
		self.generated = ''
		self.primer = "I want to have a big"

		self.LSTM = LiteLSTM()
		if fn is None:
			fn = "app/model/lstm_weights.pkl"
		self.load_model(fn)

	def load_model(self, fn):
		[weights, meta] = pickle.load(open(fn, 'rb'), encoding='latin1')
		self.LSTM.load_weights(weights)
		self.char_indices = meta['char_indices']
		self.indices_char = meta['indices_char']

	def clean_slate():
		self.generated = ''
		self.primer = "This primer has 20 c"

	def predict(self, primer = None, length = 1, stream=True, diversity = 0.2):
		if primer is None or stream:
			pass
		elif type(primer) is not str:
			raise ValueError("Primer must be a string")
		else:
			self.primer = ' ' * (self.maxlen - len(primer)) + primer
			self.primer = self.primer[-20:]

		assert 1 <= length <= 1000, "Length of string to generate must be between 1 and 1000"
		self.primer = self.primer.lower()

		for i in range(length):
			X = self.vectorize_primer()
			next_index = self.LSTM.predict_i(X, diversity)
			next_char = self.indices_char[next_index]

			self.generated += next_char
			self.primer = self.primer[1:] + next_char
		if stream:
			return next_char
		self.generated = self.generated
		txt = self.generated.split('\n')
		return txt

	def vectorize_primer(self):
		X = np.zeros((1, self.maxlen, len(self.char_indices)))
		for t, char in enumerate(self.primer):
			X[0, t, self.char_indices[char] ] = 1.
		return X

class LiteLSTM:
	def __init__(self):
		self.layers = []

	def load_weights(self, weights):
		assert not self.layers, "Weights can only be loaded once!"
		for k in range(len(weights.keys())):
		    self.layers.append(weights['layer_{}'.format(k)])

	def predict_i(self, X, diversity):
		assert not not self.layers, "Weights must be loaded before making a prediction!"
		h = self.lstm_layer(X, layer_i=0, seq=True) ; X = h
		h = self.dropout(X, .2) ; X = h
		h = self.lstm_layer(X, layer_i = 2, seq=False) ; X = h
		h = self.dropout(X, .2) ; X = h
		h = self.dense(X, layer_i = 4) ; X = h
		h = self.softmax_2D(X) ; X = h[0] #convert it from a [n,1] tensor to a [n] vector
		i = self.sample(X, diversity)
		return i

	def predict_classes(self, X):
		assert not not self.layers, "Weights must be loaded before making a prediction!"
		h = self.lstm_layer(X, layer_i=0, seq=False) ; X = h
		h = self.repeat_vector(X, 4) ; X = h
		h = self.lstm_layer(X, layer_i=2, seq=True) ; X = h
		h = self.timedist_dense(X, layer_i=3) ; X = h
		h = self.softmax_2D(X) ; X = h
		preds = self.classify(X) ; X = h
		return preds

	def lstm_layer(self, X, layer_i=0, seq=False):
		X = np.flipud(np.rot90(X))

		#load weights
		w = self.layers[layer_i]
		W_i = w["W_i"] ; U_i = w["U_i"] ; b_i = w["b_i"] #[n,m] ; [m,m] ; [m]
		W_f = w["W_f"] ; U_f = w["U_f"] ; b_f = w["b_f"]
		W_c = w["W_c"] ; U_c = w["U_c"] ; b_c = w["b_c"]
		W_o = w["W_o"] ; U_o = w["U_o"] ; b_o = w["b_o"]

		#create each of the x input vectors for the LSTM
		xi = np.dot(X, W_i) + b_i
		xf = np.dot(X, W_f) + b_f
		xc = np.dot(X, W_c) + b_c
		xo = np.dot(X, W_o) + b_o

		hprev = np.zeros((1, len(b_i))) #[1,m]
		Cprev = np.zeros((1, len(b_i))) #[1,m]

		[output, memory] = self.nsteps(xi, xf, xo, xc, hprev, Cprev, U_i, U_f, U_o, U_c)

		if seq:
		    return np.flipud(np.rot90(output))
		output = np.reshape(output[-1,:,:],(1,output.shape[1], output.shape[2]))
		return np.flipud(np.rot90(output))

	def nsteps(self, xi, xf, xo, xc, hprev, Cprev, U_i, U_f, U_o, U_c):
		nsteps = xi.shape[0] # should be n long
		output = np.zeros_like(xi) # [n,1,m]
		memory = np.zeros_like(xi) # [n,1,m]

		for t in range(nsteps):
			xi_t = xi[t,:,:] ; xf_t = xf[t,:,:] ; xc_t = xc[t,:,:] ; xo_t = xo[t,:,:] # [1,m] for all

			i_t = self.hard_sigmoid(xi_t + np.dot(hprev, U_i)) #[1,m] + [m]*[m,m] -> [1,m]
			f_t = self.hard_sigmoid(xf_t + np.dot(hprev, U_f)) #[1,m] + [m]*[m,m] -> [1,m]
			o_t = self.hard_sigmoid(xo_t + np.dot(hprev, U_o)) #[1,m] + [m]*[m,m] -> [1,m]
			c_t = f_t*Cprev + i_t * np.tanh(xc_t + np.dot(hprev, U_c)) #[1,m]*[m] + [1,m] * [1,m] -> [1,m]
			h_t = o_t * np.tanh(c_t) #[1,m]*[1,m] (elementwise)

			output[t,:,:] = h_t ; memory[t,:,:] = c_t
			hprev = h_t # [1,m]
			Cprev = c_t # [1,m]

		return [output, memory]

	def dense(self, X, layer_i=0):
		w = self.layers[layer_i]
		W = w["W_i"]
		b = w["U_i"]
		output = np.dot(X, W) + b
		return output

	def timedist_dense(self, X, layer_i=0):
		w = self.layers[layer_i]
		W = w["W_i"]
		b = w["U_i"]
		output = np.tanh(np.dot(np.flipud(np.rot90(X)), W) + b)
		return np.flipud(np.rot90(output))

	@staticmethod
	def sigmoid(x):
		return 1.0/(1.0+np.exp(-x))

	@staticmethod
	def hard_sigmoid(x):
		slope = 0.2
		shift = 0.5
		x = (x * slope) + shift
		x = np.clip(x, 0, 1)
		return x

	@staticmethod
	def repeat_vector(X, n):
		y = np.ones((X.shape[0], n, X.shape[2])) * X
		return y

	@staticmethod
	def softmax_2D(X):
		w = X[0,:,:]
		w = np.array(w)
		maxes = np.amax(w, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		e = np.exp(w - maxes)
		dist = e / np.sum(e, axis=1, keepdims=True)
		return dist

	@staticmethod
	def classify(X):
		return X.argmax(axis=-1)

	@staticmethod
	def sample(X, temperature=1.0):
		# helper function to sample an index from a probability array
		X = np.log(X) / temperature
		X = np.exp(X) / np.sum(np.exp(X))
		return np.argmax(np.random.multinomial(1, X, 1))

	@staticmethod
	def dropout(X, p):
		retain_prob = 1. - p
		X *= retain_prob
		return X
