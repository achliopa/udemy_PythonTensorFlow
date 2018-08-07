# Udemy Course: Complete Guide to TensorFlow for Deep Learning with Python

* [Course Link](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/v4/overview)

## Section 1 - Introduction

### Lecture 1 - Introduction

* we will learn how to use Google's TensorFlow framework to prerform the latest techniques in the Deep Learning ecosystem
* we wll start with a crashcourse in the essential Python data science libraries (we know them from PythonDSMLBootcamp course)
	* numpy
	* pandas
	* matplotlib
	* scikit learn
* we will then dive straight into Neural Networks (Perceptrons, Activation Functions, Back Propagation etc)
* Then we ll dive in Tensorflow
* A variety of networks are covered
	* Densely Connected Neural Networks (basic classification and regression tasks)
	* Convolutional Neural Networks (complex image classification tasks)
	* Recurrent Neural Networks + Long-Short Term Memory Units + Gated Recurrent units (analyze sequences of data like time series)
	* Word2Vec algorithm (word embeddings)
	* Auto Encoders (revised tasks)
* Generative Adversarial Networks 
* Reinforcement Learning with [OpenAI Gym](https://gym.openai.com/) to teach algorithms to play video games

### Lecture 2 - Course Overview

* reference notebooks.
* use the course environment file with provided notebooks
* run the provided notebook in the provided environment
* the curriculum
	* Installation and Set-Up
	* Machine-Learning Overview
	* Crash Course
	* Intro to Neural Networks
	* TensorFlow Sections

## Section 2 - Installation and Setup

### Lecture 5 - Intalling TensoFlow Environment

* we go to anaconda.com (anaconda is a high performance python distribution with of a lot of data science packages)
* we download for ubuntu (python3) => follow installation instructions
* do not add anaconda to Path (unless we want anaconda to be our main distro for python). for this course it is ok to do so
* with anaconda installed on our machine its time to restore the env file
* we open the terminal
* cd to where the tfdl_env.yml file is placed
* then we run `conda env create -f tfdl_env.yml`
* we now activate the conda env `source activate tfdeeplearning`
* we run `jupyter notebook` and it should open a browser for us
* to check installation of tensorflow in jupyter
```
import tensorflow as tf
hello = tf.constant("hello world")
sess = tf.Session()
print(sess.run(hello))
>>>b'hellow world'
```

## Section 3 - What is Machine Learning

### Lecture 6 - Machine Learning Overview

* recap from PythonDSMLBootcamp course lecture
* we will talk about supervised learning,unsupervised learning, reinforced learning , evaluation methods and more (AGAIN)
* unlike normal computer programs machine learning techniques learn from data
* ML algos can find insights in data even if they are not specifically instructed what to look for in data
* we define a set of rules. We do not tell it what to do
* three Major types of ML algos
	* Supervised,
	* Unsupervised
	* Reinforced
* We ll also look into word embeddings with Word2Vec
* *Supervised Learning*:
	* uses *labeled* data to predict a label given some features
	* if a label is *continuous* it is called a regression problem, if its *categorical* it is a *classification* problem
	* it has the model train on historical data that is already labeled
	* once the model is trained it can be used on new data where only the feats are known, to attempt prediction
* If we dont have labels for our data (only feats) we have no right answer to fit on. we can only look for patterns in the data and find structures AKA *Unsupervised Learning*:
	* *clustering* problems: e.g heights and weights for breeds of dogs. (no labels). we cluster the data into similar groups (clusters). its up to the scientist to interpret the clusters (assign lables 'breeds'). clustering algorithms cluster datapoints based on common feats
* *Reinforced Learning*:
	* algorithms that learn how to play a video game, drive a car etc
	* it works through trial and error to find which actions yield the greatest rewards
* *Reinforced Learning Components*:
	* Agent-Learning/Decision Maker
	* Environment - What Agent is interacting with
	* Actions - What the agent can do
* Τhe agent chooses actions that maximize some specified reward metric over a given amount of time
* Learning the best policy with the environment and responding with the best actions
* First we walk through the basic machine learning process for a supervised learning problem
* After we discuss some key differences for unsupervised learning and hold out data sets
* ML process is discussed before (DSMLBootcamp)
* A *Hold Out Set* or Evaluation Set. is a similar process to the Train/Test split. We split in three groups train/test/holdout. we train our model on the train set, we test our model on the test data. we repeat
* once we are ready to deploy, we use our holdout set to get a final metric (evaluation) on the performance after deployment. 
* the reason is that we tune our model to better the results on the test data so we might have bias. that's why we use the holdout set to confirm the metrics as it is a set unseen before by the model.
* we dont retune after we get the metrics from the holdout set. we keep them
* Evaluation Metrics:
	* Supervised Learning-Classification Eval: Accuracy (cirrectly classified/total samples), Recall, Precision
	* Supervised Learning-Regression Eval: MAE,MSE,RMSE (how far we are from the correct continuous value)
	* Unsupervised Learning-Evaluation: Much Harder to Evaluate, Dont have correct lables to compare with, Cluster Homogenity, Rand index, Scatterplot evaluation
	* Reinforcement Learning-Evaluation: usually more obvious, the evaluation is built into the actual training of the model, how well the model performs the task it is assigned to

## Section 4 - Crash Course Overview

### Lecture 7 - Crash Course Section Intro

* Nupmy,Pandas,matplotLib + pandas data viz, scikit learn prerpocessing
* Check the sections of PythonDSMLBootcamp Transcript.

## Section 5 - Introduction to Neural Networks

### Lecture 14 - Introduction to Neural Networks

* This series of lectures will cover key theory aspects
	* Neurons and Activation Functions
	* Cost Functions
	* Gradient Descent
	* Backpropagation
* Once we get a general high level understanding we will code out all these topics with Python, without the use of a deep learning library
* Then we will move to using TensorFlow
* Understanding a high level overview of these key elements will make it mush easier to understand what is happening when we begin to use TensorFlow
* TensorFlow has direct connections to these concepts in its syntax

### Lecure 15 - Introduction to Perceptron

* Artificial Neural networks (ANN) have a basis in biology
* We ll see how we can attempt to mimic biological neurons with an artificial neuron known as perceptron
* the Biological neuron work in the following way:
	* dendrites (many) feed in the body of the cell electrical signals
	* from the body a single electrical signal is passed to the axon to connect to another neuron
* The Artificial neuron also has inputs and outputs
* A simple neuron model is known as perceptron (2 inputs + 1 output)
	* inputs can have feature values
	* a weight value (adjustable and different per input) is applied on each input
	* weights initialy start off as random
	* inputs are multiplied by weights
	* these results are passed to an activation function (body of neuron): there are many activation functions
	* we assume a simple activation function (e.g. if sum of weighted inputs >0 return 1, if negative return 0)
	* the output is the result of the activation function
	* a possible problem if when inputs are 0. weights dont have any effect and activation always outputs 0
	* we fix it adding a 3rd adjustable input: the bias (we set it initially to 1)
* Mathematically we can represent a perceptron as : Σ[i=0 -> i=n (Wi * xi)] + b

### Lecture 16 - Neural network Activation Functions

* we ve seen how a single perceptron behaves, now lets expand this concept to the idea of a neural network
* lets see how to connect many perceptrons together and how to represent this mathematically
* a multiple perceptrons network is a multilayered network with each layer containing an array of neurons. a alyers output is next layers input
* usually we have an input layer some hidden layers and an output layer
* Input Layers get the real values from the data
* Hidden Layers are in between the input and output layer, a network of 3 or more hidden layers is considered DeepNN (DNN)
* Output Layer is the final estimate of the output
* As we go forward through more layers the level of abstraction increases
* We ll now look into the activation function
* in our previous simple example the activation function was simple kind like a step function (y = 0 or 1)
* this is a dramatic function as small changes are not reflected
* it would be nice if we could have a more dynamoc function, like an S-curve (logistic curve or sigmoid function)
* we can use sigmoid function for this f(x) = 1/(1+e^(-x)) where x is z = Σwx+b
* some other activation functions: 
	* Hyperbolic Tangent tanh(z) = sinh(z)/cosh(z) , cosh(z) = (e^z+e^(-z))/2 ,  sinh(z) = (e^z-e^(-z))/2 looks like sigmoid function shape  but output varies between -1 and 1
	* Rectified Linear Unit (ReLU): this is actually a relative simple function: max(0,z) (like a diode)
* ReLU and tenh tend to have the best performance, so we will focus on them
* Deep Learning libs have these built in for us. we dont need to worry about having to implement them manually

### Lecture 17 Cost Functions

* we will explore how we can evaluate the performance of a neuron
* we can use the cost function to measure how far we are from, the expected value
* we will use the following vars
	* y to represent the true value
	* a to represent neuron's prediction
* in terms of weights and bias
	* w * x + b = z
	* Pass z into activation function e.g sigmoid function σ(z)=a
* the first cost function we see is *Quadratic Cost* : C = Σ(y-a)^2 / n
* larger errors are more prominent due to squaring
* this calculation can cause a slowdown in our learning speed
* *Cross Entropy* C = (-1/n)Σ(y * ln(a)) + (1-y) * ln(1-a)
* this cost function allows for faster learning
* the larger the difference, the faster the neuron can learn
* We have some good options to start with for 2 aspects of DNN learning. neurons and their activations functions and the cost function
* we are missing the learning part
* we have to figure out how we can use our neurons and the measurement of error (cost function) and then attempt to correct the prediction AKA learn
* we will see how we can do it with gradient descent

### Lecture 18 Grandient Descent Backpropagation

*Gradient Descent* is an optimization algorithm for finding the minimum of a function
* to find a local minimum we take steps proportional to the negative of the gradient
* gradient descent in 1 dimension plot (cost in y axis, a weight in x axis)  is like an U curve
* gradient is the dF() derivative of function. we find it and see which way it goes in the negative direction. we follow the plot untill we reach the minimum C (bottom of curve)
* what we get is we find the weight that minimizes cost
* finding this minimum is simple for 1 dimension. but our ML cases will have more params, meaning we will need to use the built-in linear algebra that our Deep Learning lib will provide
* Using gradient descent we can figure out the best params for minimizing our cost. e.g finding the best values for the weights of the neuron inputs
* our problem to solve is how to quickly adjust the optimal params or weights across our entire network
* *Backpropagation* is the way
* Backrpropagation is used to calculate the error contribution of each neuron after a batch of data is processed
* it relies heavily on the chain rule to go back through the network and calculate these errors
* *Backpropagation* works by calculating the error at the output and then distributes back through the network layers
* It requires a known desired output for each input value (supervised learning)
* the backpropagation implementation will be clarified when we dive into the math example

### Lecture 19 - TensorFlow Playground

* [TensorFlow Playground](http://playground.tensorflow.org) allows us to visualize the topics we  were talking about in the theory lectures
* we select our data aout of various datasets
	* we can add noise
	* play with train/test ratio
	* batch size
* we select our problem type (classification ore regression)
* we prerprocess our dataset (feature selectiona nd extraction)
* we add layers
* set the activation functions, the epochs
* see the output (in rt reduction of loss)
* we see the progress of perfecting the classification as we add layers
* we see in rt lines between the nerurons doing work (weight beign adjusted)
* also we can see the bias as a dot next top to the neurons
* we can modify the model adding layers and nurons and see the loss getting smaller
* we choos ethe spiral dataset. the model is struggling to train on it. to improve it
	* add hidden layers
	* change activation fuynction to relu and do a 5* 6 nurons and 4 in output 
	* we get a good output
* we max out 6 layers * 8neurons
* if we put learnign rate to 10 it learns faster but output is like

### Lecture 20 - Manual Creation of neural Network Part 1

* we will mimic tensorflow coding the classses and functions needed in a neural network ourselves
* we ll start with OOP in python and the *super()* keywords
* we create a simple class and define methods in it (also __init__ constructor)
```
class SimpleClass():
	def __init__(self,name=''):
		print('hello'+name)
	def yell(self):
		print('YELLING')
```
* we also set a string literal `s="world` of type str and see its built in methods
* to create an instance of a class we write `x = SimpleClass()` this call executes the __init__ method (constructor)
* if I write `x` i see the address of the specific object
* `x.yell()` >>YELLING
* we write a child class of SimpleClass which inherits it
```
class ExtendedClass(SimpleClass):
	def __init__(self):
		print('EXTEND!')
```
* we instantiate it `y = ExtendedClass()`
* y instance inherits all properties of its parent class `y.yell()` >> YELLING
* if we want the parent class constructor to exectute when we instantiate the child we write in its constructor `super().__init__()`. we can pass args in
```
class ExtendedClass(SimpleClass):
	def __init__(self):
		super().__init__('Jose')
		print('EXTEND!')
```

### Lecture 21 - Manual Creation of neural Network Part 2: Operations

* The Operation class we will create will have:
	* Input Nodes attribute (list of nodes)
	* Output Nodes attribute (list nodes)
	* Global default graph variable
	* Compute method overwritten by extended classes
* Graph will be a global var, tensorFlow runs on Graphs. we can envision it as a list of nodes. a simple graph will have some constant nodes (n1,n2) and an operation node (n3). the operation node will be a child class of the generic Operation class
* we implement the Operation class
	* at instantiation we get a list of the input node, we add the operation instance in the output node list of the input nodes
	* the compute method is an abstract method. left for instantiation to the children
```
class Operation():
	def __init__(self,input_nodes=[]):
		self.input_nodes = input_nodes
		self.output_nodes = []

		for node in input_nodes:
			node.output_nodes.append(self)
	def compute(self):
		pass
```
* we create a child class *add* to the Operation. it takes 2 input nodes. its compute method adds them up. also it satisfies the partent constructor req providing them as a list. also it has its own attribute of inputs
```
class add(Operation):
	def __init__(self,x,y):
		super().__init__([x,y])

	def compute(self,x_var,y_var):
		self.inputs = [x_var,y_var]
		return x_var+y_var
```
* we implement two more operation  child classes (named in lowercase per tensorFlow convention) multiplication and matrix multiplication (do the dot vector multiplication)
```
class multiply(Operation):
	def __init__(self,x,y):
		super().__init__([x,y])

	def compute(self,x_var,y_var):
		self.inputs = [x_var,y_var]
		return x_var*y_var

class matmul(Operation):
	def __init__(self,x,y):
		super().__init__([x,y])

	def compute(self,x_var,y_var):
		self.inputs = [x_var,y_var]
		return x_var.dot(y_var)
```

### Lecture 22 - Manual Creation of neural Network Part 3: Placeholders and Variables

* Placeholder: An empty node that needs a value to be provided to compute output
* Variables: Changeable parameter of Graph
* Graph: Global Variable connecting variables and placeholders to operations
* we create the classes
```
class Placeholder():
	def __init__(self):
		self.output_nodes = []
		_default_graph.placeholders.append(self)

class Variable():
	
	def __init__(self,initial_value=None):
		self.value = initial_value
		self.output_nodes = []
		_default_graph.variables.append(self)

class Graph():
	def __init__(self):
		self.operations = []
		self.placeholders = []
		self.variables = []

	set_as_default(self):
		global _default_graph
		_default_graph = self
```

* when we call `_default_graph.variables.append(self)` we append the current instance to the list of the graph
* we implement a simple example: z = Ax + b (A=10, b=1) => z=10x+1
* our code looks like (x we dont know what it is so we treat it as placeholder)
```
g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)
x = Placeholder()
y = multiply(A,x)
z = add(y,b)
```
* to compute we need a traverse-post-order function to do post traversal of nodes (keep correct order of computations) it is common in treee theory
* we need a session class that executes the whole thing

### Lecture 23 - Manual Creation of neural Network Part 4: Session

* Now that we have all nodes ready we need to execute all the operations within a Session
* we will use the *Postorder Tree Traversal* to make sure we execute the nodes in the correct order
* we cp the post order traverse function
```
def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
```
* now we will code the Session class,
* we create a run method passing the operation we want to compute and a feed dictionary as tensorflow actually expects in its sessions. its a dictionary mapping placeholders to actual input values
* in run we distinguish between node types taking action
* `(*node.inputs)` notation is pythons *args (variable number of arguments)
```
clas Session():
	def run(self,operation,feed_dict={}):
		nodes_postorder = traverse_postorder(operation)

		for order in nodes_postorder:
			if type(node) == Placeholder:
				node.output = feed_dict[node]
			elif type(node) == Variable:
				node.output = node.value
			else
				# Operation => fill its inputs
				node.inputs = [input_node.output for input_node in node.input_node]
				node.output = node.compute(*node.inputs)

			if type(node.output) == list:
				node.output = np.array(node.output)

		return operation.output
```
* we execute it. 
```
sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
```

* our result is 101
* we try it with a matrix. we create a graph
```
g = Graph()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,2,])
x = Placeholder()
y = matmul(A,x)
z = add(y,b)
sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
```
* our result is an 2*2 matrix

### Lecture 24 - Manual Neural network Classification Task

* We will do linear classification using out custom neuron (perceptron)
* we need an activation function to classify based on results
* we will plot the activation function to see it graphicaly
```
import matplotlib.pyplot as plt
%matplotlib inline
```
* we create the sigmoid function
```
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
```
* we make a linspace and pass it to the func ploting x to y
```
sample_z = np.linspace(-10,10,100)
sample_a = signmoid(sample_z)
plt.plot(sample_z,sample_a)
```
* we make Sigmoid an operation to use it in our ANN
```
class Sigmoid(Operation):
	def __init__(self,z):
		super().__init__([z])

	def compute(self,z_val):
		return 1 / (1+ np.exp(-z_val))
```
* we need datasets to do a proper classification so we import from sklearn
```
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=75)
```
* data is a tuple with 2 eleemtns. firstis an array and second is a label val
* we extract and plot the feats
```
featues = data[0]
labels = data[1]
plt.scatter(features[:,0],features[:,1],c=labels)
```
* they are clearly separable so clasification is easy
* we first draw a line that separates the clusters on the plot
```
x = np.linspace(0,11,10)
y = -x + 5
plt.plot(x,y)
```
* we feed the  perceptron using a matrix representation of the featts (tensorflow wants it)
* x and y in our plot are feats. so we need to make a featmatrix (pivot table)
* our separation line is `feat2 = -1*feat1 + 5` = > feat2+feeat1 -5 = 0 => featmatrix[1,1] -5 = 0 => (1,1) * f -5 = 0 where f is 2by1 matrix `np.array([1,1]).dot(np.array([[8,[10]]) -5` is 13 > 0 so class1 this is how perceptron will look like
* we code our network
```
g = Graph()
g.set_as_default()
x  = Placeholder()
w = Variable([1,1])
b = Variable(-5)
z = add(matmul(w,x),b)
a = Sigmoid(z) # activator. good choice as we classify on 0
sess = Session()
sess.run(operation=a, feed_dict={x:[8,10]}) 
```
* we get 99.9% certainty its in class 1

## Section 6 - TensorFlow Basics

### Lecture 25 - Introduction to TensorFlow

* that section will expand on what we ve learned and explore the TensorFlow framework approach to Neural Networks
* we ll see a lot of similarities with our simplw python implementation
* TensorFlow Basics
	* TF Basic Syntax
	* TF Graphs
	* TF Variables
	* TF Placeholders
* TensorFlow NeuralNetwork
* TensorFlow  Regression and Classification Example and Exercise

### Lecture 26 - TensorFlow Basic Syntax

* we import tensorflow `import tensorflow as tf`
* we check version `print(tf.__version__)`
* we create a tensor (1d array) constant passing a string literal 
```
hello = tf.constant('Hello ')
world = tf.constant('World')
```
* we check its type `type(hello)` >> tensorflow.python.framework.ops.Tensor. 
* its a Tensor object. if we try to print it we get no literal. to see the content we need to do the print in a session
* to create a session we write `with tf.Session() as sess:` what we write in the block is the session code
```
with tf.Session() as sess:
	# Session code
	result = sess.run(hello+world)
```
* if we `print(result)` after the session we get >> b'Hello World'
* if we set numeric constants
```
a = tf.constant(10)
b = tf.constant(20)
```
* and check their type thery are Tensor objects
* if we add them outside a session `a+b` the result is a Tensor which has the add of type int32 but we dont get the result. to get the result we should do the addition in a session run
```
with tf.Session() as sess:
	result = sess.run(a+b)
```
* if i see result after session run is 30
* *tf.fill* fills a matrix of certain dimensions with a value `fill_mat = tf.fill((4,4),10)` dimensions are passed as a tuple
* *tf.zeros* fills a matrix of certain dimensions with zeroes `myzeros = tf.zeros((4,4))`
* *tf.ones* does the same as zeroes with 1s
* we can fill a matrix with random vals froma normal distrr with *tf.random_normal* `myrandn = tf.random_normal((4,4),mean=0,stddev=1.0)`
* we can use uniform distribution instead *tf.random_uniform* `myrandu = tf.random_uniform((4,4),minval=0,maxval=1)`
* we add all these methods to a list `my_ops = [const,fill_mat,myzeros,myones,myrandu,myrandn]`
* we will execute it in a session. apart from teh syntax we saw thereis an interactive session syntax which is useful for notebooks
```
sess = tf.InteractiveSession()
for op in my_ops:
	print(sess.run(op))
```
* we see the printouts on screen
* instead of `sess.run(op)` we can run `op.eval()` in a session context with the same results
* matrix multiplication is amajor topic of tensorflow basics. we define a 2by2 matrix of constants `a = tf.constant([[1,2],[3,4]])` as a tensor object
* if we call `a.get_shape()` >> TensorShape([Dimension(2),Dimension(2)])
* we define a 2by1 matrix `b = tf.constant([[10],[100]])`
* we multiply them `result = tf.matmul(a,b)` we call `sess.run(result)` in an interactive session and get the 2by1 array as a result

### Lecture 27 - TensorFlow Graphs

* Graphs a re sets of connected nodes (vertices)
* the connections are referred to as edges
* In TensorFlow each node is an operation with possible inputs that can supply some output
* with TensorFlow we will construct a graph and then execute it
* we will build a simple graph of 3 nodes (2 input nodes of constants + one operation node that adds the two)
* we code it 
```
import tensorflow as tf
n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1+n2
```
* this is a primitive graph we can run in a session
```
with tf.Session() as sess:
	result = sess.run(n3)
```
* we print the result outside the session `print(result)` and get 3
* if we print the n3 we get its strigified version as a Tensor of add type operating on int32
* when we start TensorFlow a default graph is created. we can easily create additional graphs
* we can retrieve the default graph object `print(tf.get_default_graph())` >> <tensorflow.python.framework.ops.Graph object at 0x000001971C96F8D0>
* we can create a grapoh object `g = tf.Graph()` and print it. the output is the same but it resides in another prt of the memory
* to set another graph as default we use
```
with g.as_default():
	print (g is tf.get_default)graph())
```
* so when we set as_default() we do it in a sepcific context (session like) where we run our code (temporary assignment)

### Lecture 28 - Variables and Placeholders

* There are two main types of tensor objects in  a Graph. Variables and Palceholders
* during the optimization process tensorflow tunes the parameters of the model
* variables can hold the values of weights and biases throughout the session
* variables need to be initialized
* Placeholders are initialy empty and are used to feed in the actual training examples
* however they do need a declared expected data type (e.g tf.float32) with an optional shape argument
* we see them in action in anew notebook
* we import tensorflow `import tensorflow as tf`
* we create an interactive session `ses s= tf.InteractiveSession()`
* we create a tensor of size 4by4 of uniform random nums 0-1 `my_tensor = tf.random_uniform((4,4),0,1)`
* we create a variable `my_var = tf.Variable(initial_value = my_tensor)` passing a tensor as initial val
* if we run our var in a session we get an error because we need to initalize it first
* we do this by runing the grobal variables initializer to initalize the variable first
```
init = tf.global_varialbes_initializer()
sess.run(init)
```
* if we now run `sess.run(my_var)` we see the array in output
* we create a placeholder `ph = tf.placeholder(tf.float32)` usually we set the shape as a tuple of (None,5) we use none because it can be filled by the actual number of samples in the data passed in fed in batches

### Lecture 29 - TensorFlow: A Neural Network Part One

* we will build our first TF neural network
* we ve learned about Sessions, Graphs, Varialbes and Placeholders
* usining these building blocks we can create our first neuron
* we will create a neuron that performs a simple linear fit to 2d data
* Our steps are:
	* Build a graph
	* initate a session
	* feed data in and get output
* we ll use the basics learned so far to  pull through the task
* our graph will execute wx+b=z
	* w: variable
	* x: placeholder
	* y = tf.matmul(w,x): operation
	* b: variable
	* z = tf.add(y,b): operation
	* z passed through activation function (e.g sigmoid)
* afterwards we can add in the cost function in order to train the network and optimize the params
* we import numpy and tensorflow
```
import numpy as np
import tensorflow as tf
```
* we add random seeds 
```
np.random.seed(101)
tf.set_random_seed(101)
```
* we create a random 5by5 array `rand_a = np.random.uniform(0,100,(5,5))`
* and one 5by1 `rand_b = np.random.uniform(0,100,(5,1))`
* we create two placeholders 
```
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
```
* in convolutional neural networks placeholder shape is important
* we create our operations using tf default math
* tensorflow can understand math expressions ttrnasforming the to build ins `a+b => tf.add(a,b)`
```
add_op = a+b
mul_op = a*b
```
* a and b are placeholder objects so in a session we need to feed them with dictionaties of inputs
```
with tf.Session() as sess: 
	add_result = sess.run(add_op, feed_dict={a:10,b:20})
	print(add_result)
```
* we get 30.0 as result
* we repeat the drill passing in the dictionaries the random numpy arrays
```
with tf.Session() as sess: 
	add_result = sess.run(add_op, feed_dict={a:rand_a,b:rand_b})
	print(add_result)
	mult_result = sess.run(mul_op, feed_dict={a:rand_a,b:rand_b})
	print(mult_result)

```

### Lecture 30 - Tensorflow: A Neural Network Part Two

* we create a simple neural network
* usually we define our global constnat vars in atop cell e.g num of feats `n_features = 10` or the dencity of neurons per layer `n_dense_neurons = 3`
* we create teh placeholder for x `x = tf.placeholder(tf.float32,(None,n_features))` keeping the num of samples open ofr the data batches
* we define our variables (weights and bias) initalizing them with random nums
```
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones[n_dense_neurons])
```
* W gets multiplied to x so num of rows for W is the same as num of cols for x
* we now define operations and activation functions
```
xW = tf.amatmul(x,W)
z = tf.add(xW,b)
```
* we pass z to the activation function to get the output `a = tf.sigmoid(z)`
* we are now ready to run all steps in a session, first we need to initialize vars 
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	layer_out = sess.run(a,feed_dict=x:np.random.random([1,n_features]))
```
* we print the output is a 3 by 1 array with vals between 0 and 1
* what we have done so far is just one iteration. we are not adjusting our variables (w and b trying to minimize the error or cost
* we need a cost function and an optimizer to do it
* we show this with a simple regression example
* we set x data as a linspace from 0 to 10 adding a bit of noise `x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)`
* we also create some labels in the same way ``y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)`
* we import matplotlib and we plot them `plt.plot(x_data,y_label,'*')`. we see that there is a linear trend despite the randomness so a good candidate for regression
* we now go and create our neural network in tensorflow (y = mx+b)
* we create our vars passin in random initial nums we get from numpy
```
np.random.rand(2) # we get arra([0.44242354, 0.87758732])
m = tf.Variable(0.44)
b = tf.Variable(0.87)
```
* we create the cost function for our ANN using the square error
```
error = 0

for x,y in zip(x_data,y_label):
	y_hat = m*x+b
	error +=(y-y_hat)**2
```
* we need to minimize the error. we use an optimizer for that (Gradiewnt descent). we tell it explicitly what it has to otpimize
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
```
* we are almost ready . we initalize vars `init = tf.global_variables_initializer()`
* we create the session and run it
```
with tf.Session() as sess:
	sess.run(init)
	training_steps = 1
	for i in range(training_steps):
		sess.run(train)
	final_slope, final_intercept = sess.run([m,b])
```
* we expect bad results with just 1 training session. we evaluate the results with some test data
```
x_test = np.linspace(-1,11,10)
# y = mx+b
y_pred_plot = final_slope*x_test+final_intercept
```
* we overlay the prediction as a line on our train data. it is ok
```
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')
```
* if we set training_steps to 100 the fit is better

### Lecture 31 - Tensorflow Regression Example: Part One

* we ll code along a more realisting regression example and introduce the *tf.estimator*
* tensorflow is used for tasks that classical ML algorithms cannot solve. estimator is used for the simple ML problems
* we import numpy, pandas and matplotlib and set it inline, we import tensorflow
* we will create a very large dataset ourselves with linspace `x_data = np.linspace(0.0,10.0,1000000)`
* we add some noise `noise = np.random.randn(len(x_data))` a million normal distr random points
* we will use for our linear fit for our second feat (y of datapoint or label)  y = mx + b where b=5 and m = 0.5 `y_true = (0.5 * x_data) + 5 + noise` so its not a perffectly fitted line
* we convert our x_data to a dataframe `x_df = pd.DataFrame(data=x_data,columns=['X Data'])` 
* we convert our y data to a Dataframe `my_data = pd.DataFrame(data=y_true,columns=['Y'])`
* we concat 2 dfs along the y axis `my_data = pd.concat([x_df,y_df],axis=1)` in that way we can easily plot it out
* we want to plot the dataset but its huge. we can select random samples of a df with `.sample(n=numofsamples)` we plot that `my_data.sample(n=250).plot(kind='scatter',x='X Data', y='Y')`
* as expected we see a linear trend with randomness
* we want tensorflow to find the linear fit
* we cannot feed 1 million points in a NN we should feed batches
* we start creating our NN
```
batch_size = 8
random = np.random.randn(2)
m = tf.Variable(random[0])
b = tf.Variable(random[1])
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])
y_model = m*xph + b # my graph
# we create our cost function using tensorflow optimized functions
error = tf.reduce_sum(tf.square(yph-ymodel))
# we define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	batches = 1000
	for i in range(batches):
		# choose 8 random index points
		rand_ind = np.random.randint(len(x_data),size=batc_size)
		feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
		sess.run(train,feed_dict = feed)
	model_m,model_b = sess.run([m,b])
```