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
* we also create some labels in the same way `y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)`
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
* model_m is 0.52569, model_b is 4.92
* we can visualize it
```
y_hat = x_data*model_m+model_b
my_data.sample(250).plot(kind='scatter',x='X Data', y='Y')
plt.plot(x_data,y_hat,'r')
```
* we repeat for 10k batches, we are getting closer to our linear fit. m=0.5,b=5

### Lecture 32 - TensorFlow Regression Example: Part Two

* We ll now explore the *Estimator API* from TensorFlow
* There are lots of other higher level APIs (e.g Keras), we ll cover them later
* *tf.estimator* has several model types to choose from
	* *tf.estimator.LinearClassifier* constructs a linear classification model
	* *tf.estimator.LinearRegressor* constructs a linerar regression model
	* *tf.estimator.DNNClassifier* constructs a neural network classification model
	* *tf.estimator.DNNRegressor* constructs a neural network regression model
	* *tf.estimator.DNNLinearCombinedClassifier* constructs a neural network and linear combined classification model
* To use the Estimator API we do the following:
	* define a list of feature columns
	* create the Estimator Model
	* create a Data Input Function
	* call train,evaluate, and predict methods on the estimator object
* we create the feature column `feat_cols = [tf.feature_column.numeric_column('x',shape[1])]` there are many options available built in the API. our data shape is 1 column and we label it (key) to 'x'
* feat columns must be alist so we use wrapper. this sets the blueprint for the data imput
* we create our estimator passing in the feature column `estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)`
* we import sklearn train_test_split to split our data `from sklearn.model_selection import train_test_split`
* we do the actual split on the data from previous lecture (artificial data) `x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)`
if we check the shape of data (rows) `prin(x_train.shape)` we get 700000 and 300000 respectively
* we set estimator inputs (like a feed_dict and batch size combined). our input data are numpy arrays (see last lecture) so we use the built in func. there is a func for pandas `input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)`
* we cp the input func for train and test input data  setting a num of epochs
```
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)
```
* we train our estimator passing in the input function. as we didnt spec num of epochs in the input we do it here `estimator.train(input_fn=input_func,steps=1000)`
* we see the training session run and the error geting smaller
* we want to check the training metrics so we run the evaluate method `train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)`. we use the training input function because its not shuffles and it helps do the evaluation correctly
* we do the same procedure for the test data `eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)`
* we compare train metrics and eval metrics by printing them
* by comparing thre results we can see if our model is overfitting to our data (if we have low  loss on training data but high loss on eval data). ideally we want both low and SIMILAR
* if we want to deploy the model we need to know how we will predict the vals
* we simulate that by creating anew bunch of data to pass in. also we create a new input function without y class data
```
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)
```
* we use the predict method to get predictions `estimator.predict(input_fn=input_fn_predict)`
* what we get is a generator object. to get the data we need to cast it into alist `list()` and get alist of dictionaries
* we now want to plot the predictions. we fill a list
```
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
	predictions.append(pred['predictions'])
```
* we plot data fro our dataset `my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')`
* we overlay a line connecting our predictions (linear fit) `plt.plot(brand_new_data,predictions,'r')`

### Lecture 33 - TensorFlow Classification Example: Part One

* we ll work with a real dataset
* we ll work with categorical and continuous feats
* we ll switch estimator models
* our data will be pandas dataframe
* we import pandas and read in our data `diabetes = pd.read.csv('pima-indians-diabetes.csv',)`
* this is a binary classification problem
* we chck the data with .head() and .columns
* we want to normalize specific columns `cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']` so essentialy all numerical colums except Age as we will make it categorical data
* we can normalize with sklearn preprocessing or pandas. we do it with pandas using our own custom lambda func `diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
* we import tensorflow
* we create a feat column for every numeric column (this is a blueprint)
```
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
```
* to create a feature column froma categorical data column there are 2 ways. if we know all possible vaslues we can use a vocabulary list. if not we use a hash bucket
* for assigned group we know there are 4 options so we use vocab list `assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])`
* if we have many groups and dont want to type them in or dont know them we use hash bucket `assigned_group = tf.feauture_column.categorical_column_with_hash_bucket('Groups',hash_bucket_size=10)` hash_bucket_size is the max number of groups
* to convert a continuous column to a categorical columns (e.g Age column) we do the following:
* plot it out as histogram  `diabetes['Age'].hist(bins=20)`
* we can bucketize a continuous column into a categorical using feature_columns `age_bucket =  tf.feature_column.bucketized_column(age,boundaries[20,30,40,50,60,70,80])` we pass in the numerical feature column and the boundaries
* we create alist of all our feature columns `feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]`
* we do the train test split on our data
```
x_data = diabetes['Class',axis=1]
labels = diabetes['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)
```

### Lecture 34 - TensorFlow Classification Example: Part One

* we will now create an input function. as our data is a panda df we will use the specific function `input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10, num_epochs=1000, shuffle=True)`
* we create our model setting the num of classes `model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classses=2)`
* we train our model `model.train(input_fn=input_func,steps=1000)`
* we make an evaluatiion input func for evaluating our model `eval_input_func = tf.estimator.pandas_input_fn(x=X_test,y=Y_test,batch_size=10,num_epochs=1,shuffle=False)`
* we evaluate to get the results `results = model.evaluate(eval_input_func)`
* we print the results. the auc and accuracy are decent but not optimal
* to see the deployment scenario in action we create a new input function with a feat only fresh dataset (we use test data again) `pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10, num_epochs=1, shuffle=False)`
* we do the actual preduction 

```
predictions = model.predict(pred_input_func)
my_pred = list(predictions) # a list of classes with all the statistical info
```
* we will now create a DNN classifier using estimators API `dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feature_cols,n_classes=2)` we provide a list with the layers specifying the neurons in each layer
* if we try to train our DNN model like simple estimator models passing input functions we get an error
* this is becauses DNN expects categorical columns as embedded cols 
* we show how to create an embedded col `embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)` the dimension is the num of categories
* we reset our feat columns `feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,embedded_group_column, age_buckets]`
* we create our new input function `input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)`
* we recreate our model passing in the new feat columns `dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feature_cols,n_classes=2)`
* we train it `dnn_model.train(input_fn=input_func, steps=1000)`
* we create an evaluate input function with the test data `eval_input_func = tf.estimator.inputs.pandas_input_fn( x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)`
* we do the actual evaluation `dnn_model.evaluate(eval_input_func)` our DNN doesnt help  alot as accuracy is the same. increasing steps doesnt help. we increase neurons ansd layers. no luck. so our dataset has limitations we should look for a better dataset

### Lecture 35 - TF Regression Exercise

* we will create amodel to predict housing prices using the tf.estimator API
* its a cali sensus dataset. each sample is a block of 1450 pople living close to each other. data is cleaned up
* the rule of thumb for DNNs is to staart with 3 layers. each layers should have neurons equalt in number to the features
* a good metric is below 100k in rmse , we got 85k rmse!! eith 20k steps and 4*6 layers

### Lecture 37 - TF Classification Exercise

* again cali census data. 
* we will try to predict in which income class an individual belongs to (< or > 50k $)
* we should conver pandas column strings to 0 or 1 
* use hashbucket for categorical columns,
* batch size of 100
* we ue linearclassifier (if we use DNN ae need embeding categorical columns)
* > 5000 steps

### Lecture 39 - Saving and Restoring Models

* once we have our tensorflow model ready the easiest way to save it is to use `saver = tf.train.Saver()` we can name it whatever we like. this creates the saver object
* the actual act of saving the model happens in the session . after we run our model and get results. we use `saver.save(sess,'path/filename.ckpt')` ckpt stands for checkpoint
* to restore it we start another session. then we need to `saver.restore(sess,'path/filename.ckpt')`
* now the session object comes from our checkpoint
* we can use it to run our stored model and get results 
* usually train data re used in the session that gets stored and the test data are used in the session where the model is retrieved

## Section 7 - Convolutional Neural Networks

### Lecture 40 - Intro to CNN Section

* in this section
	* will review NNs
	* go into new theory topics
	* see the famous MNIST dataset
	* solve MNIST with normal NN
	* learn about CNN
	* solve MNIST with CNN
	* CNN project to exercise

### Lecture 41 - Review of Neural Networks

* we ve seen single neurons.
* how to perform calcualtions in a neuron (w * x + b) = z, a = σ(z) #activation function
* activation functions
	* perceptrons
	* sigmoid
	* tanh
	* relu
* neurons connect to form a network
* a network has 
	* input layer
	* hidden layers
	* output layer
* more layers -> more abstraction (image)
* to learn we need a measuremnt of error
* we use a cost/loss function
	* quadratic
	* cross-entropy
* we minimize the error by choosing the correct weight and bias. how?
	* using gradient descent to find optimal vals
* we backpropagate the gradient descent through multiple layers from the output layer to the input layer
* there are also dense layers (fully connect to all neurons in next layer) and softmax layers

### Lecture 42 - New Theory Topics

* Up to now we always choose random values for our weights at initialization. some alternatives are:
* Initialize Weights to Zero:
	* we lose randomness
	* our dnn gets subjective
	* nota great choice
* Chose Random Distribution near Zero:
	* not optimal
	* distortion of activation functions
* Xavier (Glorot) Initialization:
	* 2 flavors: unifrom or normal distribution
	* what it is? we draw weights from a distribution with 0 mean and a specific variance *Var(W)=1/n[in]*
	* w is the initialization distribution for the neuron in question
	* n[in] is the number of neurons feeding into it
	* Xavier Initialization: Y = W1X1+W2X2+...+WnXn # linear neuron
	* Variance Var(WiXi) = E[Xi]^2Var(Wi)+E[Wi]^2Var(Xi)+Var(Wi)Var(ii) 
	* If our weights and inputs have a mean of 0 we get Var(WiXi)= Var(Wi)Var(Xi)
	* If Wi and Xi are indipendent and typicaly distributed (IID) the variance of the oputput is
	* Var(Y) = Var(W1X1+W2X2+...+WiXi) = nVar(Wi)Var(Xi) so variance of output is equal to variance of input multiplied by n the vairance of weight
	* If we want the variance of input and output to be the same The varianve of weight is Var(Wi)=1/n=1/n[in] This is the Xavier Intialization formula that is widely used. 
	* The original paper Xavier initialization formula is Var(Wi) = =2/(n[in] + n[out])
* Gradient Descent has 3 components
	* Learining Rate: Defines the step size during radient descent (small step => slow descent takes long time). large step might overshoot => never converge
	* Batch size: batches allow us to use stochastic gradient descent, smaller=>less representative of data, larger=>longer training time
* Second-Order behavior of the gradient descent allows us to adjust our learning rate based off the rate of descent (start with large steps slow down as we minimize error)
	* AdaGrad, RMSProp, Adam
	* Second order behaviour allows us to start with larger steps and then eventually go to smaller step sizes
	* Adam allows this change to happen automatically
* Unstable/ Vanishing Gradients
	* as we increase the number of layers in a network, the layers towards the input will be affected less by the error calculation occuring at the output as we go backwards through the network
	* initialization and normalization will help us mitigate these issues
	* we'll discuss vanishing gradients again in more detail when discussing Recurrent Neural Networks
* Overfitting vs Underfitting a Model (see PythonDSMLBootcamp)
	* underfitting is when we dont have good fit to our train data. so we have big error on train and test data
	* overfitting is when we fit too much on train data. we have very small error on train data but large error on test data
	* we need a balance
	* with potentially hundreds of params in a deep learning neural network the possibility of overfitting is high
	* there are ways to mitigate the issue (see below)
* L1/l2 Regularization 
	* Adds a penalty for larger weights in the model. 
	* Is not unique to neural networks. 
	* With this we dont let a strong feat overruling the model
* Dropout
	* Unique to neural networks
	* remove neurons during training randomly
	* network do not over rely on any particular neuron (mitigate overfit)
* Expanding Data
	* Artificially expand data by adding noise, tilting images, adding low white noise to sound data etc...
	* avoid overfit
* Theory to be covered: pooling layers, convolutional layers. We will cover those when we build CNNs

### Lecture 44 - MNIST Data Overview

* a classic dataset in Deep Learning
* MNIST us eesy to access with tensorflow, TF has
	* 55000 training images
	* 10000 test images
	* 5000 validation images
* MNIST contains hand written single digits from 0 to 9
* a single digit image can be represented as an array of 28*28 pixels. each pixel has a grayscale val from 0 to 1.
* we can flatten this array to an 1-D vector of 784 numbers. either (784,1) or (1,784) is fine as long as the dimensions are consistent
* flattening out the image ends up removing some of the 2-D info like the relationship of a pixel to its neighboring pixels
* we ll ignore this now. CNNs take onto account the relationship of a pixel to its neighbors. we will see then hwo to use it
* we can think of the entire group of the 55000 train images as a tensor (aka an n-dimensional array) of size 784by 55000
* for the labels we will use one hot encoding: instead of having labels like 'One'.'Two' etc. we will have a single array for each image. The label is represented based off the index position in the label array. the corresponding label will be a 1 at athe index location and zero everywhere else. e.g label '4' => [0,0,0,0,1,0,0,0,0,0]
* So the labels of the data end up being a large 2d array (10,55000)

### Lecture 45 - MNIST Basic Approach Part One

* have done it in PythonDSMLBootcamp...
* Before we dive into using CNN on the MNIST dataset we ll use a more bvasic Softmax Regression Approach
* We ll go over this method (similar to what we have done so far in previous sections)
* A softmax regression returns a list of values between0 and 1 that ad dup to one.
* we can use this as a list of probabilities *σ(z)[j] = e^z[j]/Σ[k=1][K]e^z[k] for j=1,...,K* if we have alist of 10 potential labels (0-9) we will get 10 probabilites . if we adde thm up we get 1
* we will use the softmax function as an activation function
* *z[i] = Σ[j](W[i,j]x[j])+b[i]* it starts like the sigmoid calculating z in the standard way
* *y=softmax(z)[i]=exp(z[i])/Σ[j]exp(z[j])* we pass the z to the softmax ( exponential of this output divided by the sum of the exponents of all output neurons)
* we impleemnt  this in jupyter with tensorflow
* we import tensorflow
* we get the data importing it from tf
```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
```
* mnist is a specialized tensorflow dataset `type(mnist)`
* we can see an image array `mnist.train.images`
* we can see the num of train or test  samples `mnist.train.num_examples` `mnist.test.num_examples`
* we can visualize the data , we check the flattened array shape `mnist.train.images[1].shape` 
* we import matplotlib and plot it after resaping it to a 2d matrix `plt.imshow(mnist.train.images[1].reshape(28,28))`
* image is already normzalized .min() is 0 and .max() is 1

### Lecture 46 - MNIST Basic Approach Part Two

* we start building our model: placeholders, variable, graph operations, loss func, optimizer, create session
* set the placeholders `x = tf.placeholder(tf.float32,shape=[None,784])`
* set the variables (Weights, bias)
```
W = tf.Variable(tf.zeros([784,10])) # not good choice but helps keep things simple
b = tf.Variable(tf.zeros([10]))
```
* we create our graph operations `y = tf.matmul(x,W) + b`
* we create our loss function (first we create our labels placeholder). we use cross entropy with softmax
```
y_true = tf.placeholder(tf.float32,shape=[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
```
* we create our optimizer 
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
```
* when we create a session we need to initialize all variables. we do the train in 1000 steps. mnist has build in method to feed batches in our session. we use tf.equal to check equality between pred and actual val. what we get is an array of booleans. we want to convert it to 0 and 1.we use tf built in cast method. to get the average we use reduce_mean
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for step in range(1000):
		batch_x, batch_y = mnist.train.next_batch(100)
		sees.run(train,feed_dict={x:batch_x,y_true:batch_y})
	# Evaluate the model
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1)) # it will return the index position with the highest probability (because of softmax)
	# [True,False,True...] --> [1,0,1..]
	acc =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
```

### Lecture 47 - CNN Theory Part One

* we solved the MNIST task with avery simple linear approach
* a much better approach is to use CNNs
* just like the simple perceptron CNNs also have their origins in biology
* Hubel and Wiesel studied the structure of the visual cortex in mammals winning the nobel Prize in 1981
* Hubel and Wiesel research revealed that neurons in the visual cortex had a small local receptive field. these neurons are only looking at a local subsection of the entire image the person is viewing. these subsections later overlap to create a large image and visual field
* these neurons in the visual cortex are only activated when they detect certain things (e.g a horizontal line or a circle)
* This idea the inspired an ANN architecture that would become CNN
* CNN was famously implemented int eh 1998 paper by Yann LeCun et al.
* The LeNet-5 architecture was first used to classify the MNIST dataset
* when we learn about CNNs we often see a diagram where a NbyN array (image) is first processed with feature extraction and the output of feat extraction is passed in tehe calssification NN.
* feature extraction is a series of convolutions and subsamplings.
* e.g NxN input -> 5x5 convolution => C1 = 6 (N-4)x(N-4) feature maps -> 2x2 subsanpling => S1 =  6 (N-4)/2x(N-4)/2 feature maps -> 5x5 convolution => C2 = 16 (N-4)/2 -4 x (N-4)/2 -4 feauture maps -> 2x2 subsamnpling => S2 = 16 ((N-4)/2 -4)/2 x ((N-4)/2 -4)/2 feature maps. the  classification NN is fully connected
* The new topics are Convolutions and Subsampling or Pooling
* To build CNNs we will learn about
	* Tensors
	* DNN vs CNN
	* Convolutions and Filters
	* Padding
	* Pooling Layers
	* Review Dropout
* Tensors are N-Dimensional Arrays we build up to:
	* Scalar e.g 3 (individual digits)
	* Vector e.g [3,4,5] (1-D arrays)
	* Matrix e.g [[3,4],[5,6],[7,8]] (2-D arrays)
	* Tensors (High dimensioanl arrays)
* Tensors make it very convenient to feed in sets of images into our model - (I,H,W,C) or 4D tensor
	* I: Images 
	* H: Height of Image in Pixels
	* W: Width of Image in Pixels
	* C: Color Channels. 1-Grayscale, 3-RGB
* Now lets explore the difference between a DenselyconnectedNN and a ConvolutionalNN
* We have already created DNNs using the tf.estimator API
* A DenselyConnectedLayer is a layer where every neuron connects to all neurons in the next layer
* In a Convolutional Layer, each unit (neuron) is connected to a smaller number of nearby units in next layer (loacal receptive fielrs in visual cortex)
* Why use a CNN instead of a DenseNN?
	* the MNIST dataset has 28by28 pixels (784)
	* most images are at least 256by256 or larger >56k pixels total
	* this leads to too many parameters, unscalable to new images if we use DNN
	* CNNs hav also a major advantage for image processing as pixels nearby to each other are much more correlated for image detection
* Each CNN layer looks at an increasingly larger part of the image
* Having units only connected to nearby units aids in invariance
* CBB also helps with regularization limiting the search of weights to the size of the convolution
* we ll see how a CNN relates to image recognition
* we start with the input layer. the image itself
* convolutional layers are only connected to pixels in their respective fields
* a possible issue arises for edge neurons. there may not be an inpute there for them.
* we can fix it adding a padding of zeros around the image
* we ll walk through 1D convolution in more detail. then we ll expand this idea to 2D Convolution.
* we ll revisit our DNN and convert it to CNN. we use a simple 1D example of limited neurons and layers. we convert 1st layer to a convolution layer. 2 nneighbor neurons from input layer output to one output neuron of next layer (1D convolution)
* we can treat the weights as a filter
	* y = w1x1+w2x2
	* if(w1,w2) = (1,-1) (arbitrary weights) then y = x1-x2
	* y max out when (x1,x2) = (1,0)
* this is an edge detection filter. as edges in an image appear as large differences in darkness from pixel to pixel
* we now have a set of weights tht can act as a filter for edge detection
* we can then expand this idea to multiple filters
* A filter with th efollowing characteristics: num of filters =1 , filter size =  2 (2 neurons involved), stride = 2 (from neuron subsection to next distance of 2) => every two neuros from input layer reduces to one in next layer, while weights  are aplied to both inputs at the same time (stride=2). or we call it moving up 2 neurons at athe time (stride=2)
* A 1-D convolution with filters=1,filter size=2,stride=1 involves 2 layers, 1 filter and moves up 1 neuron at a time
* we can  ad zero padding to include more edge pixels
* A 1-D convolution with filters=2,filter size=2, stride=1 goes from one input layer to two output layers that appear stacked on the z axis. is like doing a fork on the neural network. all other chars like filter size or stride apply like normal
* we can use multiple filters where each one detects a different feature
* representation gets messy as we draw multiple lines all over. for simplicity we gbegin to describe and visualize these sets of neuron as blocks 
	* for our example our input layes is depicted as 2-d recctanle  sized 1byL (1D of L neurons) 
	* the next layer is depicted as a 3D wall (#of filters by # of neurons by 1D)
* We ll now expand these concepts to 2-D Convolution since we ll be mainly dealing with images
* the block represantation serves us well
	* Input layer becomes 3d rectangle with 2 dimensions (H x W of image)
	* THe output layer becomes a #D block with 3 dimensions (# of filters x # of units W x # of units H )
	* subsections are also easy to depict
	* If our image is colored (multiple color values) we add one more dimension for color
* Filters are commonly visualized  with grids in image processing
	* say we have a 4x4 array (image) where outer pixels are 1 (black) and inner pixels -1 (white)
	* we add padding of o for edge cases maing it 5x5
	* we apply a 3x3 filter  on a topleft subsection => [[0,0,0],[0,1,1],[0,1,-1]]
	* we multiply it by filter weigths (0 for padding 1 otherwhise) output grid is the same
	* we sum the outputs => 2 output of convoluted image
* much like a 2d perceptron
* How *Stride* distance works? its the step we take when selecting subsections of (filtersize) in terms of pixels or the distance from subsection to subsection
* A cool website of CNNs in action [setosa](http://setosa.io/image-kernels/) where we can see filters applied on images

### Lecture 48 - CNN Theory Part Two

* We now go through the subsampling or pooling section of a CNN.
* now that we understand the convoluntional layers we ll discuss pooling layers
* pooling layers will subsample the input image, which reduces the memory use and computer load as well as reducing the number of parameters
* we imagine having alayer of pixels in out input image (6x6)
* as in our MNIST dataset. each pixel has a value repres
* pooling works by creating a pool of pixels (kernel) e.g 2x2 and evaluate the maximum value among them 
* only max value maks it to the next layer
* then we move over by a stride (e.g 2 pixels) and repeat
* this pooling layer will end up removing a lot of information. even a small pooling kernel of 2x2 with a stride of 2 will remove 75% of the input data
* another common technique  in CNNs is called "Dropout"
* "Droptout" can be thought of as a form of regularization to help prevent overfitting
* During training units are randomly dropped along with their connections
* THis helps prevent units from "co-adapting" too much
* Famous CNN Architectures
	* LeNet-5 by Yann LeCun
	* AlexNet by Alex Krizhevsky et al.
	* GoogleNet by Szegedy at Google Research
	* ResNet by Kaiming He et al.
* Advanced CNNs are computational expensive so they need to be distributed along multiple GPUs

### Lecture 49 - CNN MNIST Code Along: Part One

* we import tensorflow `import tensorflow as tf`
* we import mnist data fom tf (extract and import)
```
from tf.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
```
* we create a set of helper functions(initlalize weights, intialize bias, take a tensor and a filter and return a 2D convolution, pooling helper functionnat)
* initialize weights
```
def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init_random_dist)
```
* initialize bias vals
```
def init_bias(shape):
	init_bias_vals = tf.constant(0.1,shape=shape)
	return tf.Variable(init_bias_vals)
```
* create 2Dconvolution. there is a builtin tensorflow func that builds a 2dconvolution, taking an input tensor and an input kernel (filter tensor). our function will be a wrapper func setting just strides in any direction and padding to zeros using 'SAME' keyword
```
def conv2d(x,W):
	# input tensor x --> [batch,H,W,Channels]
	# kernel W --> [filter H, filter W, Channels IN, Channels OUT]
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
```
* for subsampling or pooling we use again tensorflow conveninece function with a wrapper like before. size is the size of the kernel window. i dont care about batch and color so i put 1 in these indexes
```
def max_pool_2by2(x):
	# input x --> [batch,h,w,c]
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides[1,2,2,1],padding='SAME')
```
* we are going to create 2 functions that will create the layers
* for the convolutional layer
```
def convolutional_layer(input_x,shape):
	W = init_weights(shape)
	b = init_bias([shape[3]])   
def normal_full_layer(input_layer,size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size,size])
	b = init_bias([size])
	return tf.matmul(input_layer,W) + b
```
* the most difficult part is to keep track of the dimensions
* we create our placeholders
```
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
```
* we want to reshape the flattened out array into an image again, 1 is for the 1 greyscale channel
```
x_image = tf.reshape(x,[-1,28,28,1])
```
* we start building the layers foir the network, we provide the shape of weight tensor (5x5 convolutional layer, 32 feats for every 5by5 patch) 1 is for input channels (grayscale),332 are the feats or output channel
```
convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)
```
* second layer group *32 input channel = the output feats from prev one. to make the flat image we use 7x7x64 ,7x7 as our image was pooled 2 times by 2 so reduced to 1/4 of 28. 64 are the feats
```
convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
```
* we do the dropout to improve results (in fully networked network). we use a holding probability placeholder to pass it as a param 
```
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep+prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
```
* we have all the layers in place. we now need our cost function and optimizer

### Lecture 50 - CNN MNIST Code Along: Part Two

* loss function (cross_entropy)
```
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_withlogits(labels=y_true,logits=y_pred))
```
* we implement our Dm optimizer
```
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
```
* we init our vars
```
init = tf.global_variables_initializer()
```
* we define our sessions
```
steps = 5000

with tf.session() as sess:

	sess.run(init)

	for i in range(steps):
		batch_x, batch_y = mnist.train.next_batch(50)
		sess.run(train,feed_dict={x=batch_x,y_true:batch_y,hold_prob:0.5})

		if i%100 == 0:

			print("ON STEP: {}".format(i))
			print("ACCURACY: ")
			matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
			acc = tf.reduce_mean(tf.cast(matches,tf.float32))
			print(sess.run(acc,feed_dict={x.mnist.test.images,y_true:mnis.test.labels,hold_prob:1.0}))
			print('\n')
```

### Lecture 52 - Introduction to CNN Project

* optional exercise to classify the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset
* main challenge is dealing with data amd creating tensor batches and sizing
* with MNIST batching was done automatically. now we need to do it
* data is batched in folders

### Lecture 53 - CNN Project Exercise Solution: Part One

## Section 8 - Recurrent Neural Networks

### Lecture 55 - RNN Theory

* we ve used neural networks to solve classification and regression problems
* we still havent seen how neural networks deal with sequential info (time series)
* We need to learn about RNN to solve such problems
	* RNN Theory
	* Basic Manual RNN
	* Vanishing Gradients
	* LSTM and GRU units
	* Time series with RNN
	* TimesSeries Exercise
	* Word2Vec
* Example of Sequential Data
	* time series data (sales)
	* sentences
	* audio
	* car trajectories
	* music
* lets imagine a sequence [1,2,3,4,5,6]. are we able to predict a similar sequence shifted one time step into the future? => [2,3,4,5,6,7]
* We can do this by using a recurent nerural network (RNN)
* a Normal Neuron in a Feed Forward Network is like an logic unit (FPGA): gets inputs, aggregates them, produces an output based on an activation function result
* A recurrent Neuron sends back the output back to itself. (mem unit)
* If we unroll it over time. it looks like a daisy chain of cascading neurons (its the same neuron at different moments) with a feedback capability
* cells that are a function of inputs from previous time steps are known as memory cells
* RNNs are flexible in their inputs and outputs for both sequences and single vector values
* we can build layers of Recurent neurons (instead of unrolling neurons over time , we unroll layers over time)we feedback  th output of the layer and add it to the inputs
* unrolling a neuron or a layer shows its status over time. and the use of the output at a given moment to build together with the other inputs to to build the output of the next moment
* We are building memory in this fashion as we use historical info to guide decisions
* RNNs are very flexible in their inputs and outputs
	* sequence input to sequence output (sequence shifted x steps into the future)
	* sequence input to vector output (sequence of words as input and output like the class of the sentence (possitive,negative) as single vector)
	* vector to sequence (pass an image get words describing this image, auto captions
* We will explore how to build a simple RNN model in tensorflow manually
* next we will see how to use TFs builtin RNN classes

### Lecture 56 - Manual Creation of RNN

* we ll create a 3 neuron RNN layer with TF
* the main idea is to focus on the input format of the data
* the layer we we wil build
	* will have one input X that goes to the 3 neurons. the 3 neuron  layer produces one output Y. the output is fed back to the layer (all neurons). weigths will be applied both to input X and fedback output Y
* we will start by running the RNN for 2 batches of data. t=0 and t=1
* each RNN has 2 sets of weights: Wx for input weights on X, Wy for weights on output of original X
* our data example: 
	* 5 timestamps t=0,t=1,t=2,t=3,t=4
	* 1st sample [The, brown, fox, is,quick]
	* 2nd sample  [The,red,fox,jumped,high]
	* words_in_dataset[0] = [The,The]
	* words_in_dataset[1] = [brown,red]
	* words_in_dataset[2] = [fox,fox]
	* words_in_dataset[3] = [is,jumped]
	* words_in_dataset[4] = [quick,high]
	* num_batches = 5, batch_size = 2, time_steps = 5 
* we feed based on timestamp (all samples)
* we implement our simple example (with numerical samples) in jupyter
* we import tensorflow numpy and matplotlib
* we set our model constants `num_inputs = 2` `num_neurons=3` inputs are 2 one the x and one the feedback
* we create a placeholder for each timestamp
```
x0 = tf.placeholder(tf.float32,[None,num_inputs])
x1 = tf.placeholder(tf.float32,[None,num_inputs])
```
* we create our variables (weights)
```
Wx = tf.Variables(tf.random_normal(shape=[num_inputs,num_neurons]))
Wy = tf.Variables(tf.random_normal(shape=[num_inputs,num_neurons]))
b = tf.Variable(tf.zeros[1,num_neurons])
```
* we create our graphs (activation function is tanh)
```
y0 = tf.tanh(tf.matmul(x0,Wx)+b)
```
* up to now all is like simple NNs. we now add recurrency coding out a simple unrollern RNN
```
y1 = tf.tanh(tf.matmul(y0,Wy)+tf.matmul(x1,Wx)+b)
```
* we initialize vars `init = tf.global_variables_initializer()`
* we create our dummy data for timestamp 0 and 1
```
x0_batch = np.array([[0,1],[2,3],[4,5]])
x1_batch = np.array([[100,101],[102,103],[104,105]])
```
* we run our sesion
```
with tf.Session() as sess:
	sess.run(init)
	y0_output_vals, y1_output_vals = sess.run([y0,1],feed_dict={x0:x0_batch,x1:x1_batch})
```
* we see our y1 and y0 results.
* our code doesnt scale well for larger timeseries with more steps as it will involve too much coding
* tf has an api for this

### Lecture 57 - Vanishing Gradients

* backpropagation goes backwards from the output to the input layer propagating the error gradient
* for deeper networks, issues can arise from backpropagation. Vanishing and exploding gradients
* as we go back to the "lower" layers or "front" layers closer to the input gradients often get smaller, eventually causing wights to never change at lower laevels
* the opposite may also occur. eg for actvation functions that use derivatives that on larger vals. gradients explode on the way back causing issues
* We ll see why this issue occurs and how we fix it.
* In next lecture we ll see how these issues affect RNN and how to use LSTM and GRU to fix them
* The vanishing gradients happen because of the activation function choice. e.g sigmoid goes from 0 to 1 in an s-curve
* backpropagation computes the gradients using the chain rule. so if we have a very large input (positive or negative) the slope or gradient is practically 0
* the chain rule of bavckpropagation computing the gradients has the effect of multiplying n of these very small numbers (gradients) to compute to compute gradients of lower or front layers (further towards the input) so gradient is going to decrease exponenentialy. so front layers  train very very slowly
* to solve it we attempt to use different activation methods (like RELU). RELU does not saturate positive values. but RELY for engatives outputs 0 so there is an issue
* "Leaky" RELU has anegative linear slope for engative input nums
* "Exponential Linear Unit (ELU)" also tries to solve these issues
* Another solution is to perform batch normalization, where our model will normalize each batch using the batch mean and standard deviation
* also "gradient clipping" where gradients are cut off beafore reaching a predermined limit (e.g cu off grtadients between -1 and 1) also mitigates the issue
* RNN for time series present tehir own gradient challenges. we ll explore special neuron units atha help fix the issues

### Lecture 58 - LSTM and GRU Theory

* LSTM = Long Short Term Memory, GRU = Gated Recurrent Units
* many of the solutions previously presented for vanishing gradients can be applied on RNNs (change activation function, batch normalizations etc...)
* Because of the length of the time series input, they can slow down training
* a possible solution is to shorten the time steps used for predictions. this makes the model worse at predicting longer trends
* another issue RNN face is that after a while the network will begin to forget the first inputs, as information is lost at each step going through the RNN
* we need some sort of 'long-term memory' for our networks
* The LSTM cell was created (1997) to address these RNN issues
* LSTM cell is more complex than a simple recurrent neuron
	* it still has as inputs xt and ht-1 (previous timestep neuron output or 'HIDDEN STATE') but get one more input ct-1 (previous timestep cell state)
	* it outputs ht (cell 'hideen state') and ct (cell state)
	* 1st step (forget gate layer): f[t] = σ(W[f]dot[h[t-1],x[t]] + b[f]). in this step we decide what informationw e are going to forget from the cell state. we pass ht-1 and xt in a sigmoid function. its going to output a num between 0 and 1. 1 means keep Ct-1 and 0 means forget it
	* 2nd step (decide what new information to store in the cell state): a) i[t] = σ(W[ι]dot[h[t-1],x[t]]+b[i]) b) ~C[t] = tanh(W[C]dot[h[t-1],x[t]] + b[C]). It is composed by 2 layers a) the sigmoid layer (input gate layer) b) the hyperbolic tanh layer which produces a vector of new candidate values that could be added to the state. 
	* 2nd step a and b are multiplied to update the cell state (through addition) and 1st step is multiplied on cell state to produce the new state `C[t] = f[t]*C[t-1]+i[t]*~C[t]`
	* 3rd step (decision on what to output as ht based on a filtered version of cell state): a) produce a filter value based on sigmoid on previous output o[t]=σ(W[o]dot[h[t-1],x[t]]+b[o]) b) use this filter with the hyperbolic tanh of cell state `h[t] = o[t]*tanh(C[t])`
* There are variants of LSTm e.g with 'peepholes' that adds peep holes to all gates aka use also cell state in each sigmoid function
* Another LSTM variant is GRU (gated recurrent unit) (2014). it simplifies things. combines forget and input gates toa single update gate itmerges cell state and hidden state (ht)
* DepthGRU (DGRU) in 2015
* Tensorflow comes with these neuron models built-into a nice API making it easy to swap them out
* Up next we ll explore this TF RNN API for Timeseries prediction and generation

### Lecture 59 - Introduction to RNN with TensorFlow API

* we ll use TF build in tf.nn function API to solve sequence problems
* our original simple sequnece problem involved  [1,2,3,4,5,6] as input asking us to predinct the sequence shifted one step forward => [2,3,4,5,6,7]
* say we have a time series that looks like `[0,0.84,0.91,0.14,-0.75,-0.96,-0.28] its actually just sin(x) where nest time step is [0.84,0.91,0.14,-0.75,-0.96,-0.28,0.65]
* we ll start creating a RNN that attempts to predict atimeseries shifted over 1 unit into the future
* then we will attempt to generate new sequences with a seed series
* we ll create simple class to generate sin(x) and also grab random batches of sin(x)
* then the trained model will be given a time series and attempts to predict a time series shifted one time step ahead
* afterwards we ll use the same model to generate much longer time series given a seed series. predict mush further into future time steps

### Lecture 60 - RNN with TensorFlow: Part One

* we import numpy, matplotlib and tensorflow
* we create a python class that will initialize our data and send batched of data back based on a generator method (np.sin)
* we add a convenience function to return the y_true values based on input series (sin())
* we add a method to generate batches of these data
* formating the data is important. we did this in our manual recurrnet network
```
class TimeSeriesData():
	
	def __init__(self,num_points,xmin,xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin)/num_points
		self.x_data = np.linspace(xmin,xmax,num_points)
		self.y_true = np.sin(self.x_data)

	def ret_true(self,x_series):
		return np.sin(x_series)

	def next_batch(self,batch_size,steps,return_batch_ts=False):
		
		# Grab a random starting point for each batch of data
		rand_start = np.random.rand(batch_size,1)
		
		# Convert to be on time series
		ts_start = rand_start * (self.xmax - self.xmin -(steps*self.resolution))
		
		# Create a batch time series on the x axis
		batch_ts =  ts_start + np.arange(0.0,steps+1) * self.resolution
		
		# Create the Y data for the time series x axis from previous step
		y_batch = np.sin(batch_ts)
		
		# Formatting for  RNN
		if return_batch_ts:
			# time series + x axis 
			return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
		else
			# return time series and timeseries shifted one step into the future
			return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)
```
* we create time series data `ts_data = TimeSeriesData(250,0,10)`
* we plot the timeseries data to the y values `plt.plot(ts_data.x_data,ts_data.y_true)`
* we want our random batches to have 30 steps in them `num_time_steps =30`
* we get our batch `y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)` ts is a 2d matrix so we ened to flatten it out
* we plot our batch `plt.plot(ts.flatten()[1:],y2.flatten(),'*')`
* we will overlay our batch on the general plot
```
plt.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')
plt.plot(ts.flatten()[1:],y2.flatten(),'*',label='Single Training Instance')
plt.legend()
plt.tight_layout()
```
* we create our training instance of data (x data) `train_inst = np.linspace(5,5+ts_data.resolution*(num_time_steps+1),num_time_steps+1)`
* we plot our train data and their y_true vals (timeseries + shifted one)
```
plt.title('A TRAINING INSTANCE')
plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label='instance')
plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='target')
plt.legend()
```
* we give the instance points to the RNN and get the target points

### Lecture 61 - RNN with TensorFlow: Part Two

* we will now create the RNN model
* we are going to build an rnn graph which is different fromt he default tf graph so we reset it `tf.reset_default_graph()`
* we create our model constants `num_inputs = 1` as we have only one feature in the time series 'x'
* we want 100 neurons per layer `num_neurons = 100`
* we have only one output y `num_outputs = 1`
* we define our learning rate `learning_rate = 0.0001`
* our num of iterations `num_train_iterations = 2000`
* the batch size will be 1 at a time `batch_size = 1`
* we  define our placeholders
```
X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])
```
* we will now create the RNN cell layer (many options available... lstm, gru etc). setting our num of usints to the num of neurons = 100 we will have 100 outputs. we want just 1. so we need a wrapper (output projection wrapper)
```
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_outputs)
```
* to get outputs and states of RNN cells `outputs, states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)` this creates an RNN based on the cell we defined and get the outputs and steps (unrolling of rnn)

* we use mean square error as loss function `loss = tf.reduce_mean(tf.square(outputs-y))`
* use ADAM optimizer 
```
optimizer = tf.train.AdamOptimizer(learning-rate=learning_rate)
train = optimizer.minimize(loss)
```
* we init globals `init = tf.global_variables_initializer()`
* when we run Tf on GPU sometimes all gpu resources are taken from tf and the machine crashes so if we run on gpu we need to limit resource allocation `gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)`
* we use saver to save our model for later use `saver = tf.train.Saver()`
* we now run our session passing gpu confifg option
```
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    for iteration in  range(num_train-iterations):

    	X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
    	sess.run(train,feed_dict={X:X_batch,y:y_batch})

    	if iteration % 100 == 0:

    		mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
    		print(iteration,"\tMSE: ".mse)

    saver.save(sess,"./rnn_time_series_model_codealong")
```
* with the model trained we will use it to predict a timeseries one step into the future (we restore our model and use it)
```
with tf.Session() as sess:
	saver.restore(sess,'./rnn_time_series_model_codealong')

	X_new = np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_inputs)))
	y_pred = sess.run(outputs, feed_dict={X:X_new})
```
* we plot our resutls
```
plt.title('Testing the model')

# Training Instance
plt.plot(train_inst[:-1],np.sin(train_inst[:-1]),'bo'markersize=15,alpha=0.5, label="Training Instance")
# Target to Predict (correct test vals)
plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")
# Model prediction
plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,label='predictions')
```
* initial our predictions are not good but get very accurate as we move on
* we swap cell types repreating the procedure to check performance (GRUCell) changing the learnign rate. GRUCell is better even from the beginning

### Lecture 63 - RNN with Tensorflow: Part Three

* we will now generate a brand new sequence further far in time
* we reuse our saved model. we seed it with a batch of zeros and see what happens. we have 30 zeros and then the generated values
```
with tf.Session() as sess:
	saver.restore(sess,'./rnn_time_series_model_codealong')

	# SEED ZEROS
	zer_seq_seed = [0.0 for i in range(num_time_steps)]

	for iteration in range(len(ts_data.x-data) - num_time_steps):

		X_batch = np.array(zero_seq_seed[-num_time_steps:].reshape(1,num_time_steps,1))
		y_pred = sess.run(outputs,feed_dict={X:X_batch})
		zero_seq_seed.append(y_pred[0,-1,0])
```
* we plot it
```
plt.plot(ts_data.x_data,zero_seq_seed,'b-')
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
```
* what we get is something that looks like sinusoid but not actual sinusoid
* we repeat passing actual sinusoid seed
```
with tf.Session() as sess:
	saver.restore(sess,'./rnn_time_series_model_codealong')

	# SEED with training instance
	training_instance = list(ts_data.y_true[:30])

	for iteration in range(len(ts_data.x-data) - num_time_steps):

		X_batch = np.array(training_instance[-num_time_steps:].reshape(1,num_time_steps,1))
		y_pred = sess.run(outputs,feed_dict={X:X_batch})
		training_instance.append(y_pred[0,-1,0])
```
* we plot again and get a sinusoidal plot
*Q:I don't understand the final shape of the X_batch in 3:18, why is (1,num_time_steps,1) ? what does it mean [ [ [......] ] ]  form? A: [[[]]] is used to indicate a tensor (basically an N-dimensional array), so we can have a dimension for batch, a dimension for a feature X, and a dimension for all the time steps. Basically think of it a single batch of a (n,1) array

### Lecture 64 - Time Series Exercise Overview

* our aim is to build an RNN that will sucessfully predict monthly milk production based of a real dataset
* its a old dataset but has a trend and a seasonality so its good for evaluating a model
* in time series there is no point in radom train test split
* we need to scale the data (fit only to train data)
* creating the batches is done.  to create the y_batch we need to cast the dataframe to a numpy array
* in the generative session my train set is the train seed except the last 12 months `  train_seed = list(train_scaled[-12:])`

### Lecture 66 - Quick note on Word2Vec

* These are Optional series of Lectures descibing how to implement Word2Vec with Tensorflow. It does embeddings in a vector space for individual words with tensorflow
* It is recommended we check out the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library if we are further intenrested in Word2Vec

### Lecture 67 - Word2Vec Theory

* now that we understand how to work with time series of data, we ll have a look in naother common series data source. words
* For example a sentence can be ['Hi','how','are','you']
* In classical NLP words are typically replaced by numbers indicating some frequency relationship to their documents e.g TFIDF 
* in doing this we lose information about the relationship between the words themselves
* NLP has 2 approaches: 
	* Count based: frequency of words in corpus (e.g TFIDF)
	* Predictive based: neighboring words are predicted based on a vector space (e.g Word2Vec)
* One of the NNs most famous cased in NLP: The Word2Vec model created by Mikolov et al.
* The goal of the Word2Vec model is to learn word embeddings by modeling each word as a vector in a n-dimensional space
* But why use word-embeddings? Audio and Images are Dense datasets. 
* when we take a count based approach to text data we end up with a sparce dataset. replacing words with numbers is not how brain works we lose the similarites between words (semantics)
* Word2Vecs creates vector spaced models that represent (embed) words in a continuous vector space
* with words represented as vectors we can perform vector mathematics on words (e.g check similarity 'cosine similarity', add/subtract vectors)
* at the start of training each embedding is random, but through backpropagation the model will adjust the value of each vector in the given number of dimensions
* More dimensions means more training, but also more 'information' per word
* similar words will find their vectors closer together
* even more impressively, the model may produce axes that represent concepts such as gender, verbs, singular vs plural e.g king - (man+woman) = queen or country-capital relations
* How Word2Vec creates these word embeddings and how it learns them from raw text
* Word2Vec comes in 2 flavors: 
	* Continuous bag of Words (CBOW) model: typically better for smaller data sets
	* Skip-Gram Model
* algorithmicaly both models are very similar except on the way they end up redicting target words
* CBOW model takes in source context words (the dog chews the) and then it attempts to find its prediction target => bone. good for shorter texts as it will smooth over a lot of the distributional info treating the entire context as one observation
* Skip-Gram Model does the inverse: it attempts to predice source context words from target words bone => the dog chews the. good for larger texts
* How CBOW trains the model? with a technique called Noise Contrastive Training
	* The dog chews the w=? => bone[target word] vs [book,car,house,sun...] noise words
	* words2vec does not need a full probabilistic model. we use a binary classification objectve (e.g logistic regression) to discriminate the real target words (Wt) from imaginary noise words (Wk) in the same context
	* We have a projection layer (the cat sits on the ...), a noie classifier to pair the target word vs the noise words. we pass this from anumber of hidden layers and get some embeddings
	* Target Word is predicted by maximizing J[NEG] = logQ[θ](D=1|w[t],h) _ k[n~P[noise]]E[logQ[θ](D=0|w[n],h)]
	* Q[θ](D=1|w[t],h) is binary logistic regression in the probability that the word wt is in the context h in the dataset D parametrized by θ
	* wn  are k contrastive words drown from noise distribution
	* we only draw k words not all our dataset words,  so its computationaly efficient (the loss function scales only to the noise words) not to all words in teh vocabulary. we can get reasonable results with a small k
	* the goal is to assign a high probability to correct words and low probability to noise words
	* once we have vectors for each word we can visualize relationships by reducing dimensions from 150 to 2 using t-distributed stochastic neighbor embedding
* we end up getting a plot of a cloud of words. points that are close to each other have siilarities

### Lecture 68 - Word2Vec Code Along: Part One

* tf documentation has an example implementation of word2vec (we ll cp that)
* its a manual and tedious process (better use gensim)
* we do the imports (cp them)
```
import collections
import math
import os
import errno
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf
```
* we will grab the data as a zip  from a url
* we set the url and target dir
```
data_dir = "word2vec_data/words"
data_url = 'http://mattmahoney.net/dc/text8.zip'
```
* we implement  a method (fetch word data) that unzips the data if we dont have them unzipped. or downloads it and unzip it if we dont have them
* we pass url and local dir
* check that firewall allows urlretrieve
* we unzip encode anbs split the data....
```
def fetch_words_data(url=data_url, words_data=data_dir):
    
    # Make the Dir if it does not exist
    os.makedirs(words_data, exist_ok=True)
    
    # Path to zip file 
    zip_path = os.path.join(words_data, "words.zip")
    
    # If the zip file isn't there, download it from the data url
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        
    # Now that the zip file is there, get the data from it
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    
    # Return a list of all the words in the data source.
    return data.decode("ascii").split()
```
* we use the method to get the data `words = fetch_words_data()`
* we check the length of our words list... `len(words` its 17Million!!!!!!!!!!!!!
* we get a slice of the words to visualize them `words[9000:9040]` they actually are part of cohesive text documents (probably a whole book)
* even though our words list contains sentences, we format it a little better for our understanding... still there is no punctuation.. its essentially a stream of words
```
for w in words[9000:9040]:
    print(w,end=' ')
```
* we ll now build a word count and test it in some small test data. we use the collections library. we import it `from collections import Counter`
* we make a dummy list `mylist = ['one','one','two']`
* we pass it to the Counter `Counter(mylist)` (fit it) it gets the occurencies of each word (counter). returns a dictionary
* and get the metrics `Counter(mylist).most_common(1)` returns a tuple
* we can now apply it to our word list  and use it to build a vocabulary of 50000 unique words
* we cast our vocab to a numpy array using list comprehention (ignoring the second couter param with tuple unpacking)
* we build a vocab from the words witha  dictionary comprehension on an enumeration object ([(0, 'the'),.....]). code is the index and word the word
* to build our data we use list comprehension on the dictionary, what we do is we make a list of the dictionary index for words in the word list
```
def create_counts(vocab_size=50000):
	vocab = [] + Counter(words).most_common(vocab_size)
	vocab = np.array([word for word,_ un vocab])
	dictionary = {word:code for code,word in enumerate(vocab)}
	data = np.array([dictionary.get(word,0) for word in words])
	return data,vocab
```
* we get our data and vocab `data,vocabulary = create_counts(vocab_size=vocab_size)`
* data has a size of 17MIllion and vocab 50000
* how they work ? `words[100]` => 'interpretations' , `data[100]` => 4186 (index in vocab) , `vocab[4186]` => 'interpretations'
* we now have to impelent a function to feed the model with batches of data (OMG!!!!!!)
```
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
```
* we set our model constants (embedding size is how many dimensions our embeding vector will have). more dimensions is more info and better results (but it costs more...)
* skip window (how many words to consider to the left or to the right). the bigger we set it the longer it will take to train
* num_skips how many times to reuse aninput to generate a label
```
batch_size = 128
embedding_size = 150
skip_window = 1
num_skips = 2
```
* we now have to pick a random validation set to sample neiarest neighbors `valid_size = 16` (a random set of words to evaluate similarity on)
* we ll pick samples from the head of the distribution. we are limiting the validation samples to words that have a low numeric id (the most frequent words)
```
valid_window=100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
```
* we get a number of negative examples to sample `num_sampled = 64`
* we set the learning rate of our model `learining_rate = 0.01`
* we set our vocab size `vocabulary_size = 50000`
* we now create our tf palceholders and constant ANd reset the graph
```
tf.reset_default_graph()
train_inputs = tf.placeholder(tf.int32,shape=[None]) #just ints, no actual feats just intdex
train_labels = tf.placeholder(tf.inst32,shape=[batch_size,1])
valic-dataset = tf.constant(valid_examples,dtype=tf.int32)
```
* we create the vars and a loss function (NCE)
* we randormly start our word embeddings
```
init_embeds = tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0)
embedding = tf.Variable(init_embeds)
embed = tf.nn.embedding_lookup(embeddings,train_inputs) # looks up ids in embedding tensors
```

### Lecture 69 - Word2Vec part Two

* we need to define our nCE loss function. we first set up the nce weights
```
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```
* we calculate loss
```
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))
```
* we write the optimizer (Adam)
```
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)
```
* Compute the cosine similarity between minibatch examples and all embeddings.
```
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
```
* we initialize globals `init = tf.global_variables_initializer()`
* he runs session on gpu so we adds config to limit mem use `gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)`
* session implementation
```
num_steps = 200001

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    average_loss = 0
    for step in range(num_steps):
         
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        _, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            # The average loss is an estimate of the loss over the last 1000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

       

    final_embeddings = normalized_embeddings.eval()
```
* we ll try to visualize the results using t distribution stochastic neighbors embedding which allows us to transform the final embeddinds in 2d for visualizing. we chech the shape of final embeddings `final_embeddings.shape`
and is (50000,150) so 50000 vectors of 150 dimensions
* we use TSNE from sklearn `from  sklearn.manifold import TSNE`
* we create an instance, n_components is the final dimensions, we use pca to initialize the process `tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)`
* we will plot only 500 of the words `plot_only = 500`
* we get the low dimension embeddings using the tsne. `low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])` for only yhe 500 first
* we are now ready to plot we get the labels (words from vocabulary) `labels = [vocabulary[i] for i in range(plot_only)]`
* we use a ready made plot function to do the plot (a scatterplot with labels for each dor)
```
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
```
* we do the plot `plot_with_labels(low_dim_embs, labels)`
* there is not much similarity as our training was weak with 5000 steps
* we load the results of 200k steps `final_embeddings = np.load('trained_embeddings_200k_steps.npy')` and repeat to do the plot

## Section 9 - Miscellaneous Topics

### Lecture 71 - Deep Nets with TensorFlow Abstractions API: Part One

* TensorFlow has many abstractions:
	* TF Learn
	* Keras
	* TF-Slim
	* Layers
	* Estimator API
	* and more...
* We lose control in return of ease of use and simplicity
* Many of these abstractions reside in TFs tf.contrib section
* Typically libraries get developed and accepted into contrib and then 'graduate' to be accepted as part of standard Tensorflow
* Its difficult to tell which TF abstraction are worth learning and which not
* The speed of development of TensorFlow has caused abstractions to come and go quickly
* We' ll focus on presenting the most common abstractions used: Estimator API, Keras (TF + Theano), Layers
* We' ll focus on understanding how to use these abstractions to build deep densely connected neural networks
* Using these abstractions makes it easy to stack layers on top of each other for simpler tasks
* we ll start by exploring the datasets we will use. we use sklearns wine dataset `from sklearn.datasets import load_wine` we make a dataset out of it `wine_data = load_wne()` it is an sklearn bunch file much like a dictionary `wine_data.keys()` => dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
* we check DESCR. its data about chemical analysis on samples from 3 types of wine from 3 wineries inm ITaly.
* we ll try to do a classification task on these data
* we build our dataset 
```
feat_data = wine_data['data']
labels = wine_data['target']
```
* we perform a train-test split
* we scale our data before passing them to the DNN abstractions

### Lecture 72 - Deep Nets with TensorFlow Abstractions API: Estimator API

* we recap what we have learned so far about estimator API
* we import tensorflow
* we import estimator `from tensorflow import estimator`
* we check xtrain shape `X_train.shape` => (124,13) 13 feats by 124 samples
* to use the estimator API we need to make feature columns. all feats are numeric vals so we can do it at once without labeling them first passing the shape of [13]. `feat_cols[tf.feature_column.numeric_column('x',shape=[13])]`
* we go straight to the model `deep_model = estimator.DNNClassifier(hidden_units=[13,13,13],feature_columns=feat_cols,n_classes=3,optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))`
* we create the input function `input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train,shuffle=True,batch_size=10,num_epochs=5)`
* we train our model `deep_model.train(input_fn=input_fn,steps=500)`
* we write downb the eval input function `input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)`
* we get the predictions `preds = list(deep_model.predict(input_fn=input_fn_eval))`
* we cast resultsin a list `predictions = [p['class_ids'][0] for p in preds]`
* we use sklearn metrics (classification report ) to evaluate results

### Lecture 73 - Deep Nets with TensorFlow Abstractions API: [Keras](https://keras.io/)

* keras is a python based high level neural network api able to run on top of TensorFlow, CNTK, or Theano
* if we want to use it for multiple frameworks we need to download it as a separate library
* we import tensorflow
* we import keras models  `from tensorflow.contrib.keras import models`
* we create a sequential model `dnn_keras_model = models.Sequential()`
* we import layers from keras `from tensorflow.contrib.keras import layers` there are a bunch of layer types available
* we choose a simple dense layer which we add to the model `dnn_keras_model.add(layers.Dense(units=13,input_dim=13,activation='relu'))` we specify the num of neurons per layer and inputs (feats)
* we add more layers. for subsequent layers (hidden) we dont have to specify inputs `dnn_keras_model.add(layers.Dense(units=13,activation='relu'))`
* we add a layer with softmax activation as output layer (units=3) like the output classes (hot encoded)
* we compile our model . to do it we import more tools from keras `from tensorflow.contrib.keras import losses,optimizers,metrics,activations`
* to see all our activation methods available we use `activations.+TAB` same for optimizers
* hot encoded classes are sparce matrices and categorical so for loss func we use *sparse_categorical_crossentropy*
* we compile our model `dnn_keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`
* wew train our model passing train data `dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)`
* to get the predictions from our model we use predict_classes `predictions = dnn_keras_model.predict_classes(scaled_x_test)`
* we import and print classification report from sklearn. we get 91% easy

### Lecture 74 - Deep Nets with TensorFlow Abstractions API: Layers

* currently Layers [API](https://www.tensorflow.org/api_docs/python/tf) is split between the tensorflow.contrib.layers and tensorflow.layers
* tf.contrib.layers used to be more complete than tf.layers
* at Tutotials => Layers there is a CNN example to solve MNIST classification problem
* we ll solve the wine problem with layers
* we import tensorflow
* we will hot encode our results with pandas using get_dummies`onehot_y_train = pd.get_dummies(y_train)`
* layers needs the data in numpy array format so we convert them `onehot_y_train = onehot_y_train.as_matrix()`
* we do the same for test labels `onehot_y_test = pd.get_dummies(y_test).as_matrix()`
* we set our model params
```
num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01
```
* we use contrib.layers lib to import fully connected model `from tensorflow.contrib.layers import fully_connected`
* layers API stands halfway between tf and keras giving more control, but simplifies layer building
* we define our placeholders
```
X = tf.placeholder(tf.float32,shape=[None,num_feat])
y_true = tf.placeholder(tf.float32,shape=[None,num_outputs])
```
* we set our activation func `actf = tf.nn.relu`
* we build our layers using layers api 
```
hidden1 = fully_connected(X,num_hidden1,activation_fn=actf)
hidden2 = fully_connected(hidden1,num_hidden2,activation_fn=actf)
output = fully_connected(hidden2,num_outputs)
```
* we use softmax for loss using pure tf `loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)`
* we set optimizer
```
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
```
* we init globals `init = tf.global_variables_initializer()`
* we set 1000 trainind_steps and run the session
```
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        sess.run(train,feed_dict={X:scaled_x_train,y_true: onehot_y_train })
    
    logits = output.eval(feed_dict={X:scaled_x_test})
    
    preds = tf.argmax(logits,axis=1)
    
    results = preds.eval()
	
```
* the output will give the probability for each hot encoded class. so argmax will choose the class with highest probability along columns
* we import sklearn classification report and print it `print(classification_report(results,y_test))`
* when we get perfect results we skew params to see that model is actually working (reduce learning_steps)

### Lecture 75 - [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)

* Tensorboard is a visualization tool built in tensorflow
* it allows visualize complex graphs inan organized manner
* we import tensorflow
* we ll build a simple graph session
```
a = tf.add(1,2)
b = tf.add(3,4)
c = tf.multiply(a,b)
```
* we run the session
```
with tf.Session() as sess:
    print(sess.run(c))
```
* we get 21. for a simple graph like this its easy to see how this value came from.
* for complex graphs is not that easy. visualization can assist our insight
* to use it in our session (at the start) we add a summary output to a file passsing the folder path and what to store (session graph) `writer = tf.summary.FileWriter('./myoutput',sess.graph)`
* after we finish our session operation we close the writer
```
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./myoutput',sess.graph)
    print(sess.run(c))
    writer.close()
```
* a tf events file is stored in our speced location. it has all the info for the graph
* we open up terminal to run tensorboard (we need to activatea conda environment where tensorflow is installed)
* we run it passing the folder with the event file `tensorboard --logdir='./myoutput'`
* tensorboard runs as a webservice at `http://<computername>:6006`. we open it in a browser
* OUR GRAPH IS ON SCREEN ready for analysis
* we see our graph has aautomatically assigned names. in big graphs is not very helpful so we should name the at our preference
* every tensorflow operation can be named. NO SPACES ALLOWED `a = tf.add(1,2,name='first_add')`
* our named graph is on screen... (the old one is still there we need to clear event files we dont want from folder)
* we can add scopes (organizational blocks) which are specked with `with tf.name_scope("OPERATION B"):` all operations that we write in this block are considered part of this namescope. the namescope in tensorboard graph appears as a box encaptulating the operations (if we click we see its contents)
```
with tf.name_scope("OPERATION_A"):
    a = tf.add(1,2,name='first_add')
    a1 = tf.add(100,200,name='a_add')
    a2 = tf.multiply(a,a1)
```
* we can have nested scopes (subscopes)
* tensorboard can do more than just visualizing graphs. we can visualize data
* we ll visualize a histogram (cp from dovumentation)
	* we make a placeholder
	* we make a normal distribution (with a shifting mean) based on the placeholder val
	* we record the histogram in a  file
	* we create asession and start recording
	* tf.summary.merge_all merges all summaries to aone file
	* we do aloop calculating summaries basedon a feed dictionary
* we lauch tensorboard (CAREFUL we have to be in same folder as the event fil ewhen we launch) `tensorboard --logdir='./'`
* in that way we can visualize our weights as they move during training
* there are full tensorboard tutorials available

## Section 10 - AutoEncoders

### Lecture 76 - Autoencoder Basics

* the autoencoder is a very simple neural network and will feel similar to a multi-layer perceptron model
* it is designed to reproduce its input in the output layer
* the key difference between an autoencoder and an MLP(MulitLayerPeerceptron) network is that the number of input neurons is equal to the number of the output neurons
* A typical autoencoder starts wide gets narrow in the middle and widens up towards the end (like a flume) (starts with input neurons that get reduced towards the middle and the increase towards the output layer that has same num of neurons to the input)
* to understand the autoencoder we simplify things  creating a network with a single hidden layer (input -> hidden -> output layer)
* its a feed forward network (like mLP) trained to reproduce its input in the output layer
* say we start with X inputs in the input layer. they get multiplied with weight W
* in the hidden layer a method is applied h(X) and a bias i c is added . this first part is called ENcoder and is represented mathematically h(x) = sigm(b+Wx) as the method applied is the sigmoid function
* the output of the hidden layer h(x) gets a weight Wout and goes to the output layer where another activation method is applied  and a bias is added. the second part is called Decoder as is represented mathematically dashX = sigm(c+Wout h(x))
* both transformations applied are linear 
* dash X is the autoencoder reproductions of the inputs X
* a common practice in autoencoders as we try to reproduce ourt inputs in the outputs is to use tied Weigts where Wout is actially Wt or the transposed matrix of W. bias terms are noit tied, they are separated
* the real trick happens in the hidden layer. its typically going to be smaller than the inputs (undercomplete autoencoder). hidden layer could be larger (overcomplete autoencoder)
* as inputs are reduced to the hidden layer. this layer is going to have an internal representation that tries to maintain all the information of the input with less neurons
* so we can use the hidden layer to extract meaningful features from the input or even do PCA with autoencoders
* whats the whole purpose of the autoencoder nens is to train the hidden layer to get a compressed representation of our input data
* we ll see later on stacked autoencoders with more hidden layers