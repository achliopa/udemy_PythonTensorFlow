# Udemy Course: Complete Guide to TensorFlow for Deep Learning with Python

* [Course Link](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/v4/overview)

## Section 1 - Introduction

### Lecture 1 - Introduction

* we will learn how to use Google's TensorFlow framework to prerform the latest techniques in the Deep Learning ecosystem
* we wll start with a crashcourse in the essential Python data science libraries (know them)
	* numpy
	* pandas
	* matplotlib
	* scikit learn
* we will then dive straight into Neural Networks (Perceptrons, Activation Functions, Back Propagation etc)
* Then we ll dive in Tensorflow
* A variety of networks is covered
	* Densely Connected Neural Networks (basic classification and regression tasks)
	* Convolutional Neural Networks (complex image classification tasks)
	* Recurrent Neural Networks + Long-Short Term Memory Units + Gated Recurrent units (analyze sequences of data like time series)
	* Word2Vec algorithm (word embeddings)
	* Auto Encoders (revised tasks)
* Generative Adversarial Networks 
* Reinforcement Learning with [OpenAI Gym](https://gym.openai.com/) to teach algorithms to paly video games

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

* we go to anaconda.com (anaconda is a high perfirmance distribution of a lot of data science packages)
* we download for ubuntu (python3) => follow installation instructions
* do not add anaconda to Path (unless we want anaconda to be our main distro for python). for this course it is ok to do so
* with anaconda installed on our machine it time to restore the env file
* we open the terminal
* cd to where the tfdl_env.yml file is placed
* then we run `conda env create -f tfdl_env.yml`
* we now activate the conda env `source activate tfdeeplearning`
* we run `hupyter notebook` and it should open a browser for us
* to check installation of tensorflow in jupoyter
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
* ML algos can find insights in data even if the y are not specifically instructed what to look for in data
* we define a set of rules not tell it what to do
* three Major types of ML algos
	* Supervised,
	* Unsupervised
	* Reinforced
* We ll also look into word embeddings with Word2Vec
* *Supervised Learning*:
	* uses *labeled* data to predict a label given some features
	* if a label is *continuous* it is called a regfression problem, if its *categorical* itr is a *classification* problem
	* it has the model train on historical data that is already labeled
	* once the model is trained it can be used on new data where only the feats are known, to attempt prediction
* If we dont have labels fo our data (only feats) we have no right answer to fit on. we can only look for patterns in the data and find structures AKA *Unsupervised Learning*:
	* *clustering* problems: e.g heights and weights fgor breeds of dogs. (no labels). we cluster the data into similar groups (clusters). it s up to the scientist to interpret the clusters (assign lables 'breeds'). clustering algos cluster points based on common feats
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
* once we are ready to deploy we use our holdout set to get a final metric (evaluation) on the performance after deployment. 
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
* A simple nuron model is known as perceptron (2 inputs + 1 output)
	* inputs can have feature values
	* a weight value (adjustable and different per input) is applied on each input
	* wights initialy start off as random
	* inputs are multiplied by weights
	* these results are passed to an activation function (body of neuron): there are many activation functions
	* we assume a simple activation function (if sum of weighted inputs >0 return 1, if negative return 0)
	* the output is the result of the activation function
	* a possible problem if when inputs are 0. weights dont have any effect and activation always outputs 0
	* we fix it adding a 3rd adjustable input: the bias (we set it initially to 1)
* Mathematically we can represent a perceptron as : Σ[i=0 -> i=n (Wi * xi)] + b

### Lecture 16 - Neural network Activation Functions

* we ve seen how a single perceptron behaves, now lets expand this concept to the idea of a neural network
* lets see how to connect many perceptrons together and how to represent this mathematically
* a multiple perceptrons lnetwork is a multilayered network with each layer containing an array of neurons. each layer feeds its output to the next layer as its inputs
* usually we have an input layer some hidden layers and an output layer
* Input Layers get the real values from the data
* Hidden Layers are in between the input and output layer, a network of 3 or more hidden layers is considered DeepNN (DNN)
* Output Layer is the final estimate of the output
* As we go forward through more layers the level of abstraction increases
* We ll now look into the activation function
* in our previous simple exampe the activation function was simple kind like a one step dunction (y 0 or 1)
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
	* a to represent neueron's prediction
* in terms of weights and bias
	* w * x + b = z
	* Pass z into activation function e.g sigmoid function σ(z)=a
* the first cost function we see is *Quadratic Cost* : C = Σ(y-a)^2 / n
* larger errors are more prominent due to squaring
* this calculation can cause a slowdown in our learning speed
* *Cross Entropy* C = (-1/n)Σ(y * ln(a)) + (1-y) * ln(1-a)
* this cost function allows for faster learning
* the larger the difference, the faster the neuron can learn
* We have some good options to start with for 2 aspects of DNN learnign. neurons and their activations functions and the cost function
* we are missing the learning part
* we have to figure out how we can use our neurons and the measurement of error (cost function) and then attempt to correct the prediction AKA learn
* we will see how we can do it with gradient descent

### Lecture 18 Grandient Descent Backpropagation

*Gradient Descent* is an optimization algorithm for finding the minimum of a function
* to find a local minimum we take steps proportinal to the negative of the gradient
* gradient descent in 1 dimension plot (cost in y axis, a weight in x axis)  is like an U curve
* gradient is the dF() derivative of function. we find it a nd see which way it goes in the negative direction. we follow the plot untill we reach the minimum C (bottom of curve)
* what we get is we find the weight that minimizes cost
* finding this minimum is simple for 1 dimension. but our ML cases will have more params, meaning we will need to use the built-in linear algebra that our Deep Learning lib will provide
* Using gradient descent we can figure out the best params for minimizing our cost. e.g finding the best values for the weights of the neuron inputs
* our problem to solve is how to quickly adjust the optimal params or eweights across our entire network
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
* we max out ^ * 8
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
* we also set a string literal `s="world` of type str and see its biult in methods
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
		_defualt_graph.variables.append(self)

class Graph():
	def __init__(self):
		self.operations = []
		self.placeholders = []
		self.variables = []

	set_as_default(self):
		global _default_graph
		_default_graph = self
```

* when we call `_defualt_graph.variables.append(self)` we append the current instance to the list of the graph
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

### Lecture 26 - Introduction to TensorFlow

* 