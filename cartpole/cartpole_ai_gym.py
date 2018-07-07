'''
In a  scenario where you can have various signals, which have some degree of predictive power,
and combine them for something with more predictive power than the sum of the parts. And neural networks are fully capable of doing that.
Here, the input layer is the obervation from the environment, which includes pole position and such. The output layer is just one of two actions: Left or Right.
'''
# the idea is to have an environment and an agent which moves randomly at the beginning.

import gym				# to create the environment
import random			# to allow the agent to move randomly
import numpy as np
import tflearn
from tflearn.layers.core import input_data ,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import statistics

LR = 1e-3
##env= gym.make('CartPole-v0')      # to make the environment from the gymai
env= gym.make('CartPole-v0').env        #----    to access more frames from the openai gym
env.reset()						  #resets the env and make its works from the beginning

goal_steps = 500
score_requirement = 50
initial_games = 10000			# we should not make this number too big as we may be using brute-force to solve the problem

'''
def some_random_games_first():    	# our goal is to learn from the random moves and we are not gonna go with all the games but with the ones having score equal or above 50
	# each of these is its own game
	for episode in range(10):
		env.reset()
		for t in range(goal_steps):
			# This will display the environment
			# Only display if you really want to see it.
			# Takes much longer to display it.
			env.render()			# deploys the environment

			action=env.action_space.sample()			#creates and a sample action in any environment
			# In this environment the action can be 0 or 1 ,i.e, either left or right

			observation,reward,done,info=env.step(action)	# this executes the env and returns us the observation of the env and the reward, if the env is over and other info
			# the reward is either 1 or 0 ( either balanced or not) and obesrvation is the pixel data(the array of the data from the game)[pole-position,cart-position]
			# info contains useful information for debugging
			#each time an agent chooses an action ,the environment returns an observation and a reward

			if done:
				break
#some_random_games_first()		## Each time we see the scene start over because the env was done and in our case we kept loosing
'''

def initial_population():
	 # [OBS, MOVES]
	training_data = []			#the moves will all be random
	scores = []    				# all scores
	accepted_scores = []		# just the scores that met our threshold
	
	# iterate through however many games we want:
	for _ in range(initial_games):
		score=0
		game_memory=[]			# we store all the movemets here to know if we beat the threshold score or not
		prev_observation=[]
		for _ in range(goal_steps):

			action=random.randrange(0,2)		# here we colud use  env.action_space.sample() to  create ranom samples but we did the former for simplicity
			observation,reward,done,info=env.step(action)

		# notice that the observation is returned FROM the action after it is cpmpleted, so pairing the action with observation resulting from it doen't make much sense
		# so we'll store the previous observation here, pairing
		# the prev observation to the action we'll take.
			if len(prev_observation) > 0 :
			    game_memory.append([prev_observation, action]) 			# prev_observation is a list of any features
			prev_observation = observation
			score+=reward
			if done: 
				break

		# IF our score is higher than our threshold, we'd like to save every move we made.NOTE the reinforcement methodology here. 
		# all we're doing is reinforcing the score, we're not trying to influence the machine in any way as to HOW that score is reached.


		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1]==0:					#converting to one hot array here the output are either 0 or 1 we can get away wothout using one-hot but in most-- 
					output =[1,0]				#---cases the output are greater than 2(i.e, the game uses more than two arrow keys to pplay the game)
				elif data[1]==1:
					output=[0,1]

				training_data.append([data[0],output])    #saving our training data
	
		# reset env to play again
		env.reset()
		# save overall scores
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy',training_data_save)

	print('average accepted score :',mean(accepted_scores))
	print('median of the accepted scores :',median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data


# Here we use a simple multilayer perceptron model
def neural_network_model(input_size):  #  In tesorflow we can save the model and retrain them, but to load the model that is predefined it has to be identical to the model we are loading
	# Generally we separate out the model , the training of the model and the using of the model
	network = input_data(shape= [None,input_size,1],name='input') 		# in this case the input size is four but we are keeping the model dynamic for future use

	network = fully_connected(network,128,activation='relu')   # 128 nodes are present in this layer
	network = dropout(network,0.8)		# Here 0.8 is the keep rate

	network = fully_connected(network,256,activation='relu')
	network = dropout(network,0.8)

	network = fully_connected(network,512,activation='relu')
	network = dropout(network,0.8)

	network = fully_connected(network,256,activation='relu')
	network = dropout(network,0.8)

	network = fully_connected(network,128,activation='relu')
	network = dropout(network,0.8)

	#output
	network = fully_connected(network,2,activation='softmax')		# the no of outputs is two,i.e,left or right
	network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

	model=tflearn.DNN(network,tensorboard_dir='log')   # In windows it creates a separate directory for the tensorboard

	return model

def train_model(training_data,model=False):		# we assign false because if there is no model it will create a model

	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y=[i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size=len(X[0]))  # here len(X[0]) ia the length of the X column of the trainig data

	model.fit({'input': X}, {'targets': y}, n_epoch=3,snapshot_step=500, show_metric=True, run_id='openai_learning')
	#increasing the leads to overfitment

	return model


training_data = initial_population()
model = train_model(training_data)

#model.save('cartpole.model')

model.load('cartpole.model')
# Now we play the game using neural nets
scores = []
choices = []  		# left or right
for each_game in range(10):			# we play ten games
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()

	for _ in range(goal_steps):
		env.render()

		if len(prev_obs)==0:
			action = random.randrange(0,2) 				# for the algo we give the initial push and then it learns through reinforcement

		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1,len(training_data[0][0]),1))[0]) # we [0] as we are predicting based on one frame right now

		choices.append(action)

		new_observation,reward,done,info = env.step(action)
		prev_obs=new_observation

		game_memory.append([new_observation,action])		# we donot need to do this unless we want to retrain and we can save this using reinforcement--
															#  -- which makes the neural network work better.
		score += reward
		if done:
			break

	scores.append(score)		

print('Average score :', sum(scores)/len(scores))
print('choice 1:{}     choice 2:{}'.format(choices.count(1)/len(choices) , choices.count(0)/len(choices)))
print(score_requirement)