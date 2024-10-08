
from collections import defaultdict 
import random
import matplotlib.pyplot as plt

dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}

dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}

# User defined variables

training_end = 40000    # the number of  traning session steps: should be less than 100000
alpha = 0.9       # learning rate, between 0 and 1.  0 means nothing learned. 0.9 means tht learning occur very quickly.
gamma = 0.9       # discount factor, between 0 and 1.  future reward is less important than immediate reward
epsilon = 0.1     # epsilon-greedy variable.  ) means nothing implemented.  

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        # Robot's initial location and orientation
        self.location = [0, 0]
        self.heading = 'u'
        self.maze_dim = maze_dim
        
        # Initialize Q- value with 0
        self.Qval = defaultdict(int) # Adding a new key will create 0 value
        
        # Assign reward values on the maze. Goal for 100 elsewhere for -1
        self.reward = [[-1 for _ in range(maze_dim)] for _ in range(maze_dim)]
        self.reward[maze_dim/2 - 1][maze_dim/2 - 1] = 100
        self.reward[maze_dim/2 - 1][maze_dim/2] = 100
        self.reward[maze_dim/2][maze_dim/2 -1] = 100
        self.reward[maze_dim/2][maze_dim/2] = 100
        
        # Initialize  program variables
        self.count = 0
        self.run = 0
        
        # Initialize state and action
        self.state = (0,0,'u')
        self.action = (0,1)
        
        # Intialize the traning path list
        self.xlist = []
        self.ylist = []
        self.step = []
        
        # Initialize the test path list
        self.xtest = []
        self.ytest = []
        
    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''

        # Start training first
        if self.run == 0 :   
            rotation, movement = self.run_training(sensors)
        # Start test after training is over
        elif self.run == 1:  
            rotation, movement = self.run_testing(sensors)

        return rotation, movement
    
    # Run training
    def run_training(self,sensors):

        print "Trainging Count: ", self.count
        self.count +=1

        # Get robot's current position          
        x1 = self.location[0]
        y1 = self.location[1]
        
        # Add the position to the list to plot path later
        self.xlist.append(x1)
        self.ylist.append(y1)

        # Define next state 
        state_next = (x1,y1,self.heading)        
        
        # Define possible actions at the next state using robot sensors
        possible_actions = []
        if sensors == [0,0,0]:  #Encounter deadend
            movement = 0
            rotation = 90
            possible_actions.append((90,0))
        else:                   #At least, one openning available   
            if sensors[0] != 0:
                possible_actions.append((-90,1))
            if sensors[1] != 0:
                possible_actions.append((0,1))
            if sensors[2] != 0:
                possible_actions.append((90,1))

        # Compute the Qmax with  possible actions: max Q[S',a']
        Qmax = -1000   # Initialize Qmax and actMax
        actMax = None
        for action_next in possible_actions:
           if  self.Qval[(state_next,action_next)] > Qmax:
               Qmax = self.Qval[(state_next,action_next)]
               actMax = action_next
           else:
               continue
        
        # Define random moves from possible actions for training. 
        random_action = random.choice(possible_actions)  
        rotation = random_action[0]
        movement = random_action[1]

        # Solve BELLMAN equation
        self.Qval[(self.state, self.action)] = (1-alpha)*self.Qval[(self.state,self.action)] + alpha*(self.reward[self.state[0]][self.state[1]] + gamma*Qmax)  

        # Define current state to be used in the next step
        self.state = state_next
        self.action = random_action
        
        # Update robot heading and location after the next step
        self.update_robot_heading_location(rotation,movement)

        # End training
        if self.count == training_end:
            rotation = 'Reset'
            movement = 'Reset'

            # Plot paths  taken by the robot during the robot
            plt.figure(1)    
            plt.plot(self.xlist,self.ylist, 'b', linewidth=3.0)
            plt.axis([ -1, self.maze_dim, -1,self.maze_dim])
            plt.grid(True)  
            
            # Intialize variables to start test
            self.run = 1
            self.location = [0,0]
            self.heading = 'u'
            self.count = 0

        return rotation, movement
    
    # Run testing
    def run_testing(self,sensors):
        
        print "Testing Count: "
        self.count +=1

        # Get robot's position
        x1 = self.location[0]
        y1 = self.location[1]
        
        # Add its position to the list to plot later
        self.xtest.append(x1)
        self.ytest.append(y1)

        # Assemble state
        state = (x1, y1, self.heading)
        
        # Determine possible actions from  wall sensors
        possible_actions = []
        if sensors == [0,0,0]:  #Encounter deadend
            movement = 0
            rotation = 90
            possible_actions.append((90,0))
        else:                   #At least, one openning available   
            if sensors[0] != 0:
                possible_actions.append((-90,1))
            if sensors[1] != 0:
                possible_actions.append((0,1))
            if sensors[2] != 0:
                possible_actions.append((90,1))
        
        # Compute Qmax and actMax from the possible actions
        Qmax = -1000   
        actMax = None
        for action in possible_actions:
            if self.Qval[(state,action)] > Qmax:
                Qmax = self.Qval[(state,action)]
                actMax = action
        
        # Implement Epsilon-greedy method.
        if random.random() > epsilon/self.count:         
            rotation = actMax[0]
            movement = actMax[1]
        else:
            rotation = random.choice(possible_actions)[0]
            movement = random.choice(possible_actions)[1]
        
        # Update robot's position
        self.update_robot_heading_location(rotation,movement)
       
        # Plot the complete & optimal path if it reached the goal
        if self.reward[self.location[0]][self.location[1]]  == 100:
            plt.figure(1)    
            plt.plot(self.xtest,self.ytest, 'r',linewidth=3.0)
            plt.axis([ -1, self.maze_dim, -1,self.maze_dim])
            plt.grid(True) 

        return rotation, movement

    # Update robot heading and location
    def update_robot_heading_location(self, rotation, movement):
   
        # Convert from [-90, 0, 90]  to [0, 1,2]
        rotation_index = rotation/90 + 1
        
        # Compute new headig from the rotation
        self.heading  = dir_sensors[self.heading][rotation_index]
        
        # Update location upon heading direction
        self.location[0] += dir_move[self.heading][0]*movement
        self.location[1] += dir_move[self.heading][1]*movement

      