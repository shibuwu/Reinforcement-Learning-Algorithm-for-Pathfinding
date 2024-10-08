### Micromouse Robot

#### Description

Navigating in unknown real world is a key challenge in autonomous vehicle or mobile robot application.  In this project, the problem is simplified as a robot navigating in an unknown maze and finding its optimal path.

The scope of the project is to develop a motion planning algorithm that enables a robot to explore an unknown maze (environment), to learn the maze layout, and then to find its fastest path from a corner of the maze to its center.  

#### Method

Used Reinforcement Learning to learn an unknown maze and to find its the fastest path. 

#### Usage

To run the program:
`python tester.py  test_maze_01.txt`

To display the maze:
`python showmaze.py test_maze_01.txt`

#### Note
* "Multiple steps at one move" has not been implemented.
* The number of training steps is set by `training_end = 40000` in robot.py. It must be less than 99000. Larger it is (=more traing), the algorithm produces more consistent an optimal result.
* Parameters such as alpha, gamma and epilson have not been optimized.
