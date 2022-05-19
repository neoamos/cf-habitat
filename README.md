# Crazyflie visual navigation
This project demonstrates a minimal end-to-end neural network policy for visual point goal navigation.  The network takes a grayscale image and an relative position of a goal coordinate from the agents current position, and output an action that gets the the agent closer to the goal coordinate position.  Actions can be either MOVE_FORWARD, TURN_LEFT, TURN_RIGHT or STOP.  The network is trained with reinforcement learning using the [AI Habitat](https://aihabitat.org/) simulator from Facebook.

This project also provides a remote control script for executing the neural network on a PC and sending commands to a Crazyflie equipped with an AI Deck camera and either a Flow Deck or Lighthouse Deck for positioning.

The full master thesis report is available [here](Masters_Thesis.pdf)


## Real world experiments

Return to home.  The crazyflie is manualy flown some distance from the starting position, and then autonmously flies back.

https://user-images.githubusercontent.com/25870484/169247602-1d23da56-43b5-4da2-8dde-5d36a29f8eee.mp4

Obstace avoidance.  The crazyflie avoids obstacles in a lab setting.

https://user-images.githubusercontent.com/25870484/169248120-152fda0b-fa0b-4860-aa1f-a4dd3899a334.mp4

