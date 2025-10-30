from ctu_crs import CRS93 # or CRS97

robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot, perform hard and soft home
q = robot.get_q()  # get current joint configuration
robot.move_to_q(q + [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # move robot all values in radians
robot.wait_for_motion_stop() # wait until the robot stops
robot.close()  # close the connection
