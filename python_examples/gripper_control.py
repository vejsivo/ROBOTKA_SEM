from ctu_crs import CRS93

robot = CRS93()
robot.initialize()
robot.gripper.control_position_relative(0.5)
robot.gripper.control_position(robot.gripper.bounds[0])
robot.close()
