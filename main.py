from config import config as conf
from ctu_crs import CRS93, CRS97


def initialize_robot():
    robot_type = conf.get("robot_type")

    if robot_type == "CRS97":
        robot = CRS97()
    elif robot_type == "CRS93":
        robot = CRS93()
    elif robot_type == "no_robot":
        robot = CRS93(tty_dev=None)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    robot.initialize()
    return robot


def end_robot(robot):
    robot.soft_home()
    robot.close()


def main():
    robot = initialize_robot()
    

    end_robot(robot)


if __name__ == "__main__":
    main()
