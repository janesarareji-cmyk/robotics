from .models import RobotDefinition

def get_dh_table(robot: RobotDefinition):
    return list(robot.joints)
