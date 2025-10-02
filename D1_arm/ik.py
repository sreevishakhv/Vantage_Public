import numpy as np
from ikpy.chain import Chain
# from ikpy.inverse_kinematics import inverse_kinematics
from scipy.spatial.transform import Rotation as R
import sys

class MyJointStatePublisher:
    def __init__(self, target_position, target_orientation, current_joint_angles_rad):
        # Load the robot chain from the URDF file
        # urdf_path = "unitree_sdk2/d1_sdk/4752f3ffb3a349ebaba90f8d69b4fefe/d1_550_description/urdf/d1_550_description.urdf"
        urdf_path = 'unitree_sdk2/d1_sdk/d1_description.urdf'
        self.my_chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])

        # Exclude fixed or non-movable joints from the calculations
        # self.my_chain.active_links_mask = [False, True, True, True, False, True, False]
        self.my_chain.active_links_mask = [False, True, True, True, False, False, False]


        self.target_position = target_position
        self.target_orientation = target_orientation
        self.initial_joint_angles = current_joint_angles_rad

        # Compute IK solution
        self.compute_ik()

    def compute_ik(self):
        # Convert Euler angles to a rotation matrix
        ik_solution = self.my_chain.inverse_kinematics(
                            target_position=self.target_position,
                            target_orientation=self.target_orientation,
                            initial_position=self.initial_joint_angles
                        )

        # Extract relevant joint angles (excluding first and last elements)
        self.ik_solution_radians = ik_solution[:7]
        self.ik_solution_degrees = np.degrees(self.ik_solution_radians)

def main():

    # Parse command line arguments
    x = float(sys.argv[1]) #/ 1000.0  # Convert mm to meters
    y = float(sys.argv[2]) #/ 1000.0
    z = float(sys.argv[3]) #/ 1000.0
    Rx = np.radians(float(sys.argv[4]))  # Convert degrees to radians
    Ry = np.radians(float(sys.argv[5]))
    Rz = np.radians(float(sys.argv[6]))

    target_position = [x, y, z]
    target_orientation = [Rx, Ry, Rz]

    current_joint_angles_rad = [np.radians(float(sys.argv[i])) for i in range(7, 14)]

    # Compute inverse kinematics
    MyJointStatePublisher(target_position, target_orientation, current_joint_angles_rad)

if __name__ == "__main__":
    main()
