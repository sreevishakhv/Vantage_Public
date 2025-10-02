import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
import sys

class ForwardKinematicsSolver_old:
    def __init__(self, joint_angles):
        # Load the robot chain from the URDF file
        # urdf_path = "unitree_sdk2/d1_sdk/4752f3ffb3a349ebaba90f8d69b4fefe/d1_550_description/urdf/d1_550_description.urdf"
        urdf_path = 'unitree_sdk2/d1_sdk/d1_description.urdf'
        self.my_chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])

        # Exclude fixed joints
        self.my_chain.active_links_mask = [False, True, True, True, True, True, True, True]


        self.joint_angles = joint_angles

        # Compute Forward Kinematics
        self.compute_fk()

    def compute_fk(self):
        # Compute the transformation matrix for the end-effector
        # fk_matrix = self.my_chain.forward_kinematics([0] + self.joint_angles + [0])
        fk_matrix = self.my_chain.forward_kinematics(self.joint_angles)

        # Extract position (translation part)
        position = fk_matrix[:3, 3]

        # Extract orientation (rotation part as Euler angles)
        rotation = R.from_matrix(fk_matrix[:3, :3]).as_euler('xyz', degrees=True)

        # Print results
        print("End-Effector Position (meters):", position)
        print("End-Effector Orientation (degrees):", rotation)
        new = [i for i in position]
        for i in rotation:
            new.append(i)

        return list(position) + list(rotation)


class ForwardKinematicsSolver:
    def __init__(self):
        urdf_path = 'unitree_sdk2/d1_sdk/d1_description.urdf'
        self.my_chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])
        self.my_chain.active_links_mask = [False, True, True, True, True, True, True, True]

    def compute_fk(self, joint_angles_radians):
        fk_matrix = self.my_chain.forward_kinematics(joint_angles_radians)

        position = fk_matrix[:3, 3]
        rotation = R.from_matrix(fk_matrix[:3, :3]).as_euler('xyz', degrees=True)

        return list(position) + list(rotation)
    

def main():
    # if len(sys.argv) != 7:
    #     print("Usage: fk_solver.py J1 J2 J3 J4 J5 J6")
    #     print("Example: fk_solver.py 0 30 -45 60 90 0")
    #     sys.exit(1)

    # Parse joint angles from command-line arguments
    # joint_angles_degrees = [float(angle) for angle in sys.argv[1:7]]
    joint_angles_degrees = [float(angle) for angle in sys.argv[1:]]

    joint_angles_radians = np.radians(joint_angles_degrees)  # Convert degrees to radians

    # Compute forward kinematics
    ForwardKinematicsSolver_old(joint_angles_radians)

if __name__ == "__main__":
    main()
