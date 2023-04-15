import numpy as np 
from utils.inerf_utils import rot_psi, rot_phi, rot_theta

def load_init_pose(pose_dict, label, est_pose):
    # pose_dict: {class_label : 4x4 pose matrix, ...}
    # label:     (H, W) pixel class label
    # From poseCNN
    # TODO: get the R and t of the center object in the scene
    # for cl, obj_pose in pose_dict.items():
    #     # rotation matrix and translation vector from camera to object
    #     R_cam_obj = obj_pose[0:3, 0:3]
    #     t_cam_obj = obj_pose[0:3, -1]
    #     print(cl, t_cam_obj)
    
    R_cam_obj = pose_dict[2][0:3, 0:3]
    t_cam_obj = pose_dict[2][0:3, -1] * 4

    # task: for a picture, know {angles, z, y} of the camera 
    # find x

    # distance between camera and object 
    # dist = np.linalg.norm([-2.296917, -0.12373751, 3.310417])
    dist = np.linalg.norm(t_cam_obj)

    est_trans = np.sqrt(dist**2 - est_pose[1][3]**2 - est_pose[0][3]**2)
    print("calculated x: ", est_trans)
    est_pose[2][3] = est_trans # z
    print("calculated start pose: \n", est_pose)

    R_obj_cam = R_cam_obj.T                                     # rotation matrix from object to camera
    P_obj_cam = R_obj_cam @ t_cam_obj[:, None]   # position of camera in object coordinate
    
    # Assume the object is at cenetr of scene and global (x,y,z) is aligned with object (x,y,z)
    start_pose = np.hstack((R_obj_cam, P_obj_cam.reshape(3, 1))) # 3x4 matrix
    start_pose = np.vstack((start_pose, np.array([0, 0, 0, 1]))) # 4x4 matrix
    # start_pose = rot_phi(np.pi/2) @ rot_psi(np.pi/2) @ start_pose # obs_img 0
    start_pose = rot_psi(np.pi/2) @ start_pose # obs_img 2
    start_pose[:3, -1] = est_pose[:3, -1]
    print("Posecnn pose\n", start_pose)

    # return start_pose
    return est_pose