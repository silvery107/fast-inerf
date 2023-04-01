import numpy as np 

def load_init_pose(obj_pose):

    # From poseCNN
    # TODO: get the R and t of the center object in the scene
    R_cam_obj, t_cam_obj = obj_pose     # rotation matrix and translation vector from camera to object

    R_obj_cam = R_cam_obj.T                                     # rotation matrix from object to camera
    P_obj_cam = R_obj_cam @ (np.array([0, 0, 0]) - t_cam_obj)   # position of camera in object coordinate
    
    # Assume the object is at cenetr of scene and global (x,y,z) is aligned with object (x,y,z)
    start_pose = np.hstack((R_obj_cam, P_obj_cam.reshape(3, 1))) # 3x4 matrix
    start_pose = np.vstack((start_pose, np.array([0, 0, 0, 1]))) # 4x4 matrix

    return start_pose