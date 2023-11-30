import os
import torch
import imageio
import numpy as np
import skimage
import cv2
from nerf import NeRF, get_embedder, run_network
from utils.inerf_utils import config_parser, load_blender, show_img, find_POI, load_llff_data, camera_transf
from utils.render_utils import render, get_rays, to8b, img2mse
torch.autograd.set_detect_anomaly(True)
from utils.fast_inerf_utils import load_init_pose
import torchvision.models as models
from pose_cnn import PoseCNN
from numpy import savetxt
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# np.random.seed(0)

def load_nerf(args, device):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                      input_ch=input_ch, output_ch=output_ch, skips=skips,
                      input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)
    # Load checkpoint
    ckpt_dir = args.ckpt_dir
    ckpt_name = args.model_name
    ckpt_path = os.path.join(ckpt_dir, ckpt_name+'.tar')
    print('Found checkpoints', ckpt_path)
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)

    # Load model
    model.load_state_dict(ckpt['network_fn_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs['ndc'] = False
        render_kwargs['lindisp'] = args.lindisp

    # Disable updating of the weights
    for param in model.parameters():
        param.requires_grad = False
    for param in model_fine.parameters():
        param.requires_grad = False

    return render_kwargs


def run_inerf(_overlay=False, _debug=False):
    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    obs_img_num = args.obs_img_num
    batch_size = args.batch_size
    spherify = args.spherify
    kernel_size = args.kernel_size
    lrate = args.lrate
    dataset_type = args.dataset_type
    sampling_strategy = args.sampling_strategy
    delta_phi, delta_theta, delta_psi, delta_t = args.delta_phi, args.delta_theta, args.delta_psi, args.delta_t
    noise, sigma, amount = args.noise, args.sigma, args.amount
    delta_brightness = args.delta_brightness
    posecnn_dir = args.posecnn_dir
    posecnn_init_pose = args.posecnn_init_pose
    _overlay = args.overlay or _overlay

    # Load and pre-process an observed image
    # obs_img -> rgb image with elements in range 0...255
    if dataset_type == 'blender':
        obs_img, hwf, start_pose, obs_img_pose = load_blender(args.data_dir, model_name, obs_img_num,
                                                args.half_res, args.white_bkgd, delta_phi, delta_theta, delta_psi, delta_t)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender
    else:
        obs_img, hwf, start_pose, obs_img_pose, bds = load_llff_data(args.data_dir, model_name, obs_img_num, delta_phi,
                                                delta_theta, delta_psi, delta_t, factor=8, recenter=True, bd_factor=.75, spherify=spherify)
        H, W, focal = hwf
        H, W = int(H), int(W)
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

    obs_img = (np.array(obs_img) / 255.).astype(np.float32)
    
    if posecnn_init_pose:
        # PROPS Pose dataset intrinsic TODO: should be parsed in
        cam_intrinsic = np.array([
                            [902.19, 0.0, 342.35],
                            [0.0, 902.39, 252.23],
                            [0.0, 0.0, 1.0]])
        
        # Load trained posecnn model  
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        posecnn_model = PoseCNN(pretrained_backbone = vgg16, 
                        models_pcd = None,
                        cam_intrinsic = cam_intrinsic).to(device)
        posecnn_model.load_state_dict(torch.load(os.path.join(posecnn_dir, "posecnn_model.pth")))
        posecnn_model.eval()

        # Prepare rgb as posecnn input format
        pose_cnn_rgb = torch.tensor(obs_img.transpose((2,0,1))[None, :]).to(device)
        inputdict = {'rgb': pose_cnn_rgb}

        # pose_dict[0]: {class_label : 4x4 pose matrix, ...}
        # label[0]:     (H, W) pixel class label
        with torch.no_grad():
            pose_dict, label = posecnn_model(inputdict) 

        start_pose = load_init_pose(pose_dict[0], label[0], start_pose)
        # import matplotlib.pyplot as plt
        # plt.imshow(prediction.transpose((1,2,0)))
        # plt.show()


    # change brightness of the observed image (to test robustness of inerf)
    if delta_brightness != 0:
        obs_img = (np.array(obs_img) / 255.).astype(np.float32)
        obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2HSV)
        if delta_brightness < 0:
            obs_img[..., 2][obs_img[..., 2] < abs(delta_brightness)] = 0.
            obs_img[..., 2][obs_img[..., 2] >= abs(delta_brightness)] += delta_brightness
        else:
            lim = 1. - delta_brightness
            obs_img[..., 2][obs_img[..., 2] > lim] = 1.
            obs_img[..., 2][obs_img[..., 2] <= lim] += delta_brightness
        obs_img = cv2.cvtColor(obs_img, cv2.COLOR_HSV2RGB)
        show_img("Observed image", obs_img)

    # apply noise to the observed image (to test robustness of inerf)
    if noise == 'gaussian':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='gaussian', var=sigma**2)
    elif noise == 's_and_p':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='s&p', amount=amount)
    elif noise == 'pepper':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='pepper', amount=amount)
    elif noise == 'salt':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='salt', amount=amount)
    elif noise == 'poisson':
        obs_img_noised = skimage.util.random_noise(obs_img, mode='poisson')
    else:
        obs_img_noised = obs_img

    obs_img_noised = (np.array(obs_img_noised) * 255).astype(np.uint8)
    if _debug:
        show_img("Observed image", obs_img_noised)

    ################### Start ###################
    # bbx: size: (N,4) with (x1, y1, x2, y2)
    # (x1, y1) is the top left corner of the bounding box 
    # (x2, y2) is the bottom right corner of the bounding box.
    if args.mask_region:
        from pose_cnn import getBbx, getSegMask
        # bbx = getBbx(label)
        # for ii in range(bbx.shape[0]):
        #     x1, y1, x2, y2 = int(bbx[ii, 0]), int(bbx[ii, 1]), int(bbx[ii, 2]), int(bbx[ii, 3])

        #     x = np.arange(x1, x2+1, 1)
        #     y = np.arange(y1, y2+1, 1)
        #     xx, yy = np.meshgrid(x, y)

        #     obj = np.c_[yy.ravel(), xx.ravel()]

        #     if ii == 0: 
        #         POI = obj
        #     else:
        #         POI = np.concatenate((POI, obj), axis=0)
        obj_mask = getSegMask(label).cpu().numpy() # (H, W)
        seg_index = obj_mask.nonzero()
        POI = np.stack([seg_index[1], seg_index[0]], axis=1)
        print(f"POI: {POI.shape}")

    ################### End ###################

    else:
        # find points of interest of the observed image
        POI = find_POI(obs_img_noised, _debug)  # xy pixel coordinates of points of interest (N x 2)
        print(f"POI: {POI.shape}")

    obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)

    # create meshgrid from the observed image
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                            dtype=int)

    # create sampling mask for interest region sampling strategy
    interest_regions = np.zeros((H, W, ), dtype=np.uint8)
    interest_regions[POI[:,1], POI[:,0]] = 1
    if not args.mask_region:
        I = args.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    interest_regions = coords[interest_regions]

    # not_POI -> contains all points except of POI
    coords = coords.reshape(H * W, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)

    # Load NeRF Model
    render_kwargs = load_nerf(args, device)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    # Create pose transformation model
    start_pose = torch.Tensor(start_pose).to(device)
    cam_transf = camera_transf(device).to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

    # calculate angles and translation of the observed image's pose
    phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)
    #translation_ref = obs_img_pose[2, 3]

    testsavedir = os.path.join(output_dir, model_name)
    os.makedirs(testsavedir, exist_ok=True)

    # TODO save imgs to a valid gif or mp4 format
    if _overlay is True:
        imgs = []

    errors = np.array([[0, 0, 0, 0]])

    for k in range(200):

        if sampling_strategy == 'random':
            rand_inds = np.random.choice(coords.shape[0], size=batch_size, replace=False)
            batch = coords[rand_inds]

        elif sampling_strategy == 'interest_points':
            if POI.shape[0] >= batch_size:
                rand_inds = np.random.choice(POI.shape[0], size=batch_size, replace=False)
                batch = POI[rand_inds]
            else:
                batch = np.zeros((batch_size, 2), dtype=np.int)
                batch[:POI.shape[0]] = POI
                rand_inds = np.random.choice(not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                batch[POI.shape[0]:] = not_POI[rand_inds]

        elif sampling_strategy == 'interest_regions':
            rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
            batch = interest_regions[rand_inds]

        else:
            print('Unknown sampling strategy')
            return

        target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
        target_s = torch.Tensor(target_s).to(device)
        pose = cam_transf(start_pose)

        rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
        rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 1], batch[:, 0]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=k < 10, retraw=True,
                                        **render_kwargs)

        optimizer.zero_grad()
        # print(rgb.shape, target_s.shape)
        loss = img2mse(rgb, target_s) # (batch_size, 3)
        loss.backward()
        optimizer.step()

        new_lrate = lrate * (0.8 ** ((k + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (k + 1) % 20 == 0 or k == 0:
            print('Step: ', k)
            print('Loss: ', loss.item())

            with torch.no_grad():
                pose_dummy = pose.cpu().detach().numpy()
                # calculate angles and translation of the optimized pose
                phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                #translation = pose_dummy[2, 3]
                # calculate error between optimized and observed pose
                phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                rot_error = phi_error + theta_error + psi_error
                translation_error = abs(translation_ref - translation)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                print('-----------------------------------')

                # err = np.array([[k, loss.item(), rot_error, translation_error]])
                # errors = np.concatenate((errors, err), axis=0)

            if _overlay is True:
                with torch.no_grad():
                    rgb, disp, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                    rgb = rgb.cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(obs_img)
                    filename = os.path.join(testsavedir, str(k)+'.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    imgs.append(dst)

        err = np.array([[k, loss.item(), rot_error, translation_error]])
        errors = np.concatenate((errors, err), axis=0)

    t = int(time())
    savetxt(f'logs/{t}_data.csv', errors, delimiter=',')

    # ## load data and read
    # training_history = np.loadtxt("data.csv", delimiter=",", dtype=float)
    # ks = training_history[1:, 0]
    # losses = training_history[1:, 1]
    # rot_errors = training_history[1:, 2]
    # translation_errors = training_history[1:, 3]

    # plt.figure()
    # plt.plot(ks, losses)
    # plt.savefig('loss.png')
    # plt.figure()
    # plt.plot(ks, rot_errors)
    # plt.savefig('rotationError.png')
    # plt.figure()
    # plt.plot(ks, translation_errors)
    # plt.savefig('translationError.png')

    if _overlay is True:
        imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format
