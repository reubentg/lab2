import math
import numpy as np

'''
Apply the kinematic model to the passed pose and control
  pose: The current state of the robot [x, y, theta]
  control: The controls to be applied [v, delta, dt]
  car_length: The length of the car
Returns the resulting pose of the robot
'''
def kinematic_model_step(pose, control, car_length):
    # Apply the kinematic model
    # Make sure your resulting theta is between 0 and 2*pi
    # Consider the case where delta == 0.0
    v = control[:, 0] # get velocity column. shape (MAX_PARTICLES,)
    delta = control[:, 1] # get angle column. shape (MAX_PARTICLES,)
    dt = control[:, 2] # get time change column, shape (MAX_PARTICLES,)

    xt_minus_1 = pose[0] # only one value for initial pose position x
    yt_minus_1 = pose[1] # only one value for initial pose position y
    theta_t_minus_1 = pose[2] # only one value for initial pose orientation theta
    beta = np.arctan((1.0 / 2.0) * np.tan(delta)) # beta shape (1000,)
    theta = theta_t_minus_1 + v / car_length * np.sin(2 * beta * dt) # theta shape (1000,)

    ## Basically the part below is a just a vectorized version of this
    # if beta == 0: # case where delta == 0.0
    #     xt = xt_minus_1 + v * dt  # distance = speed * time
    #     yt = 0  # no change in y because we are going straight along the x axis in the car's frame
    # else:
    #     xt = xt_minus_1 + car_length / math.sin(2 * beta) * (math.sin(theta) - math.sin(theta_t_minus_1))
    #     # print "xt,", xt
    #     yt = yt_minus_1 + car_length / math.sin(2 * beta) * (-math.cos(theta) + math.cos(theta_t_minus_1))
    #     # print "yt", yt

    xt = np.zeros(control.shape[0]) # xt shape (MAX_PARTICLES,)
    yt = np.zeros(control.shape[0])  # yt shape (MAX_PARTICLES,)

    zero_indices = np.argwhere(beta == 0) # case where delta == 0.0
    non_zero_indices = np.argwhere(beta != 0)


    xt[zero_indices] = xt_minus_1 + v[zero_indices] * dt[zero_indices]
    yt[zero_indices] = 0

    xt[non_zero_indices] = xt_minus_1 + car_length / np.sin(2 * beta[non_zero_indices]) * (
        np.sin(theta[non_zero_indices]) - np.sin(theta_t_minus_1))
    yt[non_zero_indices] = yt_minus_1 + car_length / np.sin(2 * beta[non_zero_indices]) * (
            -np.cos(theta[non_zero_indices]) + np.cos(theta_t_minus_1))

    noisy_particles = np.zeros(control.shape) # shape (MAX_PARTICLES, 3)
    noisy_particles[:, 0] = xt
    noisy_particles[:, 1] = yt
    noisy_particles[:, 2] = theta
    return noisy_particles