#!/usr/bin/env python

# packages
import rospy
import numpy as np
import range_libc
import time
from threading import Lock
import tf.transformations
import tf
import utils as Utils

# messages
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

# visualization packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # this part is needed even though it is grayed out by PyCharm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

###
# This code is from here: https://github.com/mit-racecar/particle_filter/blob/master/src/particle_filter.py
# It is used to visualize our sensor model in SensorModel.py
# This file is not part of our project but only exists to assist us with verifying our project
###

def show_viz(sensor_model_table, table_width):
    # code to generate various visualizations of the sensor model
    if True:
        # visualize the sensor model
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0, table_width, 1.0)
        Y = np.arange(0, table_width, 1.0)
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, sensor_model_table, cmap="bone", rstride=2, cstride=2,
                               linewidth=0, antialiased=True)

        ax.text2D(0.05, 0.95, "Precomputed Sensor Model", transform=ax.transAxes)
        ax.set_xlabel('Ground truth distance (in px)')
        ax.set_ylabel('Measured Distance (in px)')
        ax.set_zlabel('P(Measured Distance | Ground Truth)')

        plt.show()
    if True:
        plt.imshow(sensor_model_table * 255, cmap="gray")
        plt.show()
    if True:
        plt.plot(sensor_model_table[:, 140])
        plt.plot([139, 139], [0.0, 0.08], label="test")
        plt.ylim(0.0, .08)
        plt.xlabel("Measured Distance (in px)")
        plt.ylabel("P(Measured Distance | Ground Truth Distance = 140px)")

    plt.show()