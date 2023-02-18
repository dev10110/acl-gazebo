#!/usr/bin/env python

# Devansh Agrawal

# This file simulates a quadrotor

#This files plots in gazebo with the position and orientation of the drone according to the desired position and acceleration specified in the goal topic

import roslib
import rospy
import math
from snapstack_msgs.msg import Goal, State
from geometry_msgs.msg import Pose, Vector3, Quaternion
from gazebo_msgs.msg import ModelState
import numpy as np
from numpy import linalg as LA

from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply, quaternion_matrix, quaternion_from_matrix

from visualization_msgs.msg import Marker

import tf


def vector3_to_np(v):
    return np.array([v.x, v.y, v.z])

def quat_to_np(q):
    return np.array([q.x, q.y, q.z, q.w])

def np_to_vector3(v):
    return Vector3(v[0], v[1], v[2])

def quaternion_to_rotation(q):
    qq = np.array([q.x, q.y, q.z, q.w])
    H = quaternion_matrix(qq / LA.norm(qq))
    return H[0:3, 0:3]

def hat(v):
    return np.array([
                [0., -v[2], v[1]],
                [v[2], 0., -v[0]],
                [-v[1], v[0], 0.]
            ])

def vee(M):
    return np.array([
        M[2,1],
        M[0,2],
        M[1,0]
        ])

def np_rotation_to_quat(R):
    
    # first project to SO(3)
    usvT = LA.svd(R)
    R = np.matmul(usvT[0], usvT[2])

    # make it a homogenous matrix
    H = np.block([[R, np.zeros([3,1])],[np.zeros([1,3]), 1]])

    # convert to quaternion
    q = quaternion_from_matrix(H)

    # return as quaternion object
    return Quaternion(q[0], q[1], q[2], q[3])


def normalize(v):
    return v / LA.norm(v)

# I despise python's np.matmul
def mm(v1, v2):
    return np.matmul(v1, v2)

def mmm(v1, v2, v3):
    return mm(v1, mm(v2, v3))

def mmmm(v1, v2, v3, v4):
    return mm( mm(v1, v2), mm(v3, v4) )

def log(v, label, every):
    str_ = label + ": %f, %f, %f" % (v[0], v[1], v[2])
    rospy.loginfo_throttle(every, str_)

class Quadrotor:


    def __init__(self):

        self.mass = 1.0
        self.g = 9.81
        self.J = np.diag([0.03, 0.03, 0.06])
    
        self.kx_ = 9.5
        self.kv_ = 4.0
        self.kR_ = 3.35
        self.kw_ = 0.35

    def step(self, state, goal, dt):

        # convert goal to f, M 
        f, M = self.goal_to_fM(state, goal)

        # run dynamics
        next_state = self.dynamics(state, f, M, dt)

        # return new state
        return next_state


    def dynamics(self, state, thrust, torque, dt):
        # state is of type snapstack_msgs.msg.State: [pos, vel, quat, w, abias, gbias]
        # returns next state 

        # extract state
        x = vector3_to_np(state.pos)
        v = vector3_to_np(state.vel)
        R = quaternion_to_rotation(state.quat) 
        w = vector3_to_np(state.w)

        # define constants
        e3 = np.array([0,0,1.])

        # get dynamics
        xdot = v 
        vdot = -self.g * e3 + (thrust / self.mass) * np.matmul(R, e3)
        Rdot = np.matmul(R , hat(w))
        wdot = np.matmul(LA.inv(self.J), torque - np.cross(w, np.matmul(self.J, w)))

        # propagate (euler ugh)
        x_next = x + xdot * dt
        v_next = v + vdot * dt
        R_next = R + Rdot * dt
        w_next = w + wdot * dt

        # convert to new state
        state_next = State()
        state_next.pos   = np_to_vector3(x_next)
        state_next.vel   = np_to_vector3(v_next)
        state_next.quat  = np_rotation_to_quat(R_next)
        state_next.w     = np_to_vector3(w_next)
        state_next.abias = Vector3(0,0,0.)
        state_next.gbias = Vector3(0,0,0.)

        return state_next

    def geometric_controller(self, state, xd, vd, ad, b1d, wd, alphad):

        # controller gains
        kx = self.kx_ # 4.0
        kv = self.kv_ # 2.0
        kR = self.kR_ # 3.35
        kw = self.kw_ # 0.15
        
        # extract state
        x = vector3_to_np(state.pos)
        v = vector3_to_np(state.vel)
        R = quaternion_to_rotation(state.quat) 
        w = vector3_to_np(state.w)

        # define constants
        e3 = np.array([0, 0, 1.0])

        # errors
        ex = x - xd
        ev = v - vd

        # construct desired rotation matrix
        fd =  -kx * ex - kv * ev + self.mass * self.g * e3 + self.mass * ad
        b3 = normalize(fd)
        b2 = normalize(np.cross(b3, normalize(b1d)))
        b1 = normalize(np.cross(b2, b3))

        Rd = np.column_stack([b1, b2, b3])

        eR = 0.5 * vee(np.matmul(Rd.T ,  R) - np.matmul(R.T , Rd) )
        ew = w - np.matmul(R.T, np.matmul(Rd, wd))

        f = np.dot(fd, np.matmul(R, e3))
        M =  -kR * eR - kw * ew + np.cross(w, np.matmul(self.J, w)) \
                - np.matmul(self.J ,  mmmm(hat(w),  R.T, Rd,  wd) - mmm(R.T, Rd , alphad) )

        return f, M


    def goal_to_fM(self, state, goal):
        
        # extract state
        p = vector3_to_np(goal.p)
        v = vector3_to_np(goal.v)
        a = vector3_to_np(goal.a)
        j = vector3_to_np(goal.j)
        s = np.zeros(3) # since this is not specified in the goal


        # implement my own yaw controller instead
        #yaw = goal.yaw
        #dyaw = goal.dyaw
        #ddyaw = 0.0

        ## TODO(dev): update dyaw from goal too
        yaw = goal.yaw
        dyaw = 0.0
        ddyaw = 0.0

        # desired rotation matrix
        force = np.array([a[0], a[1], a[2]+self.g])

        zb = normalize(force)

        xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        yb = normalize(np.cross(zb, xc))
        xb = normalize(np.cross(yb, zb))

        R = np.column_stack([xb, yb, zb])
        b1 = mm(R, np.array([1,0,0]))

        # construct tau
        tau = LA.norm(force)

        # construct S matrix
        bx1 = R[0, 0]
        bx2 = R[1, 0]
        bx3 = R[2, 0]
        by1 = R[0, 1]
        by2 = R[1, 1]
        by3 = R[2, 1]
        bz1 = R[0, 2]
        bz2 = R[1, 2]
        bz3 = R[2, 2]

        # 1x3 matrix
        S = np.array([
            0.0,
            (bx2 * bz1 - bx1 * bz2) / (bx1**2 + bx2**2),
            (-bx2 * by1 + bx1 * by2) / (bx1**2 + bx2**2)
        ])


        # solve for omega,  taudot
        iz = np.array([0, 0, 1])
        hatizT = hat(iz).T
        bz = R[:, 2]

        M = np.zeros([4,4]) 
        M[0:3, 0:3] = tau * mm(R , hatizT)
        M[0:3, 3] = bz
        M[3, 0:3] = S
        invM = LA.inv(M)

        tmp_v = np.zeros(4)
        tmp_v[0:3] = j
        tmp_v[3] = dyaw
        wtd = mm(invM , tmp_v)

        w = wtd[0:3]
        taud = wtd[3]

        # construct Sdot matrix
        # expression derived using mathematica
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]

        Sd = np.array([
            0.0, 
            (bx1 * w1) / (bx1**2 + bx2**2) + (bx2 * w2) / (bx1**2 + bx2**2) + ((bx1**2 * bz1 - bx2**2 * bz1 + 2 * bx1 * bx2 * bz2) * w3) / (bx1**2 + bx2**2)**2, 
            ((bx1**2 * bx2 + bx2**3 - bx1**2 * by1 + bx2**2 * by1 - 2 * bx1 * bx2 * by2) * w3) / (bx1**2 + bx2**2)**2
        ])

        # solve for alpha, taudd
        B1 = mmm(R, 2 * taud * hatizT + tau * mm(hat(w), hatizT),  w)
        B2 = np.dot(Sd,  w)
        tmp_v2 = np.zeros(4)
        tmp_v2[0:3] = s - B1
        tmp_v2[3] = ddyaw - B2
        alpha_taudd = mm(invM, tmp_v2)

        alpha = alpha_taudd[0:3]
        # taudd = alpha_taudd[3]


        ## use geometric controller to convert to f, M
        f, M = self.geometric_controller(state, p, v, a, b1, w, alpha)

        return f, M








class ImperfectSim:

    def __init__(self):

        # dt
        self.dt = 0.005 # runs every 5 milliseconds
        
        # construct quadrotor
        self.quad = Quadrotor()

        # construct goal
        self.goal = Goal()
        
        # construct state
        self.state=State()
        self.state.header.frame_id="world"

        self.state.pos.x = rospy.get_param('~x', 0.0);
        self.state.pos.y = rospy.get_param('~y', 0.0);
        self.state.pos.z = rospy.get_param('~z', 0.0);
        yaw = rospy.get_param('~yaw', 0.0);

        pitch=0.0;
        roll=0.0;
        quat = quaternion_from_euler(yaw, pitch, roll, 'rzyx')

        self.state.quat.x = quat[0]
        self.state.quat.y = quat[1]
        self.state.quat.z = quat[2]
        self.state.quat.w = quat[3]

        # create publishers
        self.pubGazeboState = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.pubMarkerDrone = rospy.Publisher('marker', Marker, queue_size=1, latch=True)
        self.pubState = rospy.Publisher('state', State, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.step)

        name = rospy.get_namespace()
        self.name = name[1:-1]
        
        # sleep for 1 second (not sure why?)
        rospy.sleep(1.0)

        # publish initial state
        self.publish_state()

        # now the timer should kick in



    def publish_state(self):
        
        
        # publish state
        self.pubState.publish(self.state)

        # publish marker
        pose=Pose()
        pose.position.x=self.state.pos.x;
        pose.position.y=self.state.pos.y;
        pose.position.z=self.state.pos.z;
        pose.orientation.x = self.state.quat.x
        pose.orientation.y = self.state.quat.y
        pose.orientation.z = self.state.quat.z
        pose.orientation.w = self.state.quat.w
        self.pubMarkerDrone.publish(self.getDroneMarker(pose));

        # publish gazebo state
        gazebo_state = ModelState()
        gazebo_state.model_name = self.name
        
        gazebo_state.pose.position.x = self.state.pos.x
        gazebo_state.pose.position.y = self.state.pos.y
        gazebo_state.pose.position.z = self.state.pos.z

        gazebo_state.pose.orientation.x = self.state.quat.x
        gazebo_state.pose.orientation.y = self.state.quat.y
        gazebo_state.pose.orientation.z = self.state.quat.z
        gazebo_state.pose.orientation.w = self.state.quat.w

        ## HACK TO NOT USE GAZEBO'S BUILT IN SIMULATION
        gazebo_state.reference_frame = "world" 
        self.pubGazeboState.publish(gazebo_state)
        ## END OF HACK TO NOT USE GAZEBO
        
        # publish tf
        br = tf.TransformBroadcaster()
        br.sendTransform(
                (self.state.pos.x, self.state.pos.y, self.state.pos.z),
                (self.state.quat.x, self.state.quat.y, self.state.quat.z, self.state.quat.w),
                rospy.Time.now(),
                self.name,
                "vicon"
                )

    def goalCB(self, goal_msg):

        self.goal = goal_msg
        #self.goal = Goal()
        #self.goal.p.z = 1.0;
    

    def step(self, timer):
        # this is run at regular intervals

        self.state = self.quad.step(self.state, self.goal, self.dt)
        
        # publish
        self.publish_state()


    def getDroneMarker(self, pose):
        marker=Marker();
        marker.id=1;
        marker.ns="mesh_"+self.name;
        marker.header.frame_id="world"
        marker.type=marker.MESH_RESOURCE;
        marker.action=marker.ADD;

        marker.pose=pose
        marker.lifetime = rospy.Duration.from_sec(0.0);
        marker.mesh_use_embedded_materials=True
        marker.mesh_resource="package://acl_sim/meshes/quadrotor/quadrotor.dae"
        marker.scale.x=1.0;
        marker.scale.y=1.0;
        marker.scale.z=1.0;
        return marker 



             

def startNode():
    c = ImperfectSim()
    rospy.Subscriber("goal", Goal, c.goalCB)

    rospy.spin()

if __name__ == '__main__':

    ns = rospy.get_namespace()
    try:
        rospy.init_node('relay')
        if str(ns) == '/':
            rospy.logfatal("Need to specify namespace as vehicle name.")
            rospy.logfatal("This is tyipcally accomplished in a launch file.")
            rospy.logfatal("Command line: ROS_NAMESPACE=mQ01 $ rosrun quad_control joy.py")
        else:
            print "Starting imperfect tracker node for: " + ns
            startNode()
    except rospy.ROSInterruptException:
        pass
