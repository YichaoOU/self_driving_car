# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:29:56 2018

@author: Lester
"""
import multiprocessing
from scipy.spatial.distance import euclidean as eDist
# from rplidar import RPLidar
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import numpy as np
import math as mt
# from msvcrt import kbhit
#import queue
#import threading
#import time

#from matplotlib import interactive
#interactive(True) # Draw the plots as soon as they are created

PORT_NAME = 'com4'
x_min = -4.5
x_max = 4.5
y_min = -4.5
y_max = 4.5


"car Paramter"
L = 0.72   # wheelbase
d_f = 0.3 # front overhang
d_r = 0.21 # rear overhang
b = 0.32   # half width
R_min = 1.2 # 1.7 minimum turning radius 2.68
r = L/R_min # tan(delta_max)
delta_max = mt.atan(r) # Maximum steering angle ~ 25 degree = 0.4 rad


scale = 10 # Map scale
# Translation distance from the body frame to the discrete body frame
W = -mt.floor(-b*scale) #Width of the buffer zone in matrix
W_t = -mt.floor(-b*scale*2) # Width of the outer tolerance zone zone in matrix
# under good condition
int_W = int(W)
int_W_t = int(W_t)
tabu_left = int_W_t - int_W
tabu_right = int_W_t + int_W
tabu_top = tabu_left
tabu_bottom = tabu_right
"Initialize the map matrix"
MAP = np.zeros((int(4.5*2*scale),int(4.5*2*scale)), dtype=int)
MAP_obs_ind = []

"Stop signal"
stop = 0

"The left line, right line and middle line: left line: 4, right line: 6, middle line: 5"
for i in range(0,4*scale):
    MAP[int(4.5*scale + i),int(-0.75*scale+4.5*scale)] = 4
    MAP[int(4.5*scale + i),int(0.75*scale+4.5*scale)] = 6
    #MAP[int(4.5*scale + i),int(4.5*scale)] = 5
left_line = np.array([(np.where(MAP == 4)[0].astype(float))/scale - 4.5, (np.where(MAP == 4)[1].astype(float))/scale - 4.5])
right_line = np.array([(np.where(MAP == 6)[0].astype(float))/scale - 4.5, (np.where(MAP == 6)[1].astype(float))/scale - 4.5])
#mid_line = np.array([(np.where(MAP == 5)[0].astype(float))/scale - 4.5, (np.where(MAP == 5)[1].astype(float))/scale - 4.5])

"Virtual MAP_lidar"
Range_lidar = 0.3
MAP_lidar = np.zeros((int(4.5*2*scale),int(4.5*2*scale)), dtype=int)
for i in range (-5,6):
    for k in range (-5,6):
        if i*scale < Range_lidar:
            MAP_lidar[int(7*scale + i),int(4.5*scale+k)] = 10
point_lidar = np.array([(np.where(MAP_lidar == 10)[0].astype(float))/scale - 4.5, (np.where(MAP_lidar == 10)[1].astype(float))/scale - 4.5])

"Virtual Target"
Target = np.array([8.5*scale,4.5*scale])

"Initialize the Trajectory, which is global"
"Sampling Time"
Ts = 0.01
"Step size for path planning"
step_size = 0.05
"Existence of lines"
Ex_line = 3 # Have both lines

"Initialize the lidar and figure"
#def run():
#lidar = RPLidar(PORT_NAME)
fig = plt.figure()


#iterator = lidar.iter_scans() # Read Lidar data sequence

    
def ReadLidar():
    
    "The data are unstacle, so read three times and get the mean value"
    scan1 = next(iterator)
    scan2 = next(iterator)
    scan = scan1 + scan2 
    obs_pos = np.array([(round(mt.cos(np.radians(meas[1]))*meas[2]/(1000/scale)), round(mt.sin(np.radians(meas[1]))*meas[2]/(1000/scale))) for meas in scan])  
    
    "Build the map"
    global MAP

    MAP = np.zeros((int(4.5*2*scale),int(4.5*2*scale)), dtype=int) # Update map
    for i in range(0,obs_pos.shape[0]):
        if abs(obs_pos[i,0]) < 4.5*scale and abs(obs_pos[i,1]) < 4.5*scale:
            a = int(4.5*scale + obs_pos[i,0])  
            b = int(4.5*scale + obs_pos[i,1])
            MAP[a,b] = 10
    for i in range(0,4*scale): # Set lines after we set the obstacles
        MAP[int(4.5*scale + i),int(-0.75*scale+4.5*scale)] = 4
        MAP[int(4.5*scale + i),int(0.75*scale+4.5*scale)] = 6
#        if MAP[int(4.5*scale + i),int(4.5*scale)] == 0:
#            MAP[int(4.5*scale + i),int(4.5*scale)] = 5
    points = obs_pos/scale   
    return points

def Pad_Zone_parallel(MAP):
    # list every point
    
    usable_args = []
    n,m = MAP.shape
    for i in range(n):
        for j in range(m):
            status = MAP[i,j]
            if not status == 0:
                usable_args.append[i,j,status]
    not_used_points_index = []
    # filter points to speed up
    for i in range(len(usable_args)):
        if i in not_used_points_index:
            continue
        for j in range(i,len(usable_args)):
            if i[0] - j[0] < 3 or i[1] - j[1] < 3:
                not_used_points_index.append(j)
    # get zone for every obstacles
    pool = multiprocessing.Pool(processes=4)
    result_list = pool.map(Pad_zone_unit, usable_args)
    for item in result_list:
        MAP_out,left,right,bottom,top = item
        for i in range(left,right+1):
            for j in range(top,bottem+1):
                # put values
                if MAP[i,j] == 1:
                    continue
                MAP[i,j] = MAP_out[i-left,j-top]
    return MAP


def unit_circle_vectorized(r,code):
    A = np.arange(-r,r+1)**2
    # print (A)
    dists = np.sqrt(A[:,None] + A)
    # print (dists)
    B = dists<r+0.5
    # print (B)
    return B.astype(int) * code

def Pad_zone_unit_circle_position(a,b,n,m):
    # return orignal_points and Flag
    left = int(a-W_t)
    right = int(a+W_t)
    bottom = int(b+W_t) # note that this is different to the original coordinates
    top = int(b-W_t)
    left_margin_flag = left < 0
    top_margin_flag = top < 0
    if not left_margin_flag and right < n and bottom < m and not top_margin_flag:
        return True,left,right,bottom,top,left_margin_flag,top_margin_flag
    else:
        left = max(0,left)
        right = min(n,right)
        bottom = min(m,bottom)
        top = max(0,top)
        return False,left,right,bottom,top,left_margin_flag,top_margin_flag

def distance_calculation_for_pad_zone(list_of_points,center):
    return eDist(list_of_points,center)
dist_vec = np.vectorize(distance_calculation_for_pad_zone)
def Pad_zone_unit_MAP(buffer_code,tabu_code,a,b,MAP,n,m):
    flag,left,right,bottom,top,left_margin_flag,top_margin_flag = Pad_zone_unit_circle_position(a,b,n,m)
    if flag:
        MAP_out = unit_circle_vectorized(W_t,buffer_code)
        # tabu square
        MAP_out[tabu_left:tabu_right,tabu_top:tabu_bottom] = tabu_code
        return MAP_out,left,right,bottom,top
    else:
        # computational intensive
        n_rows = bottom-top+1
        n_cols = right-left+1
        MAP_out = np.full((n_rows,n_cols),0,dtype=int)
        center_x = int_W_t
        center_y = int_W_t
        if left_margin_flag:
            center_x = left
        if top_margin_flag:
            center_y = top
        # iterative every points
        list_of_points = []
        
        for x in range(n_rows):
            for y in range(n_cols):
                list_of_points.append([x,y])
        dist_list = dist_vec(list_of_points,[center_x,center_y])

        for i in range(len(list_of_points))
            x = list_of_points[i][0] 
            y = list_of_points[i][1] 
            if dist_list[i] < W_t:
                MAP_out[x,y] = buffer_code
            if dist_list[i] < W:
                MAP_out[x,y] = tabu_code
        return MAP_out,left,right,bottom,top



def Pad_zone_unit(arg_list):
    # this is a computational unit for Pad_zone in order to do parallel computing
    # input: points a,b and Map
    a = arg_list[0]
    b = arg_list[1]
    status = arg_list[2]
    if status == 10:
        return Pad_zone_unit_MAP(8,9,a,b,n,m)
    if status == 4:
        return Pad_zone_unit_MAP(8,3,a,b,n,m)
    if status == 6:
        return Pad_zone_unit_MAP(8,7,a,b,n,m)



def Pad_zone(MAP): # 8 --- The zones that the Segments cannot intersect
    #for a,b in MAP_obs_ind:
    for a in range(0,MAP.shape[0]):
        for b in range(0,MAP.shape[1]):
            if MAP[a,b] == 10:
                for aa in range(max(0,a-W_t),min(MAP.shape[0],a+W_t)):
                    for bb in range(max(0,b-W_t),min(MAP.shape[1],b+W_t)):
                        if MAP[aa,bb] == 0 and W_t*W_t >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 8
                        if (MAP[aa,bb] == 0 or MAP[aa,bb] == 8) and W*W >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 9
            elif MAP[a,b] == 4:
                for aa in range(max(0,a-W_t),min(MAP.shape[0],a+W_t)):
                    for bb in range(max(0,b-W_t),min(MAP.shape[1],b+W_t)):
                        if MAP[aa,bb] == 0 and W_t*W_t >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 8
                        if (MAP[aa,bb] == 0 or MAP[aa,bb] == 8) and W*W >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 3
            elif MAP[a,b] == 6:
                for aa in range(max(0,a-W_t),min(MAP.shape[0],a+W_t)):
                    for bb in range(max(0,b-W_t),min(MAP.shape[1],b+W_t)):
                        if MAP[aa,bb] == 0 and W_t*W_t >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 8
                        if (MAP[aa,bb] == 0 or MAP[aa,bb] == 8) and W*W >= (aa-a)**2+(bb-b)**2:
                            MAP[aa,bb] = 7
    zone_obs = np.array([(np.where(MAP == 9)[0].astype(float))/scale - 4.5, (np.where(MAP == 9)[1].astype(float))/scale - 4.5])
    zone_left = np.array([(np.where(MAP == 3)[0].astype(float))/scale - 4.5, (np.where(MAP == 3)[1].astype(float))/scale - 4.5])
    zone_right = np.array([(np.where(MAP == 7)[0].astype(float))/scale - 4.5, (np.where(MAP == 7)[1].astype(float))/scale - 4.5])
    return MAP,zone_obs,zone_left,zone_right

def IsTwoLine(Ex_line): # Check if there are two lines
    if Ex_line == 1: # No left line
        for i in range(0,4*scale): 
            MAP[int(4.5*scale + i),0] = 4
    elif Ex_line == 2: # No right line
        for i in range(0,4*scale):
            MAP[int(4.5*scale + i),int(2*4.5*scale)-1] = 6
    elif Ex_line == 0: # No line
        for i in range(0,4*scale):
            MAP[int(4.5*scale + i),0] = 4
            MAP[int(4.5*scale + i),int(2*4.5*scale)-1] = 6


def Midpoint():
    " Contour_r --- the left contour of the connected part of the right line"
    ""
    global right_line
    global stop
    mid = np.zeros((right_line.shape[1]))
    "Find the left contour of the connected part of the right line"
    Contour_r, Situation = Contour(MAP, right_line, scale)
    for i in range(0,Contour_r.shape[0]):
        a = Contour_r[i,0] #int((right_line[0,i]+4.5)*scale)
        b = Contour_r[i,1] #int((right_line[1,i]+4.5)*scale)
        #mid[i] = (min(np.where(MAP[a,:] == 7))).astype(float)/scale - 4.5
        k = b
        while True:
            if MAP[a,k] == 0: 
                right = k
                p = k
                while True:
                    if MAP[a,p] == 4 or MAP[a,p] == 3 or MAP[a,p] == 10 or MAP[a,p] == 9:
                        left = p+1
                        break
                    p = p-1                
                mid[i] = (left+right)/2  #float(k)/scale-4.5
                break
            elif MAP[a,k] == 4 or MAP[a,k] == 3:
                mid[i] = k+1  #float(k)/scale-4.5
                stop = 1
                break
            k = k-1   
    return mid/scale-4.5,mid
    
def LOS(mid):
    mid_y = mid
    mid_x = (right_line[0,:]+4.5)*scale
    n = right_line.shape[1]
    LOS_x = 4.5*scale+1
    LOS_y = 4.5*scale
    for i in range(0,n): # The i-th midpoint
        d_mid_x = mid_x[i]-4.5*scale
        d_mid_y = mid_y[i]-4.5*scale
        length = mt.ceil(mt.sqrt(d_mid_x*d_mid_x+d_mid_y*d_mid_y)) # The length from the car to the i-th midpoint
        hit = 0 # Check if the obstacles are hit
        for j in range(0,length):
            if MAP[int(round(j*d_mid_x/length+4.5*scale)), int(round(j*d_mid_y/length+4.5*scale))] != 0 and MAP[int(round(j*d_mid_x/length+4.5*scale)), int(round(j*d_mid_y/length+4.5*scale))] != 5:
                hit = 1
                break
        if hit == 0:
            LOS_x = mid_x[i]
            LOS_y = mid_y[i]
    LOS_point = np.array([LOS_x,LOS_y])/scale-4.5
    LOS_angle = mt.atan2(LOS_y,LOS_x)
    return LOS_angle, LOS_point




def OnClick(event):
    global i
    i = 2
    
def Contour(MAP, right_line, scale):
    "right_line (2x? float) --- The positions of the points on the right line in the b-frame"
    "MAP (int8) --- The discret map in the translated b-frame"
    "Contour_r (?x2 int8) --- the left contour of the connected part of the right line"
    "scale --- the transform scale from the b-frame to the discret b-frame"
    n_r = right_line.shape[1] # The number of points on right_line
    x_r = ((right_line[0,:]+4.5)*scale).astype(int)    # The discrete x coordinates of the points on the right line
    y_r = ((right_line[1,:]+4.5)*scale).astype(int)    # The discrete y coordinates of the points on the right line
    n = x_r[n_r-1]-x_r[0] # The vertical length of the right line
    Contour_r = np.zeros((n, 2), dtype=int) # Initialize Contour_r 
    # Set the Contour as the right line
    for i in range(0,n):
        Contour_r[i,0] = x_r[i]
        Contour_r[i,1] = y_r[i]
    "Find the left contour point of the right line at the nearest position"
    a = x_r[0] # [a,b] will be used to move along the contour
    b = y_r[0]
    #Contour_r[0,0] = x_r[0]
    Block = 1 # If the course if blocked, set Situation = Block
    Situation = not Block # Course situation
    while b > 0:
        if MAP[x_r[0], b] == 0 or MAP[x_r[0], b] == 5: # If reach the free space
            b = b+1
            break
        elif MAP[x_r[0], b] == 4 or MAP[x_r[0], b] == 3: # If touch the left line and its buffer zone
            b = b+1
            Situation = Block # The course is blocked
            break
        b = b-1 
    Contour_r[0,1] = b
    "Now move the contour point along the boundary of the connected buffer zones"
    "Set rotation matrices"
    A1 = np.matrix([[0,1],[-1,0]], dtype='int8')
    A2 = np.matrix([[1,1],[-1,1]], dtype='int8')
    A3 = np.matrix([[1,0],[0,1]], dtype='int8')
    A4 = np.matrix([[1,-1],[1,1]], dtype='int8')
    A5 = np.matrix([[0,-1],[1,0]], dtype='int8')
    # Initialize the Last move of [a,b]
    LastMove = np.array([[1],[0]], dtype='int8')
    
    while (a - x_r[0] < n) and (Situation != Block): #(a <= x_r[n-1])
        # Calculate the next posiiton candidates : NextPos (This is a numpy Matrix)
        NextMove1 = np.sign(A1*LastMove)
        NextMove2 = np.sign(A2*LastMove)
        NextMove3 = np.sign(A3*LastMove)
        NextMove4 = np.sign(A4*LastMove)
        NextMove5 = np.sign(A5*LastMove)
        NextPos1 = NextMove1 + np.array([[a],[b]], dtype='int8') 
        NextPos2 = NextMove2 + np.array([[a],[b]], dtype='int8') 
        NextPos3 = NextMove3 + np.array([[a],[b]], dtype='int8') 
        NextPos4 = NextMove4 + np.array([[a],[b]], dtype='int8') 
        NextPos5 = NextMove5 + np.array([[a],[b]], dtype='int8')
        if NextPos1[0,0] - x_r[0] < 0 or NextPos2[0,0] - x_r[0] < 0 or NextPos3[0,0] - x_r[0] < 0 or NextPos4[0,0] - x_r[0] < 0 or NextPos5[0,0] - x_r[0] < 0:
            Situation = Block
            break
        if MAP[NextPos1[0,0],NextPos1[1,0]] != 0 and MAP[NextPos1[0,0],NextPos1[1,0]] != 5:
            if MAP[NextPos1[0,0],NextPos1[1,0]] == 4 or MAP[NextPos1[0,0],NextPos1[1,0]] == 3:
                Situation = Block
                break
            else:
                a = NextPos1[0,0]
                b = NextPos1[1,0]
                LastMove = NextMove1
        elif MAP[NextPos2[0,0],NextPos2[1,0]] != 0 and MAP[NextPos2[0,0],NextPos2[1,0]] != 5:
            if MAP[NextPos2[0,0],NextPos2[1,0]] == 4 or MAP[NextPos2[0,0],NextPos2[1,0]] == 3:
                Situation = Block
                break
            else:
                a = NextPos2[0,0]
                b = NextPos2[1,0]
                LastMove = NextMove2
        elif MAP[NextPos3[0,0],NextPos3[1,0]] != 0 and MAP[NextPos3[0,0],NextPos3[1,0]] != 5:
            if MAP[NextPos3[0,0],NextPos3[1,0]] == 4 or MAP[NextPos3[0,0],NextPos3[1,0]] == 3:
                Situation = Block
                break
            else:
                a = NextPos3[0,0]
                b = NextPos3[1,0]
                LastMove = NextMove3
        elif MAP[NextPos4[0,0],NextPos4[1,0]] != 0 and MAP[NextPos4[0,0],NextPos4[1,0]] != 5:
            if MAP[NextPos4[0,0],NextPos4[1,0]] == 4 or MAP[NextPos4[0,0],NextPos4[1,0]] == 3:
                Situation = Block
                break
            else:
                a = NextPos4[0,0]
                b = NextPos4[1,0]
                LastMove = NextMove4
        elif MAP[NextPos5[0,0],NextPos5[1,0]] != 0 and MAP[NextPos5[0,0],NextPos5[1,0]] != 5:
            if MAP[NextPos5[0,0],NextPos5[1,0]] == 4 or MAP[NextPos5[0,0],NextPos5[1,0]] == 3:
                Situation = Block
                break
            else:
                a = NextPos5[0,0]
                b = NextPos5[1,0]
                LastMove = NextMove5
    
        # If the new contour point is on the left of the old one, then update the old one
        if Contour_r[a-x_r[0]-1,1] > b:
            Contour_r[a-x_r[0]-1,1] = b

    return Contour_r, Situation

def FeasiblePath(MAP, SegPoint, Go):
    Collision = 1 # Flag for collision
    Awareness = 0 # Aware if collision happens
    n_L = 6 #divide the car body into how many piecese 
    Forward = 1
    Stop = 0
    Reverse = -1
    "Initialize Go_next"
    Go_next = Go
    "Control gain"
    K_1 = 1
    K_2 = 20
    "Initialize the pose of the virtual car"
    x_v = np.zeros((100), dtype=float)
    y_v = np.zeros((100), dtype=float)
    psi_v = np.zeros((100), dtype=float)
    delta_v = np.zeros((100), dtype=float)
    "# of SegPoints"
    n = SegPoint.shape[0]
    "Initialize the reversed distance"
    l_rev = 0
    "Initialize the index of directional change"
    ind_change = np.array([])
    "Plan the path "
    i = 0  # Initialize "i", the index of segment
    # Initialize Segment_next
    if n>2:
        Segment_next = np.array([SegPoint[2,0]-SegPoint[1,0],SegPoint[2,1]-SegPoint[1,1]])
            
    for k in range(0,99):
        "The switching control law"
        "The i-th Segment"
        
        if i == n-2: # For the last segment, use LOS method
            psi_err = np.angle((mt.cos(psi_v[k]) + mt.sin(psi_v[k])*1j)/((SegPoint[n-1,0]-x_v[k])+(SegPoint[n-1,1]-y_v[k])*1j))
        else:
            Segment = np.array([SegPoint[i+1,0]-SegPoint[i,0],SegPoint[i+1,1]-SegPoint[i,1]])
            "The distance from the virtual car to the segment"
            Distance = np.linalg.det(np.array([[x_v[k]-SegPoint[i,0],y_v[k]-SegPoint[i,1]],[Segment[0],Segment[1]]]))/max(np.linalg.norm(Segment),0.01)
            psi_nom = K_1 * Distance * Go
            psi_err = np.angle((mt.cos(psi_v[k]) + mt.sin(psi_v[k])*1j)/(Segment[0]+Segment[1]*1j)) - psi_nom
            
        if abs(K_2 * psi_err) < delta_max: 
                delta_v[k] = -K_2 * psi_err * np.sign(Go)
        else:
                delta_v[k] = -np.sign(psi_err * Go) * delta_max
        x_v[k+1] = x_v[k] + step_size * mt.cos(psi_v[k]) * Go
        y_v[k+1] = y_v[k] + step_size * mt.sin(psi_v[k]) * Go
        psi_v[k+1] = psi_v[k] + step_size / L * mt.tan(delta_v[k]) * Go

        if i <n-2:
            Distance_next = np.linalg.det(np.array([[x_v[k+1]-SegPoint[i+1,0],y_v[k+1]-SegPoint[i+1,1]],[Segment_next[0],Segment_next[1]]]))/max(np.linalg.norm(Segment_next),0.01)
            "If the car is close enough to the next segment, switch to the next segment"
            if abs(Distance_next) < 0.2:
                Segment_next = np.array([SegPoint[i+2,0]-SegPoint[i+1,0],SegPoint[i+2,1]-SegPoint[i+1,1]])
                i = i+1
                
        "If the car arrives at the Target, finish the planning"
        if np.linalg.norm([SegPoint[n-1,0]-x_v[k+1],SegPoint[n-1,1]-y_v[k+1]]) < 0.2:
            break
        "Update the reversed distance"
        if Go == Reverse:
            l_rev = l_rev + step_size
        "Collision detection"
        dL = L/n_L*np.array([mt.cos(psi_v[k+1]),mt.sin(psi_v[k+1])])
        Awareness = not Collision
        for a in range(1,n_L+1):
            x_body = int(round(scale*(x_v[k+1]+a*dL[0]+4.5)))
            y_body = int(round(scale*(y_v[k+1]+a*dL[1]+4.5)))
            if MAP[y_body,x_body] != 0 and MAP[y_body,x_body] != 8: 
                Awareness = Collision
        "If collision occurs,"
        if Awareness == Collision:
            Awareness = not Collision
            "change the drving direction"
            if Go == Forward:
                Go = Reverse
            elif Go == Reverse:
                Go = Forward
                # Reset the reversed distance
                l_rev = 0
            "Do not move"
            x_v[k+1] = x_v[k-1]
            y_v[k+1] = y_v[k-1]
            psi_v[k+1] = psi_v[k-1] 
            if k <= 3:
                Go_next = not Go_next
            "Record the index of the directional change point"
            ind_change = np.append(ind_change,[k+1])
        
        "If planning the last segment and the LOS angle is larger than 75 degree, reverse the virtual car"
        if i == n-2 and abs(psi_err) >= 1.4:
            Go = Reverse
        
        "If the virtual car has reversed for 1.5m, or the angle is small enough"
        "go foward, reset the reversed distance l_rev"
        if l_rev >= 1.5 or abs(np.angle((mt.cos(psi_v[k]) + mt.sin(psi_v[k])*1j)/((SegPoint[i+1,0]-x_v[k])+(SegPoint[i+1,1]-y_v[k])*1j)))<0.5:
            Go == Forward
            l_rev = 0
            
    return x_v[0:k], y_v[0:k], psi_v[0:k], delta_v[0:k], ind_change, Go_next

def OpenAreaPlanning(MAP_lidar, Target, SafeDist, Go, Range_lidar):
    MAP_lidar_pad, zone_obs, zone_left,zone_right= Pad_zone(MAP_lidar)
    VisablePoint = VisSelection(MAP_lidar_pad, Range_lidar, SafeDist) # Select the visable points candidates
    SegPoint = SegGeneration(VisablePoint, MAP_lidar_pad, Target) # Get the path segements that connect to the target
    x_v, y_v, psi_v, delta_v, ind_change, Go_next = FeasiblePath(MAP_lidar_pad, SegPoint, Go)     # Path planning for each segements


    return VisablePoint.astype(float)/scale-4.5, zone_obs, zone_left,zone_right, SegPoint, x_v, y_v, psi_v, ind_change, Go_next
    
def VisSelection(MAP_lidar, Range_lidar, SafeDist):
    "Select the visable points candidates"
    VisablePoint = np.array([]) # Initialize the VisablePoint
    SafeDist = int(SafeDist*scale) # Translate the Safe Distance to the discrete b-frame
    
    for i in range(0,int((Range_lidar+4.5)*scale)):
        p_old = MAP_lidar[i,0] # p_old tells whether the last point is occupied or not
        
        for k in range(1,MAP_lidar.shape[1]):
            if (MAP_lidar[i,k] == 10 or MAP_lidar[i,k] == 9 or MAP_lidar[i,k] == 8) and p_old == 0 and k - SafeDist > 0:
                VisablePoint = np.append(VisablePoint,np.array([i,k-SafeDist]))
            if (p_old == 10 or p_old == 9 or p_old == 8) and MAP_lidar[i,k] == 0 and k + SafeDist < MAP_lidar.shape[1]:
                VisablePoint = np.append(VisablePoint,np.array([i,k+SafeDist]))
            p_old = MAP_lidar[i,k]
    #length = int(VisablePoint.shape[0]/2)
    VisablePoint = VisablePoint.reshape(-1,2)
    return VisablePoint

def  SegGeneration(VisablePoint, MAP_lidar_pad, Target):
    Vis_y = VisablePoint[:,1]
    Vis_y = np.append(Vis_y,Target[1])
    Vis_x = VisablePoint[:,0]
    Vis_x = np.append(Vis_x,Target[0]) # Append the target to the VisablePoint list
    n = VisablePoint.shape[0]+1
    Seg_x = 4.5*scale
    Seg_y = 4.5*scale
    SegPoint = np.array([0,0]) # Initialize the SegPoint
    while True:
        DistToTgt = scale*20 # Initialize the distance from the VisablePoint_i to the Target
        NoPass = 1 # Flag to show if can pass from the current SegPoint
        for i in range(0,n): # The i-th VisablePoint
            d_Vis_x = Vis_x[i]-Seg_x
            d_Vis_y = Vis_y[i]-Seg_y
            length = mt.ceil(mt.sqrt(d_Vis_x*d_Vis_x+d_Vis_y*d_Vis_y)) # The length from the car to the i-th midpoint
            hit = 0 # Check if the obstacles are hit
            for j in range(0,length):
                if MAP_lidar_pad[int(round(j*d_Vis_x/length+Seg_x)), int(round(j*d_Vis_y/length+Seg_y))] != 0:
                    hit = 1
                    break
            if hit == 0 and mt.sqrt((Vis_x[i]-Target[0])**2 + (Vis_y[i]-Target[1])**2)<DistToTgt:
                Seg_x_new = Vis_x[i]
                Seg_y_new = Vis_y[i]
                DistToTgt = mt.sqrt((Vis_x[i]-Target[0])**2 + (Vis_y[i]-Target[1])**2)
                NoPass = 0
        if Seg_x_new == Target[0] and Seg_y_new == Target[1]: # If the target is reached, break
            SegPoint = np.append(SegPoint,np.array([Seg_x_new,Seg_y_new])/scale-4.5)
            break
        if NoPass == 1: # If No Pass from the current SegPoint
            SegPoint = SegPoint[:-2] # Remove the current SegPoint from the list, and go back to the previous one
            Seg_x = SegPoint[SegPoint.shape[0]-2]
            Seg_y = SegPoint[SegPoint.shape[0]-1]
        else: # Otherwise add this one to the list
            SegPoint = np.append(SegPoint,np.array([Seg_x_new,Seg_y_new])/scale-4.5)
            Seg_x,Seg_y = Seg_x_new,Seg_y_new
        
    return SegPoint.reshape(-1,2)

def PathToTraj(x_v,y_v,psi_v,ind_change,Speed,MotionState):
    "Obtain the sensored state from HiQ"
    x = MotionState[:,0] # Inertial x
    y = MotionState[:,1] # Inertial y
    psi = MotionState[:,2] # Inertial psi
    u = MotionState[:,3] # Longitudinal speed
    "Define the speed level"
    Stop = 0
    Slow = 1
    Medium = 2
    Fast = 3
    "Define the speed-changing distance"
    Dist_Stop = 0.1
    Dist_Slow = 0.8
    Dist_Medium = 1.6
    "Initialize time"
    t = 0
    "Initialize trajectory"
    x_bar = np.zeros((150), dtype=float)
    y_bar = np.zeros((150), dtype=float)
    psi_bar = np.zeros((150), dtype=float)
    
    "# of directional change points, corresponding to the traveled distance"
    n_change = ind_change.shape[0]
    i_change = 0 # Initialize the change point that will be arrived
    
    "Convert the path to a trajectory in the b-frame"
    
    #for i in range(0,x_v.shape[0]):
    i = 0
    k = 0
    "Traveled distance"
    l_Traj = 0
    while True:
        "The speed is taken as u_bar = min{Speed from CogStateMachine, Speed selection by Path}"
        "First figure out the speed according to the distance to the next change point"
        if n_change == 0:
            Speed_path = Fast # If no directional change point
        else:
            for a in range(0,n_change):
                if ind_change[a] > k:
                    break
            Dist_change = (ind_change[a] - k)*step_size # Distance to the next change point    
            if Dist_change <= 0.1:
                Speed_path = Stop
            elif Dist_change <= 0.8:
                Speed_path = Slow
            elif Dist_change <= 1.6:
                Speed_path = Medium
            else:
                Speed_path = Fast
        
        Speed_bar = min(Speed, Speed_path) # Selected Speed level
        
        if Speed_bar == Stop: # Obtain the selected u_bar
            u_bar = 0
        elif Speed_bar == Slow:
            u_bar = 0.5
        elif Speed_bar == Medium:
            u_bar = 1.2
        elif Speed_bar == Fast:
            u_bar = 2
        
        l_Traj = l_Traj + u_bar*Ts # Update traveled distance
        k = np.floor(l_Traj/step_size) # Arrive at the k-th path point
        residue = l_Traj - x_v[k] # 
        ratio = residue/step_size
        x_bar[i+1] = x_v[k] + (x_v[k+1] - x_v[k])* ratio
        y_bar[i+1] = x_v[k] + (y_v[k+1] - y_v[k])*ratio
        psi_bar[i+1] = psi_v[k]*(1-ratio)+psi_v[k+1]*ratio
        i = i+1
        if i == 111:
            break
    "Convert from b-frame to the n-frame"
    psi_bar = psi_v[101:110] + psi
    x_bar = mt.cos(psi)*x_v[101:110] - mt.sin(psi)*y_v[101:110] + x
    y_bar = mt.sin(psi)*x_v[101:110] + mt.cos(psi)*y_v[101:110] + y
    
    Trajectory =  np.vstack([x_bar,y_bar,psi_bar])
    
    return Trajectory
    
    
    

def Draw_planning(points,zone_obs,zone_left,zone_right,mid_draw,LOS_point, x_v, y_v, interval): #
    plt.plot(points[:,1],points[:,0],'ko',linewidth=1, markersize=1)
    plt.plot(zone_obs[1,:],zone_obs[0,:],'go',linewidth=1, markersize=1)
    plt.plot(left_line[1,:],left_line[0,:],'r',linewidth=2)
    plt.plot(zone_left[1,:],zone_left[0,:],'o',color='lightpink',linewidth=1, markersize=1)
    plt.plot(right_line[1,:],right_line[0,:],'b',linewidth=2)
    plt.plot(zone_right[1,:],zone_right[0,:],'o',color='lightblue',linewidth=1, markersize=1)
    # plt.plot(mid_line[1,:],mid_line[0,:],'y--',linewidth=2)
    
    plt.plot(mid_draw,right_line[0,:],'mo',linewidth=1, markersize=1)
    
    plt.plot([0,LOS_point[1]],[0,LOS_point[0]],'c',linewidth=1, markersize=1)
    
    plt.plot(y_v,x_v,'y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)  
    plt.draw()
    plt.pause(interval)
    plt.gcf().clear()
    
def Draw_planning_open(zone_obs, VisiblePoint, SegPoint, x_v, y_v, psi_v, interval): #
    #plt.plot(points[:,1],points[:,0],'ko',linewidth=1, markersize=1)
    plt.plot(zone_obs[1,:],zone_obs[0,:],'go',linewidth=1, markersize=1)
    
    # plt.plot(mid_line[1,:],mid_line[0,:],'y--',linewidth=2)
    
    #plt.plot(mid_draw,right_line[0,:],'mo',linewidth=1, markersize=1)
    
    plt.plot(SegPoint[:,1],SegPoint[:,0],'c',linewidth=1, markersize=1)
    plt.plot(VisiblePoint[:,1],VisiblePoint[:,0],'mo',linewidth=1, markersize=1)
    plt.plot(point_lidar[1,:],point_lidar[0,:],'ko',linewidth=1, markersize=1)
    plt.plot(y_v,x_v,'y')
    plt.plot(y_v+L*np.sin(psi_v),x_v+L*np.cos(psi_v),'r')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)  
    plt.draw()
    plt.pause(interval)
    plt.gcf().clear()
   
cid_up = fig.canvas.mpl_connect('button_press_event', OnClick)

i = 1   

while i == 1:
    stop = 0 # Clear the stop signal
    #points = ReadLidar()
    #IsTwoLine(Ex_line)
    #MAP_pad, zone_obs, zone_left,zone_right= Pad_zone(MAP)
    #mid_draw, mid = Midpoint()
    #LOS_angle, LOS_point = LOS(mid)
    #x_v, y_v, psi, delta = FeasiblePath(MAP, LOS_point, 1)
    #Range, Speed, Go, Light = CogStateMachine(SonarMode, RearVisionMode, Waypoint)
    VisiblePoint,zone_obs, zone_left,zone_right, SegPoint, x_v, y_v, psi_v, ind_change, Go_next = OpenAreaPlanning(MAP_lidar, Target, 0.3, 1, 3)
    #MotionState = np.array([0,0,0,2])
    #Trajectory = PathToTraj(x_v,y_v,psi_v,ind_change, Speed,MotionState)
    #Draw_planning(points,zone_obs,zone_left,zone_right,mid_draw, LOS_point, x_v, y_v, 0.01)#
    Draw_planning_open(zone_obs, VisiblePoint, SegPoint, x_v, y_v, psi_v, 0.01)
    # if kbhit():
    #     break
    
lidar.stop()
#lidar.stop_motor()
lidar.disconnect()

#if __name__ == '__main__':
#    run()