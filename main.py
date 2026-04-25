import numpy as np
import numpy.linalg as linalg
import cv2

def makeUnit(x):
    return x / linalg.norm(x)

def xParV(x, v):
    return np.dot(x, v) / np.dot(v, v) * v

def xPerpV(x, v):
    return x - xParV(x, v)

def xProjectV(x, v):
    par = xParV(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}

def rotateAbout(a, b, theta):
    proj = xProjectV(a, b)
    w = np.cross(b, proj['perp'])
    return (proj['par'] +
            proj['perp'] * np.cos(theta) +
            linalg.norm(proj['perp']) * makeUnit(w) * np.sin(theta))

voxels = np.zeros((20,10,10))

fov = 40

zenith = np.array([0,0,1])
north = np.array([1,0,0])
axis = np.array([0,1,0])

targetVoxel = [0]*8
for v in range(0,4):
    if v == 0:
        img = cv2.imread('vid1f.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = img.shape[0]
        columns = img.shape[1]
        cameracoord = np.array([-10,0,0])
        alt = 45
        az = 0
    if v == 1:
        img = cv2.imread('vid2f.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = img.shape[0]
        columns = img.shape[1]
        cameracoord = np.array([10,0,0])
        alt = 135
        az = 180
    if v == 2:
        img = cv2.imread('vid3f.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = img.shape[0]
        columns = img.shape[1]
        cameracoord = np.array([0,-10,0])
        alt = 45
        az = 270
    if v == 3:
        img = cv2.imread('vid4f.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = img.shape[0]
        columns = img.shape[1]
        cameracoord = np.array([0,10,0])
        alt = 45
        az = 90
    for i in range(0,rows):
        for j in range(0,columns):
            if(img[i][j] > 0):
                a = fov/columns
                b = fov/rows
                x = columns/2 - j
                y = rows/2 - i
                naz = az - x*a
                nalt = alt - y*b
                vx = rotateAbout(north, axis, nalt)
                vy = rotateAbout(north, zenith, naz)
                dv = vx + vy
                for z in range(10,20):
                    c = cameracoord+dv*z
                    if(c[1] > -0.1 and c[1] < 0.1 and c[2] < 10.1 and c[2] > 9.9 and c[0] < 5 and c[0] > -3):
                        print(c)
                        if(c[0]<-2):
                            targetVoxel[0]+=1
                        elif(c[0]<-1):
                            targetVoxel[1]+=1
                        elif(c[0]<0):
                            targetVoxel[2]+=1
                        elif(c[0]<1):
                            targetVoxel[3]+=1
                        elif(c[0]<2):
                            targetVoxel[4]+=1
                        elif(c[0]<3):
                            targetVoxel[5]+=1
                        elif(c[0]<4):
                            targetVoxel[6]+=1
                        elif(c[0]<5):
                            targetVoxel[7]+=1
print("voxel matches = ")
print(targetVoxel)
print("coordinates")
y=((-0.1)+0.1)/2
z=(10.1+9.9)/2
for i in range(-3,5):
    x=(i+(i+1))/2
    c = np.array([x,y,z])
    print(c)