#Potential Field Approach
#Used in a receding horizon method, taking first returned action each iteration
#discretises the world in 0.5x0.5m grid,  assumes an obstacle occupies 1 grid cell
#Paths are made up of actions moving 1 grid at a time like a King in Chess (8 options)

from __future__ import division # for Python 2.x compatibility
from PIL import Image
import numpy



def getAction(start,end,observation, size, return_path=False):
    #assumes grid is centred @ size*0.5,size*0.5 rather than 0,0

    cell_size = 0.5 #0.5m per cell

    src = [int((start[0] + size*0.5)/cell_size), int((start[1] + size*0.5)/cell_size)]
    goal = [int((end[0] + size*0.5)/cell_size), int((end[1] + size*0.5)/cell_size)]

    field_size = int(size/cell_size) #0.5m discretisation

    obstacles = []
    for obstacle in observation:
        new_x = int((obstacle.px + size*0.5)/cell_size)
        new_y = int((obstacle.py + size*0.5)/cell_size)
        obstacles.append([new_x, new_y])
    field = EuclidField((field_size, field_size) ,goal, obstacles)
    best_path = find_path(field, src, goal, 100)

    if return_path:
        return best_path
    try:
        vx = best_path[1][0]-best_path[0][0]
        vy = best_path[1][1]-best_path[0][1]    
        return (vx,vy)
    except:
        print('No action returned. Error in path:',best_path)
        return (0,0)


class EuclidField(object):
    p = 5.0
    @staticmethod
    def dist(x, y):
        return numpy.hypot(x[0]-y[0], x[1]-y[1])
    def __init__(self, size, dst, obstacles):
        w, h = size
        self.shape = (h, w)
        self.dst = dst
        self.obstacles = obstacles
    def __getitem__(self, q):
        i, j = q
        h, w = self.shape
        if not (i in range(h) and j in range(w)):
            raise IndexError
        base = self.dist(q, self.dst)
        k = 0.0
        p = self.p
        for obj in self.obstacles:
            dist_to_obj = self.dist(q, obj)
            if dist_to_obj <= p:
                k += 5.0 / (1+(dist_to_obj/2.0)**6)
        return (1.0 + k) * base**2
    def __array__(self):
        h, w = self.shape
        return numpy.array([[self[i, j] for j in range(w)] for i in range(h)])

def field_to_image(f):
    h, w = f.shape
    cached = numpy.array(f, copy=False)
    img = Image.new('RGB', (w, h))
    maxp = minp = 0.0
    for i in range(h):
        for j in range(w):
            val = cached[i, j]
            if not numpy.isinf(val):
                if val > maxp:
                    maxp = val
                elif val < minp:
                    minp = val
    px = img.load()
    for i in range(h):
        for j in range(w):
            val = cached[i, j]
            if numpy.isinf(val):
                px[j, i] = (255, 255, 255)
            elif numpy.abs(val) < 1e-9:
                px[j, i] = (0, 0, 0)
            elif val > 0:
                px[j, i] = (int(val / maxp * 255), 0, 0)
            else:
                px[j, i] = (0, 0, int(val / minp * 255))
    return img

def find_nextstep(f, src):
    h, w = f.shape
    i, j = src
    return min(
        ((i1, j1)
        for i1 in (i-1, i, i+1) if i1 in range(h)
        for j1 in (j-1, j, j+1) if j1 in range(w)
        ), key=lambda x: f[x]
    )

def find_path(f, src, dst=None, maxattempt=None):
    path = [src]
    h, w = f.shape
    if maxattempt is None:
        maxattempt = w*h
    while maxattempt > 0 and src != dst:
        maxattempt -= 1
        src = find_nextstep(f, src)
        path.append(src)
    return path

def draw_path(img, path):
    px = img.load()
    for i, j in path:
        px[j, i] = (0, 255, 0)
