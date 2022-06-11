import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from utils import plot_line_segments, line_line_intersection

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.
        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        state_dim = len(self.x_init)
        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters + 1, state_dim))
        V[0,:] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])
        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters + 1, dtype=int)
        success = False
        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.
        ## Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!
        #   - the order in which you pass in arguments to steer_towards and is_free_motion is important
        ########## Code starts here ##########

        for k in range(max_iters):
            #uniform random sample
            z = np.random.uniform(0,1)
            if z < goal_bias:
                x_rand = self.x_goal
            else:
                x_rand = np.random.uniform(self.statespace_lo,self.statespace_hi)

            x_near_index = self.find_nearest(V[range(n)], x_rand)
            x_near = V[x_near_index]

            x_new = self.steer_towards(x_near, x_rand, eps)

            if self.is_free_motion(self.obstacles, x_near, x_new):
                V[n] = x_new
                P[n] = x_near_index
                n = n + 1
                if (x_new == self.x_goal).all():
                    self.path = [V[n-1]] #np.zeros((1,state_dim))
                    x_current = n - 1
                    #self.path[0] = V[n-1]
                    for i in range(n):
                        if x_current == 0:
                            break
                        self.path.insert(0,V[P[x_current]])
                        x_current = P[x_current]
                    success = True
                    break

        ########## Code ends here ##########

        plt.figure()
        self.plot_problem()
        self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2, label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=4, label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter(V[:n,0], V[:n,1])
        else:
            print("Solution not found!")

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        success = False
        while not(success):
            success = True
            for x in range(1,len(self.path)-1):
                if (self.is_free_motion(self.obstacles, self.path[x-1], self.path[x+1])):
                    if not((self.path[x] == self.x_goal).all()) and not((self.path[x] == self.x_init).all()): 
                        self.path.pop(x)
                        success = False
                        break


        ########## Code ends here ##########

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        # Hint: This should take one line.
        min_dist = V[0]-x
        min_V_index = 0
        for i in range(len(V)):
            dist = V[i]-x
            norm_min_dist = np.linalg.norm(min_dist)
            norm_dist = np.linalg.norm(dist)
            if norm_dist < norm_min_dist:
                min_dist = dist
                min_V_index = i
        return min_V_index
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        # Hint: This should take one line.
        dist = np.linalg.norm(x1-x2)
        if dist < eps:
            x_new = x2
        else:
            x_new = x1 + (x2-x1)*eps/dist
        return x_new
########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRT(RRT):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use the dubins package at
    https://github.com/AndrewWalker/pydubins/blob/master/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions d_path = dubins.shortest_path and 
    functions of the path such as d_path.sample_many and d_path.path_length (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def find_nearest(self, V, x):
        # Consult function specification in parent (RRT) class.
        # HINT: You may find the functions dubins.shortest_path() and path_length() useful
        # HINT: The order of arguments for dubins.shortest_path() is important for DubinsRRT.
        import dubins
        ########## Code starts here ##########
        min_dist = dubins.shortest_path(V[0],x,self.turning_radius)
        min_V_index = 0
        for i in range(len(V)):
            this_dist = dubins.shortest_path(V[i],x,self.turning_radius)
            min_path = min_dist.path_length()
            this_path = this_dist.path_length()
            if this_path < min_path:
                min_dist = this_dist
                min_V_index = i
        return min_V_index
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        import dubins
        """
        A subtle issue: if you use d_path.sample_many to return the point
        at distance eps along the path from x to y, use a turning radius
        slightly larger than self.turning_radius
        (i.e., 1.001*self.turning_radius). Without this hack,
        d_path.sample_many might return a point that can't quite get to in
        distance eps (using self.turning_radius) due to numerical precision
        issues.
        """
        # HINT: You may find the functions dubins.shortest_path(), d_path.path_length(), and d_path.sample_many() useful
        ########## Code starts here ##########
        d_path = dubins.shortest_path(x1,x2,1.001*self.turning_radius)
        path_arc_length = d_path.path_length()
        if path_arc_length > eps:
            many_path = d_path.sample_many(0.1)
            pose = many_path[0]
            dist = many_path[1]
            for i in range(len(dist)):
                if dist[i] <= eps:
                    new_pose = np.array(pose[i])
                else:
                    #print("Done")
                    break      
        else: 
            new_pose = x2
        return new_pose
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        import dubins
        d_path = dubins.shortest_path(x1, x2, self.turning_radius)
        pts = d_path.sample_many(self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        import dubins
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                d_path = dubins.shortest_path(V[P[i],:], V[i,:], self.turning_radius)
                pts = d_path.sample_many(self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        import dubins
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            d_path = dubins.shortest_path(path[i], path[i+1], self.turning_radius)
            new_pts = d_path.sample_many(self.turning_radius*resolution)[0]
            pts.extend(new_pts)
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)


## TEST HARNES
'''
plt.rcParams['figure.figsize'] = [8, 8] # Change default figure size

MAZE = np.array([
    (( 5, 5), (-5, 5)),
    ((-5, 5), (-5,-5)),
    ((-5,-5), ( 5,-5)),
    (( 5,-5), ( 5, 5)),
    ((-3,-3), (-3,-1)),
    ((-3,-3), (-1,-3)),
    (( 3, 3), ( 3, 1)),
    (( 3, 3), ( 1, 3)),
    (( 1,-1), ( 3,-1)),
    (( 3,-1), ( 3,-3)),
    ((-1, 1), (-3, 1)),
    ((-3, 1), (-3, 3)),
    ((-1,-1), ( 1,-3)),
    ((-1, 5), (-1, 2)),
    (( 0, 0), ( 1, 1))
])

# try changing these!
x_init = [-4,-4] # reset to [-4,-4] when saving results for submission
x_goal = [4,4] # reset to [4,4] when saving results for submission

grrt = GeometricRRT([-5,-5], [5,5], x_init, x_goal, MAZE)
grrt.solve(1.0, 2000)

grrt.solve(1.0, 2000, shortcut=True)

x_init = [-4,-4,0]
x_goal = [4,4,np.pi/2]

drrt = DubinsRRT([-5,-5,0], [5,5,2*np.pi], x_init, x_goal, MAZE, .5)
drrt.solve(1.0, 1000, shortcut=True)

'''
