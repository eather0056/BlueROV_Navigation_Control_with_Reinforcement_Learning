import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class RunningStat(object):
    """
    Running statistics calculator for mean and variance.

    This class computes running estimates of mean and variance for three different data arrays:
    image depth, goal, and ray.

    Attributes:
        _n (int): Number of data points seen so far.
        _M_img_depth (numpy.ndarray): Running mean for image depth data.
        _S_img_depth (numpy.ndarray): Running sum of squares for image depth data.
        _M_goal (numpy.ndarray): Running mean for goal data.
        _S_goal (numpy.ndarray): Running sum of squares for goal data.
        _M_ray (numpy.ndarray): Running mean for ray data.
        _S_ray (numpy.ndarray): Running sum of squares for ray data.

    Methods:
        __init__: Initializes the RunningStat object with given shapes.
        push: Pushes new data points and updates running statistics.
    """
    def __init__(self, shape_img_depth, shape_goal, shape_ray):
        """
        Initializes the RunningStat object with given shapes.
        Args:
            shape_img_depth (tuple): Shape of the image depth data.
            shape_goal (tuple): Shape of the goal data.
            shape_ray (tuple): Shape of the ray data.
        """
        self._n = 0
        self._M_img_depth = np.zeros(shape_img_depth)
        self._S_img_depth = np.zeros(shape_img_depth)
        self._M_goal = np.zeros(shape_goal)
        self._S_goal = np.zeros(shape_goal)
        self._M_ray = np.zeros(shape_ray)
        self._S_ray = np.zeros(shape_ray)

    def push(self, img_depth, goal, ray):
        """
        Pushes new data points and updates running statistics.
        Args:
            img_depth (array_like): Image depth data.
            goal (array_like): Goal data.
            ray (array_like): Ray data.
        """

        # Convert input data to numpy arrays
        img_depth = np.asarray(img_depth)
        goal = np.asarray(goal)
        ray = np.asarray(ray)

        # Check if input shapes match with running statistics shapes
        assert img_depth.shape == self._M_img_depth.shape and\
               goal.shape == self._M_goal.shape and\
            ray.shape == self._M_ray.shape
        
        # Increment the count of data points seen
        self._n += 1

        # If it is the first data point, set running means to the input data
        if self._n == 1:
            self._M_img_depth[...] = img_depth
            self._M_goal[...] = goal
            self._M_ray[...] = ray

        # Otherwise, update running means and sum of squares
        else:
            oldM_img_depth = self._M_img_depth.copy()
            oldM_goal = self._M_goal.copy()
            oldM_ray = self._M_ray.copy()
            
            # Update running means
            self._M_img_depth[...] = oldM_img_depth + (img_depth - oldM_img_depth) / self._n
            self._M_ray[...] = oldM_ray + (ray - oldM_ray) / self._n
            self._M_goal[...] = oldM_goal + (goal - oldM_goal) / self._n

            # Update running sum of squares
            self._S_goal[...] = self._S_goal + (goal - oldM_goal) * (goal - self._M_goal)
            self._S_img_depth[...] = self._S_img_depth + (img_depth - oldM_img_depth) * (img_depth - self._M_img_depth)
            self._S_ray[...] = self._S_ray + (ray - oldM_ray) * (ray - self._M_ray)

    @property
    def n(self):
        """Returns the number of data points seen so far."""
        return self._n

    @property
    def mean_img_depth(self):
        """Returns the running mean for image depth data."""
        return self._M_img_depth

    @property
    def mean_goal(self):
        """Returns the running mean for goal data."""
        return self._M_goal

    @property
    def mean_ray(self):
        """Returns the running mean for ray data."""
        return self._M_ray

    @property
    def var_img_depth(self):
        """Returns the running variance for image depth data."""
        return self._S_img_depth / (self._n - 1) if self._n > 1 else np.square(self._M_img_depth)

    @property
    def var_goal(self):
        """Returns the running variance for goal data."""
        return self._S_goal / (self._n - 1) if self._n > 1 else np.square(self._M_goal)

    @property
    def var_ray(self):
        """Returns the running variance for ray data."""
        return self._S_ray / (self._n - 1) if self._n > 1 else np.square(self._M_ray)

    @property
    def std_img_depth(self):
        """Returns the running standard deviation for image depth data."""
        return np.sqrt(self.var_img_depth)

    @property
    def std_goal(self):
        """Returns the running standard deviation for goal data."""
        return np.sqrt(self.var_goal)

    @property
    def std_ray(self):
        """Returns the running standard deviation for ray data."""
        return np.sqrt(self.var_ray)

    @property
    def shape_img_depth(self):
        """Returns the shape of the image depth data."""
        return self._M_img_depth.shape

    @property
    def shape_goal(self):
        """Returns the shape of the goal data."""
        return self._M_goal.shape

    @property
    def shape_ray(self):
        """Returns the shape of the ray data."""
        return self._M_ray.shape
    
class ZFilter:
    """
    Z-score normalization filter for input data.

    This class provides functionality to normalize input data using the Z-score formula:
    z = (x - mean) / std

    Attributes:
        demean (bool): Flag indicating whether to demean the data (subtract mean).
        destd (bool): Flag indicating whether to destandardize the data (divide by standard deviation).
        clip (float): Value for clipping the normalized data to a specific range.
        rs (RunningStat): RunningStat object to store running estimates of mean and standard deviation.
        fix (bool): Flag indicating whether to update the running estimates.

    Methods:
        __init__: Initializes the ZFilter with specified parameters.
        __call__: Normalizes input data using running estimates of mean and standard deviation.

    """

    def __init__(self, shape_img_depth, shape_goal, shape_ray, demean=True, destd=True, clip=10.0):
        """
        Args:
            shape_img_depth (tuple): Shape of the image depth data.
            shape_goal (tuple): Shape of the goal data.
            shape_ray (tuple): Shape of the ray data.
            demean (bool, optional): Flag indicating whether to demean the data (subtract mean). Default is True.
            destd (bool, optional): Flag indicating whether to destandardize the data (divide by standard deviation). Default is True.
            clip (float, optional): Value for clipping the normalized data to a specific range. Default is 10.0.
        """
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape_img_depth, shape_goal, shape_ray)
        self.fix = False

    def __call__(self, img_depth, goal, ray, update=True):
        """
        Normalizes input data using running estimates of mean and standard deviation.
        Args:
            img_depth (array_like): Image depth data.
            goal (array_like): Goal data.
            ray (array_like): Ray data.
            update (bool, optional): Flag indicating whether to update the running estimates. Default is True.
        Returns:
            tuple: Normalized img_depth, goal, and ray data.
        """
        # Update running estimates if required
        if update and not self.fix:
            self.rs.push(img_depth, goal, ray)
        # Demean the data if enabled
        if self.demean:
            img_depth = img_depth - self.rs.mean_img_depth
            goal = goal - self.rs.mean_goal
            ray = ray - self.rs.mean_ray
        # Destandardize the data if enabled
        if self.destd:
            img_depth = img_depth / (self.rs.std_img_depth + 1e-8)
            goal = goal / (self.rs.std_goal + 1e-8)
            ray = ray / (self.rs.std_ray + 1e-8)
        # Clip the normalized data if a clipping value is specified
        if self.clip:
            img_depth = np.clip(img_depth, -self.clip, self.clip)
            goal = np.clip(goal, -self.clip, self.clip)
            ray = np.clip(ray, -self.clip, self.clip)
        return img_depth, goal, ray

