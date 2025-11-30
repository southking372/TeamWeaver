import numpy as np
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

from class_def.SingleIntegrator import SingleIntegrator
from class_def.RTA import RTA

class Swarm:
    def __init__(self, **kwargs):
        self.robots = kwargs.get('robots', [])
        self.N = len(self.robots)
        
        self.L = kwargs.get('L', [])
        if self.N == 1:
            self.L = 0
        else:
            if not self.L:
                self.L = np.zeros((self.N, self.N))
            elif not isinstance(self.L, np.ndarray):
                if self.N < 3 and self.L == 'cycle':
                    self.L = 'line'
                    
                if self.L == 'line':
                    self.L = np.zeros((self.N, self.N))
                    for i in range(self.N):
                        self.L[i, i] = 2
                        if i > 0:
                            self.L[i, i-1] = -1
                        if i < self.N - 1:
                            self.L[i, i+1] = -1
                    self.L[0, 0] = 1
                    self.L[self.N-1, self.N-1] = 1
                elif self.L == 'cycle':
                    main_diag = np.ones(self.N) * 2
                    first_diag = np.ones(self.N-1) * -1
                    self.L = toeplitz(np.concatenate(([2], [-1], np.zeros(self.N-3), [-1])))
                elif self.L == 'complete':
                    self.L = self.N * np.eye(self.N) - np.ones((self.N, self.N))
            else:
                assert np.sqrt(self.L.size) == self.N, 'Laplacian is not a square matrix whose dimension is the number of robots.'
        
        self.environment = kwargs.get('environment', [])
        if len(self.environment) > 0:
            if np.linalg.norm(self.environment[:, -1] - self.environment[:, 0]) > 1e-3:
                self.environment = np.hstack((self.environment, self.environment[:, 0].reshape(-1, 1)))
        
        self.phi = kwargs.get('densityFunction', 'uniform')
        
        # Graphics handles
        self.hG = {
            'figure': None,
            'graph': [],
            'env': None,
            'density': None,
            'voronoiCells': None,
            'voronoiCentroids': None,
            'robots': None
        }

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    def getPoses(self):
        if not self.robots:
            return None
            
        d = self.robots[0].getPose().shape[0]
        q = np.zeros((d, self.N))
        
        for i in range(self.N):
            q[:, i] = self.robots[i].getPose()
        
        if d == 2:
            q = np.vstack((q, np.zeros((1, self.N))))
            
        return q
    
    def getNeighbors(self, idx):
        neighbors = np.where(self.L[idx, :] != 0)[0]
        return neighbors[neighbors != idx]
    
    def setPoses(self, q):
        for i in range(self.N):
            self.robots[i].setPose(q[:, i])
    
    def moveSingleIntegrators(self, v):
        for i in range(self.N):
            self.robots[i].moveSingleIntegrator(v[:, i])
    
    def moveUnicycles(self, v):
        for i in range(self.N):
            self.robots[i].moveUnicycle(v[:, i])
    
    def goToPoints(self, p, **kwargs):
        for i in range(self.N):
            self.robots[i].goToPoint(p[:, i], **kwargs)
    
    def coverageControl(self, *args):
        if not args:
            p = np.eye(2, 3) @ self.getPoses()
            Np = self.N
        else:
            p = args[0]
            Np = p.shape[1]
        
        # Check if we have any robot positions to work with
        if Np == 0:
            return np.zeros((2, 0)), np.zeros((1, 0)), [None] * 0
        
        # Mirror robots about environment boundary
        mirrored = self.mirrorRobotsAboutEnvironmentBoundary(p)
        
        # Process original points only if mirroring fails or is not needed
        if mirrored.size == 0:
            P = p
        else:
            P = np.hstack((p, mirrored))
        
        # Normalize coordinates to prevent numerical issues
        # Find the scale of the data
        data_scale = np.max(np.abs(P)) if P.size > 0 else 1.0
        
        # If data scale is too large or too small, normalize
        normalize = data_scale > 1e3 or data_scale < 1e-3
        if normalize:
            # Save the original points
            P_original = P.copy()
            # Normalize to reasonable range
            P = P / data_scale
            
        # Initialize results
        G = np.zeros((2, Np))
        A = np.zeros((1, Np))
        VC = [None] * Np
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(P.T)
            V = vor.vertices
            C = [[] for _ in range(len(vor.points))]
            
            # Construct Voronoi cells
            for i, region in enumerate(vor.regions):
                if -1 not in region and region:
                    point_idx = vor.point_region.tolist().index(i)
                    if point_idx < Np:
                        C[point_idx] = region
            
            # Handle infinite vertices
            V_inf = np.any(np.isinf(V), axis=1)
            max_env = np.max(np.abs(self.environment)) if len(self.environment) > 0 else 1000
            max_bound = 1e3 * max_env
            V[V_inf] = max_bound
            
            # If we normalized, scale vertices back
            if normalize:
                V = V * data_scale
            
            # Compute centroids for each cell
            for i in range(Np):
                if C[i]:
                    VC[i] = np.vstack((V[C[i], 0], V[C[i], 1]))
                    try:
                        if self.phi == 'uniform':
                            Gi, Ai = self.centroid_uniform(VC[i])
                        else:
                            Gi, Ai = self.centroid(VC[i])
                        G[:, i] = Gi
                        A[0, i] = Ai
                    except Exception as e:
                        try:
                            # Try alternative centroid calculation
                            points = VC[i].T
                            Gi = np.mean(points, axis=0)
                            G[:, i] = Gi
                            # Approximate area using convex hull
                            hull = ConvexHull(points)
                            A[0, i] = hull.area
                        except Exception as e2:
                            G[:, i] = p[:, i]
                            A[0, i] = 1.0
                else:
                    # If no valid cell, use robot position
                    G[:, i] = p[:, i]
                    A[0, i] = 1.0
        
        except Exception as e:
            # Fallback: use robot positions as their own centroids
            G = p
            A = np.ones((1, Np))
            
            # Create basic Voronoi cells as circles around robot positions
            radius = 1.0
            if len(self.environment) > 0:
                env_size = np.max(np.ptp(self.environment, axis=1))
                radius = env_size * 0.1
            
            for i in range(Np):
                theta = np.linspace(0, 2*np.pi, 12)
                x = p[0, i] + radius * np.cos(theta)
                y = p[1, i] + radius * np.sin(theta)
                VC[i] = np.vstack((x, y))
        
        return G, A, VC
    
    def evaluateCoverageCost(self, q, VC, *args):
        p = np.eye(2, 3) @ q
        c = 0
        
        if args:
            idx = args[0]
            VC = [VC[idx]]
            indices = [idx]
        else:
            indices = range(len(VC))
        
        for i, idx in enumerate(indices):
            if VC[i] is None:
                continue
                
            P = VC[i]
            xP = P[0, :]
            yP = P[1, :]
            
            if self.phi == 'uniform':
                def f(x, y):
                    return np.linalg.norm(p[:, idx] - np.array([x, y])) ** 2
            else:
                def f(x, y):
                    return np.linalg.norm(p[:, idx] - np.array([x, y])) ** 2 * self.phi(x, y)
            
            # Triangulation of Voronoi cell
            if len(xP) > 2:
                points = np.vstack((xP, yP)).T
                tri = Delaunay(points)
                
                ci = 0
                for simplex in tri.simplices:
                    triangle = P[:, simplex]
                    ci += self.intOfFOverT(f, 8, triangle)
                
                c += ci
        
        return c
    
    def plotFigure(self):
        plt.figure(figsize=(12, 10))
        plt.axis('equal')
        
        if len(self.environment) > 0:
            env_min_x, env_max_x = np.min(self.environment[0, :]), np.max(self.environment[0, :])
            env_min_y, env_max_y = np.min(self.environment[1, :]), np.max(self.environment[1, :])
            plt.axis([env_min_x-1, env_max_x+1, env_min_y-1, env_max_y+1])
        else:
            plt.axis([-1, 1, -1, 1])
        
        plt.axis('off')
        self.hG['figure'] = plt.gcf()
        
        return self.hG['figure']
    
    def plotRobots(self, *args, **kwargs):
        for i in range(self.N):
            self.robots[i].plotRobot(*args, **kwargs)
    
    def plotRobotsFast(self, q, *args, **kwargs):
        if self.hG['robots'] is None:
            self.hG['robots'] = plt.scatter(q[0, :], q[1, :], *args, **kwargs)
        else:
            self.hG['robots'].set_offsets(np.vstack((q[0, :], q[1, :])).T)
        
        plt.draw()
    
    def plotGraph(self, *args, **kwargs):
        if not args:
            kwargs = {'color': 'black', 'linewidth': 2}
        
        q = self.getPoses()
        
        if not self.hG['graph']:
            for i in range(self.N):
                for j in self.getNeighbors(i):
                    line, = plt.plot([q[0, i], q[0, j]], [q[1, i], q[1, j]], **kwargs)
                    self.hG['graph'].append(line)
        else:
            edge_counter = 0
            for i in range(self.N):
                for j in self.getNeighbors(i):
                    edge_counter += 1
                    self.hG['graph'][edge_counter-1].set_data([q[0, i], q[0, j]], [q[1, i], q[1, j]])
        
        plt.draw()
    
    def plotEnvironment(self, *args, **kwargs):
        if not args:
            kwargs = {'linewidth': 5, 'color': 'black'}
        
        if self.hG['env'] is None:
            self.hG['env'], = plt.plot(self.environment[0, :], self.environment[1, :], **kwargs)
        else:
            self.hG['env'].set_data(self.environment[0, :], self.environment[1, :])
        
        plt.draw()
    
    def plotDensity(self, *args, **kwargs):
        if self.phi != 'uniform' and len(self.environment) > 0:
            x_min, x_max = np.min(self.environment[0, :]), np.max(self.environment[0, :])
            y_min, y_max = np.min(self.environment[1, :]), np.max(self.environment[1, :])
            
            x = np.arange(x_min, x_max, 0.01)
            y = np.arange(y_min, y_max, 0.01)
            X, Y = np.meshgrid(x, y)
            
            # Apply density function to grid
            if callable(self.phi):
                Z = np.vectorize(self.phi)(X, Y)
            else:
                Z = np.ones_like(X)  # Default uniform density
            
            if not args:
                levels = np.linspace(np.min(Z), np.max(Z), 10)
                kwargs = {'levels': levels, 'linewidths': 2}
            
            if self.hG['density'] is None:
                self.hG['density'] = plt.contour(X, Y, Z, **kwargs)
            else:
                for coll in self.hG['density'].collections:
                    coll.remove()
                self.hG['density'] = plt.contour(X, Y, Z, **kwargs)
            
            # Fill outside of environment
            self.fillout(self.environment[0, :], self.environment[1, :], 
                        [x_min-1, x_max+1, y_min-1, y_max+1], 
                        facecolor=(0.94, 0.94, 0.94))
            
            plt.draw()
    
    def plotVoronoiCells(self, VC, *args, **kwargs):
        if not args:
            kwargs = {'color': 'black', 'linewidth': 2}
        
        if not VC:
            return
        
        # Construct a list of lines to plot
        lines_x = []
        lines_y = []
        
        for cell in VC:
            if cell is not None:
                # Add the cell vertices and connect back to first point
                x = np.append(cell[0, :], cell[0, 0])
                y = np.append(cell[1, :], cell[1, 0])
                
                # Add NaN to separate cells
                lines_x.extend(list(x) + [np.nan])
                lines_y.extend(list(y) + [np.nan])
        
        # Remove last NaN
        if lines_x:
            lines_x = lines_x[:-1]
            lines_y = lines_y[:-1]
        
        if self.hG['voronoiCells'] is None:
            self.hG['voronoiCells'], = plt.plot(lines_x, lines_y, **kwargs)
        else:
            self.hG['voronoiCells'].set_data(lines_x, lines_y)
        
        plt.draw()
    
    def plotCentroids(self, G, *args, **kwargs):
        if not args:
            kwargs = {'marker': '.', 'color': 'black', 'markersize': 10}
        
        if self.hG['voronoiCentroids'] is None:
            self.hG['voronoiCentroids'] = plt.plot(G[0, :], G[1, :], **kwargs)[0]
        else:
            self.hG['voronoiCentroids'].set_data(G[0, :], G[1, :])
        
        plt.draw()
    
    # Private methods
    def mirrorRobotsAboutEnvironmentBoundary(self, p):
        if len(self.environment) == 0:
            return np.array([])
            
        n_points = p.shape[1]
        n_sides = self.environment.shape[1] - 1
        mirrored_robots = np.zeros((2, n_points * n_sides))
        
        for i in range(n_points):
            point = p[:, i]
            
            for j in range(n_sides):
                side_start = self.environment[:, j]
                side_end = self.environment[:, j+1]
                side = side_end - side_start
                
                # Point relative to side start
                point_rel = point - side_start
                
                # Project point onto side line
                side_length_squared = np.sum(side**2)
                if side_length_squared < 1e-10:  # Avoid division by near-zero
                    continue
                    
                dot_product = np.dot(point_rel, side)
                length_of_projection = dot_product / side_length_squared
                
                # Projected point on the line
                projected_point = side_start + length_of_projection * side
                
                # Mirror the point
                idx = (i * n_sides) + j
                mirrored_robots[:, idx] = point - 2 * (point - projected_point)
        
        return mirrored_robots
    
    def centroid(self, P):
        if P is None or P.shape[1] < 3:  # Need at least 3 points for a polygon
            return np.zeros(2), 0
            
        if self.phi == 'uniform':
            # Compute centroid of a simple polygon
            n = P.shape[1]
            A = 0
            Cx = 0
            Cy = 0
            
            for i in range(n):
                j = (i + 1) % n
                cross_product = P[0, i] * P[1, j] - P[0, j] * P[1, i]
                A += cross_product
                Cx += (P[0, i] + P[0, j]) * cross_product
                Cy += (P[1, i] + P[1, j]) * cross_product
            
            A = A / 2
            if abs(A) < 1e-10:  # Avoid division by near-zero
                return np.zeros(2), 0
                
            Cx = Cx / (6 * A)
            Cy = Cy / (6 * A)
            
            # Area should be positive
            if A < 0:
                A = -A
                
            return np.array([Cx, Cy]), A
        else:
            # For non-uniform density
            xP = P[0, :]
            yP = P[1, :]
            
            def phi_A(x, y):
                return self.phi(x, y)
                
            def phi_Sx(x, y):
                return x * self.phi(x, y)
                
            def phi_Sy(x, y):
                return y * self.phi(x, y)
            
            # Triangulate the polygon
            points = np.vstack((xP, yP)).T
            # 加入扰动项试图修正Delaunay三角化的问题
            # points = np.vstack((xP + 1e-8*np.random.rand(4), yP + 1e-8*np.random.rand(4))).T
            
            # Add error handling for Delaunay triangulation
            try:
                # Try with default QHull options
                tri = Delaunay(points)
                trngltn = tri.simplices
            except Exception as e:
                try:
                    # If that fails, try with QJ option (joggled input to avoid precision errors)
                    tri = Delaunay(points, qhull_options='QJ')
                except Exception as e:
                    # If that also fails, try scaling the points to unit cube
                    try:
                        tri = Delaunay(points, qhull_options='QbB')
                        trngltn = tri.simplices
                    except Exception as e:
                        # If all attempts fail, use a simple centroid calculation as fallback
                        print(f"Warning: Delaunay triangulation failed. Using fallback method. Error: {e}")
                        n = P.shape[1]
                        x_sum = np.sum(P[0, :])
                        y_sum = np.sum(P[1, :])
                        # Calculate simple polygon area
                        A = 0
                        for i in range(n):
                            j = (i + 1) % n
                            A += P[0, i] * P[1, j] - P[0, j] * P[1, i]
                        A = abs(A) / 2
                        return np.array([x_sum/n, y_sum/n]), A
            
            A = 0
            Sx = 0
            Sy = 0
            
            for simplex in trngltn:
                triangle = P[:, simplex]
                A += self.intOfFOverT(phi_A, 8, triangle)
                Sx += self.intOfFOverT(phi_Sx, 8, triangle)
                Sy += self.intOfFOverT(phi_Sy, 8, triangle)
            
            # if abs(A) < 1e-10:  # Avoid division by near-zero
            #     return np.zeros(2), 0
            if A == 0:
                return np.zeros(2), 0
            return np.array([Sx, Sy]) / A, A
    
    def centroid_uniform(self, P):
        """计算均匀密度下多边形的质心
        
        Args:
            P: shape (2, n) 的数组，表示多边形的顶点坐标
            
        Returns:
            G: 质心坐标
            A: 面积
        """
        if P is None or P.shape[1] < 3:  # 至少需要3个点才能形成多边形
            return np.zeros(2), 0
            
        # 计算简单多边形的质心
        n = P.shape[1]
        A = 0
        Cx = 0
        Cy = 0
        
        for i in range(n):
            j = (i + 1) % n
            # 计算叉积
            cross_product = P[0, i] * P[1, j] - P[0, j] * P[1, i]
            A += cross_product
            Cx += (P[0, i] + P[0, j]) * cross_product
            Cy += (P[1, i] + P[1, j]) * cross_product
        
        A = A / 2
        if abs(A) < 1e-10:  # 避免除以接近零的数
            return np.zeros(2), 0
            
        Cx = Cx / (6 * A)
        Cy = Cy / (6 * A)
        
        # 面积应该为正
        if A < 0:
            A = -A
            
        return np.array([Cx, Cy]), A
    
    # Static methods
    @staticmethod
    def intOfFOverT(f, N, T):
        """
        Numerical integration of a function f over a triangle T
        """
        x1, x2, x3 = T[0, 0], T[0, 1], T[0, 2]
        y1, y2, y3 = T[1, 0], T[1, 1], T[1, 2]
        
        # Triangle area
        A = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2
        
        # Gaussian quadrature points
        xyw = Swarm.TriGaussPoints(N)
        
        I = 0
        for j in range(xyw.shape[0]):
            # Map reference triangle to actual triangle
            xi, eta, w = xyw[j]
            x = x1 * (1 - xi - eta) + x2 * xi + x3 * eta
            y = y1 * (1 - xi - eta) + y2 * xi + y3 * eta
            
            # Evaluate function and add weighted contribution
            I += f(x, y) * w
        
        return A * I
    
    @staticmethod
    def TriGaussPoints(n):
        """
        Returns Gaussian quadrature points for triangles
        """
        if n == 1:
            return np.array([[1/3, 1/3, 1.0]])
        elif n == 2:
            return np.array([
                [1/6, 1/6, 1/3],
                [1/6, 2/3, 1/3],
                [2/3, 1/6, 1/3]
            ])
        elif n == 3:
            return np.array([
                [1/3, 1/3, -0.5625],
                [0.2, 0.2, 0.520833333333333],
                [0.2, 0.6, 0.520833333333333],
                [0.6, 0.2, 0.520833333333333]
            ])
        elif n == 4:
            return np.array([
                [0.445948490915965, 0.445948490915965, 0.223381589678011],
                [0.445948490915965, 0.108103018168070, 0.223381589678011],
                [0.108103018168070, 0.445948490915965, 0.223381589678011],
                [0.091576213509771, 0.091576213509771, 0.109951743655322],
                [0.091576213509771, 0.816847572980458, 0.109951743655322],
                [0.816847572980458, 0.091576213509771, 0.109951743655322]
            ])
        elif n == 5:
            return np.array([
                [1/3, 1/3, 0.225],
                [0.470142064105115, 0.470142064105115, 0.132394152788506],
                [0.470142064105115, 0.059715871789770, 0.132394152788506],
                [0.059715871789770, 0.470142064105115, 0.132394152788506],
                [0.101286507323456, 0.101286507323456, 0.125939180544827],
                [0.101286507323456, 0.797426985353088, 0.125939180544827],
                [0.797426985353088, 0.101286507323456, 0.125939180544827]
            ])
        elif n == 6:
            return np.array([
                [0.249286745170910, 0.249286745170910, 0.116786275726379],
                [0.249286745170910, 0.501426509658180, 0.116786275726379],
                [0.501426509658180, 0.249286745170910, 0.116786275726379],
                [0.063089014491502, 0.063089014491502, 0.050844906370207],
                [0.063089014491502, 0.873821971017001, 0.050844906370207],
                [0.873821971017001, 0.063089014491502, 0.050844906370207],
                [0.310352451033785, 0.636502499121399, 0.082851075618374],
                [0.636502499121399, 0.053145049844816, 0.082851075618374],
                [0.053145049844816, 0.310352451033785, 0.082851075618374],
                [0.636502499121399, 0.310352451033785, 0.082851075618374],
                [0.310352451033785, 0.053145049844816, 0.082851075618374],
                [0.053145049844816, 0.636502499121399, 0.082851075618374]
            ])
        elif n == 7:
            return np.array([
                [1/3, 1/3, -0.149570044467682],
                [0.260345966079038, 0.260345966079038, 0.175615257433206],
                [0.260345966079038, 0.479308067841924, 0.175615257433206],
                [0.479308067841924, 0.260345966079038, 0.175615257433206],
                [0.065130102902216, 0.065130102902216, 0.053347235608839],
                [0.065130102902216, 0.869739794195568, 0.053347235608839],
                [0.869739794195568, 0.065130102902216, 0.053347235608839],
                [0.312865496004874, 0.638444188569809, 0.077113760890257],
                [0.638444188569809, 0.048690315425316, 0.077113760890257],
                [0.048690315425316, 0.312865496004874, 0.077113760890257],
                [0.638444188569809, 0.312865496004874, 0.077113760890257],
                [0.312865496004874, 0.048690315425316, 0.077113760890257],
                [0.048690315425316, 0.638444188569809, 0.077113760890257]
            ])
        elif n == 8:
            return np.array([
                [1/3, 1/3, 0.144315607677787],
                [0.459292588292723, 0.459292588292723, 0.095091634267284],
                [0.459292588292723, 0.081414823414554, 0.095091634267284],
                [0.081414823414554, 0.459292588292723, 0.095091634267284],
                [0.170569307751761, 0.170569307751761, 0.103217370534718],
                [0.170569307751761, 0.658861384496478, 0.103217370534718],
                [0.658861384496478, 0.170569307751761, 0.103217370534718],
                [0.050547228317031, 0.050547228317031, 0.032458497623196],
                [0.050547228317031, 0.898905543365937, 0.032458497623196],
                [0.898905543365937, 0.050547228317031, 0.032458497623196],
                [0.263112829634639, 0.728492392955404, 0.027230314174435],
                [0.728492392955404, 0.008394777409957, 0.027230314174435],
                [0.008394777409957, 0.263112829634639, 0.027230314174435],
                [0.728492392955404, 0.263112829634639, 0.027230314174435],
                [0.263112829634639, 0.008394777409957, 0.027230314174435],
                [0.008394777409957, 0.728492392955404, 0.027230314174435]
            ])
        else:
            # Default to simple centroid rule if n not supported
            return np.array([[1/3, 1/3, 1.0]])
    
    @staticmethod
    def fillout(x, y, lims, **kwargs):
        """
        Fill outside of a polygon
        """
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'g'
            
        if len(np.shape(x)) > 1:
            x = Swarm.var_border(x)
        if len(np.shape(y)) > 1:
            y = Swarm.var_border(y)
            
        if len(x) != len(y):
            print("## fillout: x and y must have the same size")
            return None
        
        xi, xe = lims[0], lims[1]
        yi, ye = lims[2], lims[3]
        
        # Find starting point at minimum x
        i = np.argmin(x)
        
        # Rearrange points to start at minimum x
        x = np.concatenate((x[i:], x[:i], [x[i]]))
        y = np.concatenate((y[i:], y[:i], [y[i]]))
        
        # Create big rectangle to fill
        x_out = [xi, xi, xe, xe, xi, xi, x[0]]
        x_out.extend(x)
        
        y_out = [y[0], ye, ye, yi, yi, y[0], y[0]]
        y_out.extend(y)
        
        # Create polygon patch
        polygon = plt.Polygon(np.column_stack((x_out, y_out)), **kwargs)
        plt.gca().add_patch(polygon)
        
        return polygon
    
    @staticmethod
    def var_border(M):
        """
        Extract border of a matrix
        """
        if len(M) == 0:
            return []
            
        # Extract the four sides of the matrix
        left = M[:, 0]
        top = M[-1, :]
        right = np.flip(M[:, -1])
        bottom = np.flip(M[0, :])
        
        # Combine into a border
        border = np.concatenate((left, top, right, bottom))
        corners = [left[0], left[-1], right[0], right[-1]]
        
        return border
