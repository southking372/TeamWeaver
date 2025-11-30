import numpy as np
from matplotlib.path import Path

class DisturbanceConfig:
    """
    Configuration class for handling disturbances in the simulation.
    Manages both exogenous and endogenous disturbances.
    """
    
    def __init__(self, global_vars_manager):
        """
        Initialize the disturbance configuration
        
        Parameters:
            global_vars_manager: GlobalVarsManager instance for accessing global variables
        """
        self.global_vars = global_vars_manager
        
        # Default disturbance parameters
        self.robot_exo_dist = 4        # Robot index for exogenous disturbance
        self.task_exo_dist = 2         # Task index for exogenous disturbance
        self.t_endogenous = 15         # Time when endogenous disturbance occurs
        self.x_mud = None              # X coordinates of mud area
        self.y_mud = None              # Y coordinates of mud area
        
        # Update global variables
        self._update_global_vars()
    
    def configure_exogenous_disturbance(self, robot_idx, task_idx):
        """
        Configure the exogenous disturbance parameters
        
        Parameters:
            robot_idx: Index of the robot affected by the disturbance
            task_idx: Index of the task affected by the disturbance
        """
        self.robot_exo_dist = robot_idx
        self.task_exo_dist = task_idx
        self._update_global_vars()
    
    def configure_endogenous_disturbance(self, time):
        """
        Configure the endogenous disturbance parameters
        
        Parameters:
            time: Time when the endogenous disturbance occurs
        """
        self.t_endogenous = time
        self._update_global_vars()
    
    def set_mud_area(self, x_coords, y_coords):
        """
        Set the coordinates of the mud area
        
        Parameters:
            x_coords: X coordinates of the mud area vertices
            y_coords: Y coordinates of the mud area vertices
        """
        self.x_mud = x_coords
        self.y_mud = y_coords
        self._update_global_vars()
    
    def load_mud_from_file(self, filename):
        """
        Load mud area coordinates from a file
        
        Parameters:
            filename: Path to the file containing mud coordinates
        """
        from task_utils.tools_util import read_mud_file
        self.x_mud, self.y_mud = read_mud_file(filename)
        self._update_global_vars()
    
    def check_exogenous_disturbance(self, x, alpha, robot_idx):
        """
        Check if an exogenous disturbance affects the specified robot
        
        Parameters:
            x: Robot state matrix
            alpha: Task allocation matrix
            robot_idx: Index of the robot to check
            
        Returns:
            bool: True if the robot is affected by an exogenous disturbance
        """
        # Get current values from global vars or local instance
        robot_exo_dist_val = self.robot_exo_dist
        task_exo_dist_val = self.task_exo_dist
        x_mud_val = self.x_mud
        y_mud_val = self.y_mud
        
        # If mud coordinates are not available, return False
        if x_mud_val is None or y_mud_val is None:
            return False
        
        # Check if robot is in mud area
        path = Path(np.column_stack((x_mud_val, y_mud_val)))
        in_mud = path.contains_point((x[0, robot_idx], x[1, robot_idx]))
        
        # Return True if all conditions are met
        return (robot_idx == robot_exo_dist_val - 1) and \
               (alpha[task_exo_dist_val - 1, robot_idx] > 0) and \
               in_mud
    
    def _update_global_vars(self):
        """Update the global variables with current disturbance configuration"""
        self.global_vars.robot_exo_dist = self.robot_exo_dist
        self.global_vars.task_exo_dist = self.task_exo_dist
        self.global_vars.t_endogenous = self.t_endogenous
        if self.x_mud is not None and self.y_mud is not None:
            self.global_vars.x_mud = self.x_mud
            self.global_vars.y_mud = self.y_mud