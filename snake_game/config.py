
class SnakeConfig:
    def __init__(self):
        # --- Network ---
        # State: [HeadX, HeadY, FoodX, FoodY, DangerL, DangerR, DangerU, DangerD, DirL, DirR, DirU, DirD]
        # Normalized.
        self.state_dim = 12 
        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space_size = 4
        self.hidden_dim = 128
        
        # --- Training ---
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.training_steps = 100000 
        self.window_size = 50000
        self.priority_exponent = 0.6
        self.priority_sampling = 0.4
        self.discount = 0.95
        
        # --- Exploration ---
        self.epsilon_init = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # --- Device ---
        self.device = "cpu"
