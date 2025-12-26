
class MuZeroConfig:
    def __init__(self):
        # --- Self-Play ---
        self.num_actors = 1 # Single process for now
        self.max_moves = 500 # Max steps per episode
        self.num_simulations = 30 # MCTS simulations per turn
        self.discount = 0.95
        self.temperature_visit = 0 # 0 = Max visit count (Greedy), 1 = Proportional
        
        # --- Network ---
        self.state_dim = 6 # [rel_x, angle, height, shape, lean, stability]
        # Action space: 13 positions * 5 rotations = 65
        self.x_positions = list(range(100, 701, 50)) # 13 pos
        self.rotations = [-0.52, -0.26, 0.0, 0.26, 0.52] # 5 rots
        self.action_space_size = len(self.x_positions) * len(self.rotations)
        self.hidden_dim = 128
        
        # --- Training ---
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.training_steps = 10000 
        self.checkpoint_interval = 50
        self.window_size = 100000 # Replay buffer size
        self.priority_exponent = 0.6 # PER alpha
        self.priority_sampling = 0.4 # PER beta (initial)
        
        # --- Exploration ---
        self.epsilon_init = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # --- Device ---
        self.device = "cpu" # 'cuda' if available
