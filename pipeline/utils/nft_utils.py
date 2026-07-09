import numpy as np
from collections import defaultdict

class PerPocketStatTracker:
    """Track per-pocket statistics for advantage calculation."""
    
    def __init__(self, global_std: bool = False):
        self.stats = {}
        self.global_std = global_std
        
    def update(self, pocket_names, rewards):
        """Update statistics with new rewards."""
        # Group rewards by pocket
        pocket_rewards = defaultdict(list)
        for pocket_name, reward in zip(pocket_names, rewards):
            pocket_rewards[pocket_name].append(reward)
        
        # Update running statistics
        for pocket_name, reward_list in pocket_rewards.items():
            if pocket_name not in self.stats:
                self.stats[pocket_name] = {'rewards': [], 'mean': 0, 'std': 1}
            self.stats[pocket_name]['rewards'].extend(reward_list)
            self.stats[pocket_name]['mean'] = np.mean(self.stats[pocket_name]['rewards'])
            self.stats[pocket_name]['std'] = np.std(self.stats[pocket_name]['rewards']) + 1e-6
        
        # Calculate advantages
        advantages = []
        if self.global_std:
            all_rewards = [r for rewards in pocket_rewards.values() for r in rewards]
            global_std = np.std(all_rewards) + 1e-6
            for pocket_name, reward in zip(pocket_names, rewards):
                mean = self.stats[pocket_name]['mean']
                advantages.append((reward - mean) / global_std)
        else:
            for pocket_name, reward in zip(pocket_names, rewards):
                mean = self.stats[pocket_name]['mean']
                std = self.stats[pocket_name]['std']
                advantages.append((reward - mean) / std)
        
        return np.array(advantages)
    
    def clear(self):
        """Clear all statistics."""
        self.stats = {}
    
    def get_stats(self):
        """Get summary statistics."""
        if not self.stats:
            return 0, 0
        group_size = np.mean([len(s['rewards']) for s in self.stats.values()])
        trained_pockets = len(self.stats)
        return group_size, trained_pockets
    
    def get_mean_of_top_rewards(self, percentile):
        """Get mean of top percentile rewards."""
        all_rewards = [r for s in self.stats.values() for r in s['rewards']]
        if not all_rewards:
            return 0
        threshold = np.percentile(all_rewards, percentile)
        top_rewards = [r for r in all_rewards if r >= threshold]
        return np.mean(top_rewards) if top_rewards else 0


def return_decay(step, decay_type):
    """Compute decay factor for old model update."""
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)

