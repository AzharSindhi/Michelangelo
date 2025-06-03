# -*- coding: utf-8 -*-
import math
import torch

class LambdaWarmUpCosineFactorScheduler:
    """
    A simple learning rate scheduler that combines warmup and cosine decay.
    """
    def __init__(self, warm_up_steps=5000, f_start=1e-6, f_min=1e-3, f_max=1.0, max_steps=100):
        """
        Args:
            warm_up_steps: Number of warmup steps
            f_start: Initial learning rate factor
            f_min: Minimum learning rate factor after warmup
            f_max: Maximum learning rate factor after warmup and before decay
        """
        self.warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.max_steps = max_steps
        self.last_step = 0

    def schedule(self, step):
        """
        Compute the learning rate factor for the given step.
        
        Args:
            step: Current training step
            
        Returns:
            float: Learning rate factor
        """
        self.last_step = step
        
        if step < self.warm_up_steps:
            # Linear warmup
            factor = self.f_start + (self.f_max - self.f_start) * (step / self.warm_up_steps)
        else:
            # Cosine decay
            progress = (step - self.warm_up_steps) / max(1, self.max_steps - self.warm_up_steps)
            factor = self.f_min + 0.5 * (self.f_max - self.f_min) * (1 + math.cos(math.pi * progress))
            
        return factor
    
    # def __call__(self, step):
    #     """Alias for schedule for compatibility"""
    #     return self.schedule(step)
