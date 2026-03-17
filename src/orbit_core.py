import torch
import torch.nn.functional as F

class PhaseSpaceOrbit:
    def __init__(self, state_dim=384, alpha=0.8):
        self.state_dim = state_dim
        self.h_t = torch.zeros(1, state_dim)
        self.alpha = alpha 

    def update_orbit(self, new_data_vector):
        new_data_vector = F.normalize(new_data_vector, p=2, dim=1)
        
        if torch.sum(self.h_t) == 0:
            self.h_t = new_data_vector
        else:
            self.h_t = (self.alpha * self.h_t) + ((1.0 - self.alpha) * new_data_vector)
            self.h_t = F.normalize(self.h_t, p=2, dim=1)
            
        return self.h_t
