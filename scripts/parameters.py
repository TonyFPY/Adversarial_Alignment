class Parameters:
    def __init__(self):
        # FGSM, FFGSM
        self.epsilons = [ 
            0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 
            0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0,
        ]
        self.alpha = 10/255

        # C&W
        self.consts = [ 
            0.01, 0.025, 0.05, 0.075, 0.1,
            0.25, 0.5, 0.75, 1,
            1.5, 2.0, 2.5, 3.0
        ]

        self.const = 0.1

        self.kappa = 0

        self.steps = 10

        self.lr = 0.0075

        self.overshoot = 0.02

    def get_alpha(self):
        return self.alpha

    def get_epsilons(self):
        return self.epsilons

    def get_consts(self):
        return self.consts
    
    def get_const(self):
        return self.const

    def get_kappa(self):
        return self.kappa

    def get_steps(self):
        return self.steps

    def get_lr(self):
        return self.lr