import os

# timm
NUM_CLASSES = 1000
NEED_PRETRAINED = True

# 120 animate, 120 in-animate
IMAGNET_BIG12_RANGES = {
    'dog, domestic dog, Canis familiaris', (151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170), 
    'bird', (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 87, 88), 
    'reptile, reptilian', (33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52), 
    'carnivore', (269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 286, 287, 288, 289, 290, 291, 292, 293), 
    'insect', (300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319), 
    'primate', (384, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383),
    'structure, construction', (525, 406, 536, 410, 421, 424, 425, 437, 449, 454, 458, 460, 467, 483, 489, 497, 498, 500, 506, 509), 
    'clothing, article of clothing, vesture, wear, wearable, habiliment', (515, 518, 399, 400, 529, 411, 552, 560, 433, 439, 568, 445, 578, 451, 452, 457, 459, 601, 474, 496), 
    'wheeled vehicle', (656, 661, 407, 408, 547, 555, 428, 561, 436, 565, 569, 444, 573, 586, 468, 603, 609, 612, 627, 511), 
    'musical instrument, instrument', (513, 641, 642, 401, 402, 541, 546, 420, 683, 684, 558, 687, 432, 566, 577, 579, 593, 594, 486, 494), 
    'food, solid food', (930, 931, 932, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 949, 950, 951, 952, 953), 
    'furniture, piece of furniture, article of furniture', (516, 520, 648, 526, 532, 548, 423, 553, 431, 559, 564, 831, 703, 453, 846, 857, 861, 493, 495, 765), 
}

IMAGNET_BINARY_RANGES = {
    'animate': tuple(range(0, 398)),
    'in-animate': tuple(range(398, 1000)),
}

# ImageNet data info
IN_LABEL_INFO = '/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_label/imagenet_class_index.json'
DATA_DIR = '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full'
TRAIN_PATHS = os.path.join(DATA_DIR, 'PseudoClickMe/train/*.pth')
VAL_PATHS = os.path.join(DATA_DIR, 'PseudoClickMe/val/*.pth')
TRAIN_CLICKME_PATHS = os.path.join(DATA_DIR, 'ClickMe/train/*.pth')
VAL_CLICKME_PATHS = os.path.join(DATA_DIR, 'ClickMe/val/*.pth')
TEST_CLICKME_PATHS = os.path.join(DATA_DIR, 'ClickMe/test/*.pth')

# Restricted ImageNet Data
BIG12_DIR = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_big12"
NUM_IMGS_SC = 5

# Attack
EPS = 30
ALPHA = 0.5
STEPS = 100
