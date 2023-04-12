# %%
# Reference: 021_HFI_Benchmark_Pytorch.ipynb
#            https://colab.research.google.com/drive/1kefMUU1lpzUdRQRh47jD_IyPiCwDD4x3#scrollTo=0OJUdCH_Qdd7

# %% [markdown]
# # Libraries & Dependencies Import

# %%
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
device = torch.device("cuda:0")

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# %%
# !git clone https://github.com/bethgelab/model-vs-human.git
# !pip install -q git+https://github.com/openai/CLIP
# !pip install git+https://github.com/bethgelab/model-vs-human --no-deps
# !pip install git+https://github.com/bethgelab/model-vs-human --no-deps

# %% [markdown]
# # ImageNet Data Info

# %%
imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# %%
data_path = './datasets/clickme_test_1000.tfrecords'

# %% [markdown]
# # Building Dataset

# %%
AUTO = tf.data.AUTOTUNE
BLUR_KERNEL_SIZE = 10
BLUR_SIGMA = 10

_feature_description = {
      "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

def set_size(w,h):
    """Set matplot figure size"""
    plt.rcParams["figure.figsize"] = [w,h]
    
def show(img, p=False, smooth=False, **kwargs):
    """ Display torch/tf tensor """ 
    try:
        img = img.detach().cpu()
    except:
        img = np.array(img)

    img = np.array(img, dtype=np.float32)

    # check if channel first
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)

    # check if cmap
    if img.shape[-1] == 1:
        img = img[:,:,0]

    # normalize
    if img.max() > 1 or img.min() < 0:
        img -= img.min(); img/=img.max()

    # check if clip percentile
    if p is not False:
        img = np.clip(img, np.percentile(img, p), np.percentile(img, 100-p))

    if smooth and len(img.shape) == 2:
        img = gaussian_filter(img, smooth)

    plt.imshow(img, **kwargs)
    plt.axis('off')
    plt.grid(None)

def parse_prototype(prototype, training=False):
    data    = tf.io.parse_single_example(prototype, _feature_description)

    image   = tf.io.decode_raw(data['image'], tf.float32)
    image   = tf.reshape(image, (224, 224, 3))
    image   = tf.cast(image, tf.float32)

    heatmap = tf.io.decode_raw(data['heatmap'], tf.float32)
    heatmap = tf.reshape(heatmap, (224, 224, 1))

    label   = tf.cast(data['label'], tf.int32)
    label   = tf.one_hot(label, 1_000)

    return image, heatmap, label

def get_dataset(batch_size, training=False):
    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset([data_path], num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order) 
      
    dataset = dataset.map(parse_prototype, num_parallel_calls=AUTO)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset
  
# dataset = get_dataset(10, False)
# cnt = 0
# for imgs, hmps, labels in dataset: # image, heatmap, label
#     for img, hmp, label in zip(imgs, hmps, labels):
#         print(img.dtype, label.dtype, img.shape)
#         img = tf.cast(img, tf.float32).numpy()
#         img -= img.min()
#         img /= img.max()

#         hmp = tf.cast(hmp, tf.float32).numpy()

#         show(img)
#         show(hmp, cmap='jet', alpha=0.3)
#         print(imagenet_classes[np.argmax(label)])
#         plt.title(f"{hmp.shape} H({hmp.mean()} {hmp.std()}), X({img.min()} {img.max()})")
#         plt.axis('off')
#         plt.show()
#         print('\n\n\n')

#     cnt += 1
#     if cnt > 10:
#         break
#     break

# %% [markdown]
# # Utils

# %%
def _tf_to_torch(t): # a batch of image tensors (N, H, W, 3)
    t = tf.cast(t, tf.float32).numpy()
    if t.shape[-1] in [1, 3]:
        t = torch.from_numpy(t.transpose(0, 3, 1, 2)) # torch.from_numpy(np_array.transpose(0, 3, 1, 2)) 
        return t
    return torch.from_numpy(t) # (N, 3, H, W)

def img_normalize(imgs):
    imgs = imgs - imgs.min()
    imgs = imgs / imgs.max()
    return imgs

import csv
import os
def writeCSV(record, path):
    header = ['model', 'epsilon', 'num_correct', 'num_correct_perturbated', 'attack success rate']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(record)


# %% [markdown]
# # Load Models

# %%
try:
    import modelvshuman
except:
    pass
  
import modelvshuman
from modelvshuman import models

print(len(models.list_models("tensorflow")))
print(len(models.list_models("pytorch")))

# %%
for i, model_name in enumerate(models.list_models("pytorch")):
    print(str(i) + " : " + model_name)
    
for i, model_name in enumerate(models.list_models("tensorflow")):
    print(str(i) + " : " + model_name)

# %% [markdown]
# # Adversarial Example Generation

# %%
import torchattacks
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# %% [markdown]
# # Attack!!!

# %%
zoo = modelvshuman.models.pytorch.model_zoo
model = None

BATCH_SIZE = 1000
list_models = models.list_models("pytorch")
# print(list_models)

epsilon = 0.05

results_path = './results/FGSM_min_perturbation.csv'

# for MODEL_NAME in list_models:
offset = 3 # 6,7,61,62, 76
for i, MODEL_NAME in enumerate(list_models[offset:offset+2]): 
    print(str(i + offset), MODEL_NAME)
    
    try:
        model = getattr(zoo, MODEL_NAME)(MODEL_NAME)
        model = model.model.to(device)
    except:
        row_data = [MODEL_NAME, str(epsilon), "", "", ""]
        writeCSV(row_data, results_path)
        continue

    model.eval()
    
    keepSearching = 1
    while keepSearching == 1:
        
        clickme_dataset_test = get_dataset(10, False)
        total_cnt, init_correct, aa_correct = 0, 0, 0 # init_correct -> 'num_correct', aa_correct -> 'num_correct_perturbated'
        num_of_testing_imgs = 1000
        label_set = set()
        for imgs, hmps, labels in clickme_dataset_test: # image, heatmap, label
            imgs = _tf_to_torch(imgs).to(device)
            imgs = img_normalize(imgs)
            labels = _tf_to_torch(labels).to(device)

            for img, hmp, label in zip(imgs, hmps, labels):
                # print(img.shape, hmp.shape, logit.shape)

                img = torch.unsqueeze(img, 0)
                label = torch.unsqueeze(label, 0)
                target = torch.argmax(label, axis=-1) # tensor([343], device='cuda:0')

                # Set requires_grad attribute of tensor. Important for Attack
                img.requires_grad = True

                # Forward pass the data through the model
                output = model(img)
                init_pred = torch.argmax(output, axis=-1) # get the index of the max log-probability

                # If the initial prediction is wrong, dont bother attacking, just move on
                total_cnt += 1
                if init_pred.item() != target.item():
                    continue
                init_correct += 1

                # Collect datagrad
                model.zero_grad() # Zero all existing gradients
                loss = F.nll_loss(output, target).to(device) # Calculate the loss    
                loss.backward() # Calculate gradients of model in backward pass
                data_grad = img.grad.data

                # Call FGSM Attack
                perturbed_img = fgsm_attack(img, epsilon, data_grad)

#                 # Call FGSM Attack
#                 cw_attack = torchattacks.FGSM(model, eps=epsilon)
#                 perturbed_img = cw_attack(img, target)

                # Re-classify the perturbed image
                output = model(perturbed_img)
                final_pred = torch.argmax(output, axis=-1) # get the index of the max log-probability
                if final_pred.item() == target.item():
                    aa_correct += 1
                    break
            
            print("\repsilon: %s total_cnt: %s init_correct = %s aa_correct = %s" % (str(epsilon), str(total_cnt), str(init_correct), str(aa_correct)), end=" ")
            if aa_correct >= 1:
                break
        print("")

        if aa_correct == 0:
            row_data = [MODEL_NAME, str(epsilon), str(init_correct), str(aa_correct), str(0)]
            writeCSV(row_data, results_path)
            print("min epsilon:" + str(epsilon))
            keepSearching = 0
            break
        elif epsilon > 1:
            row_data = [MODEL_NAME, str(epsilon), str(init_correct), "", ""]
            writeCSV(row_data, results_path)
            print("min epsilon:" + str(epsilon))
            keepSearching = 0
            break
        else:
            epsilon += 0.05 # increase 0.05 if aa_correct = 1
        
    torch.cuda.empty_cache()
   

# %%



