import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2 (scale to [-1, 1])
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def get_animal_classes():
    """
    Return a mapping of ImageNet class indices to animal names
    This focuses on common animals that users are likely to photograph
    """
    # Mapping of ImageNet class indices to animal names
    # This is a subset focusing on common animals
    animal_classes = {
        # Dogs
        151: "Chihuahua", 152: "Japanese Spaniel", 153: "Maltese Dog", 154: "Pekinese",
        155: "Shih-Tzu", 156: "Blenheim Spaniel", 157: "Papillon", 158: "Toy Terrier",
        159: "Rhodesian Ridgeback", 160: "Afghan Hound", 161: "Basset", 162: "Beagle",
        163: "Bloodhound", 164: "Bluetick", 165: "Black-and-tan Coonhound", 166: "Walker Hound",
        167: "English Foxhound", 168: "Redbone", 169: "Borzoi", 170: "Irish Wolfhound",
        171: "Italian Greyhound", 172: "Whippet", 173: "Ibizan Hound", 174: "Norwegian Elkhound",
        175: "Otterhound", 176: "Saluki", 177: "Scottish Deerhound", 178: "Weimaraner",
        179: "Staffordshire Bullterrier", 180: "American Staffordshire Terrier", 181: "Bedlington Terrier",
        182: "Border Terrier", 183: "Kerry Blue Terrier", 184: "Irish Terrier", 185: "Norfolk Terrier",
        186: "Norwich Terrier", 187: "Yorkshire Terrier", 188: "Wire-haired Fox Terrier",
        189: "Lakeland Terrier", 190: "Sealyham Terrier", 191: "Airedale", 192: "Cairn",
        193: "Australian Terrier", 194: "Dandie Dinmont", 195: "Boston Bull", 196: "Miniature Schnauzer",
        197: "Giant Schnauzer", 198: "Standard Schnauzer", 199: "Scotch Terrier", 200: "Tibetan Terrier",
        201: "Silky Terrier", 202: "Soft-coated Wheaten Terrier", 203: "West Highland White Terrier",
        204: "Lhasa", 205: "Flat-coated Retriever", 206: "Curly-coated Retriever", 207: "Golden Retriever",
        208: "Labrador Retriever", 209: "Chesapeake Bay Retriever", 210: "German Short-haired Pointer",
        211: "Vizsla", 212: "English Setter", 213: "Irish Setter", 214: "Gordon Setter",
        215: "Brittany Spaniel", 216: "Clumber", 217: "English Springer", 218: "Welsh Springer Spaniel",
        219: "Cocker Spaniel", 220: "Sussex Spaniel", 221: "Irish Water Spaniel", 222: "Kuvasz",
        223: "Schipperke", 224: "Groenendael", 225: "Malinois", 226: "Briard", 227: "Kelpie",
        228: "Komondor", 229: "Old English Sheepdog", 230: "Shetland Sheepdog", 231: "Collie",
        232: "Border Collie", 233: "Bouvier des Flandres", 234: "Rottweiler", 235: "German Shepherd",
        236: "Doberman", 237: "Miniature Pinscher", 238: "Greater Swiss Mountain Dog", 239: "Bernese Mountain Dog",
        240: "Appenzeller", 241: "EntleBucher", 242: "Boxer", 243: "Bull Mastiff", 244: "Tibetan Mastiff",
        245: "French Bulldog", 246: "Great Dane", 247: "Saint Bernard", 248: "Eskimo Dog",
        249: "Malamute", 250: "Siberian Husky", 251: "Dalmatian", 252: "Affenpinscher", 253: "Basenji",
        254: "Pug", 255: "Leonberg", 256: "Newfoundland", 257: "Great Pyrenees", 258: "Samoyed",
        259: "Pomeranian", 260: "Chow", 261: "Keeshond", 262: "Brabancon Griffon", 263: "Pembroke",
        264: "Cardigan", 265: "Toy Poodle", 266: "Miniature Poodle", 267: "Standard Poodle", 268: "Mexican Hairless",
        
        # Cats
        281: "Tabby Cat", 282: "Tiger Cat", 283: "Persian Cat", 284: "Siamese Cat", 285: "Egyptian Cat",
        
        # Wild Cats
        286: "Cougar", 287: "Lynx", 288: "Leopard", 289: "Snow Leopard", 290: "Jaguar",
        291: "Lion", 292: "Tiger", 293: "Cheetah",
        
        # Bears
        294: "Brown Bear", 295: "American Black Bear", 296: "Ice Bear", 297: "Sloth Bear",
        
        # Other Mammals
        298: "Mongoose", 299: "Meerkat", 300: "Tiger Beetle", 301: "Ladybug", 302: "Ground Beetle",
        
        # Farm Animals
        345: "Pig", 346: "Wild Boar", 347: "Warthog", 348: "Hippopotamus",
        349: "Ox", 350: "Water Buffalo", 351: "Bison", 352: "Ram", 353: "Bighorn", 354: "Ibex",
        
        # Cattle/Cows (ImageNet indices for bovines)
        # These are the key missing classes for proper cow identification
        343: "Zebu", 344: "Ox", 339: "Zebra", 340: "Sorrel", 341: "Buckskin", 342: "Arabian",
        # More specific cattle classes found in ImageNet
        347: "Hog", 621: "Jersey", 622: "Guernsey", 623: "Holstein", 624: "Dairy Cow",
        355: "Hartebeest", 356: "Impala", 357: "Gazelle", 358: "Arabian Camel", 359: "Llama",
        360: "Weasel", 361: "Mink", 362: "Polecat", 363: "Black-footed Ferret", 364: "Otter",
        365: "Skunk", 366: "Badger", 367: "Armadillo", 368: "Three-toed Sloth", 369: "Orangutan",
        370: "Gorilla", 371: "Chimpanzee", 372: "Gibbon", 373: "Siamang", 374: "Guenon",
        375: "Patas", 376: "Baboon", 377: "Macaque", 378: "Langur", 379: "Colobus",
        380: "Proboscis Monkey", 381: "Marmoset", 382: "Capuchin", 383: "Howler Monkey",
        384: "Titi", 385: "Spider Monkey", 386: "Squirrel Monkey", 387: "Madagascar Cat",
        388: "Indri", 389: "Indian Elephant", 390: "African Elephant",
        
        # Birds (selection)
        8: "Hen", 9: "Ostrich", 10: "Brambling", 11: "Goldfinch", 12: "House Finch",
        13: "Junco", 14: "Indigo Bunting", 15: "Robin", 16: "Bulbul", 17: "Jay",
        18: "Magpie", 19: "Chickadee", 20: "Water Ouzel", 21: "Kite", 22: "Bald Eagle",
        23: "Vulture", 24: "Great Grey Owl", 80: "Black Swan", 81: "Tusker", 82: "Echidna",
        83: "Platypus", 84: "Wallaby", 85: "Koala", 86: "Wombat", 87: "Jellyfish",
        88: "Sea Anemone", 89: "Brain Coral", 90: "Flatworm", 91: "Nematode", 92: "Conch",
        93: "Snail", 94: "Slug", 95: "Sea Slug", 96: "Chiton", 97: "Chambered Nautilus",
        98: "Dungeness Crab", 99: "Rock Crab", 100: "Fiddler Crab", 101: "King Crab",
        102: "American Lobster", 103: "Spiny Lobster", 104: "Crayfish", 105: "Hermit Crab",
        106: "Isopod", 107: "White Stork", 108: "Black Stork", 109: "Spoonbill", 110: "Flamingo",
        111: "Little Blue Heron", 112: "American Egret", 113: "Bittern", 114: "Crane",
        115: "Limpkin", 116: "European Gallinule", 117: "American Coot", 118: "Bustard",
        119: "Ruddy Turnstone", 120: "Red-backed Sandpiper", 121: "Redshank", 122: "Dowitcher",
        123: "Oystercatcher", 124: "Pelican", 125: "King Penguin", 126: "Albatross",
        127: "Grey Whale", 128: "Killer Whale", 129: "Dugong", 130: "Sea Lion",
        
        # More animals
        131: "Chihuahua", 132: "Japanese Spaniel", 133: "Maltese Dog", 134: "Pekinese",
        135: "Shih-Tzu", 136: "Blenheim Spaniel", 137: "Papillon", 138: "Toy Terrier",
        139: "Rhodesian Ridgeback", 140: "Afghan Hound", 141: "Basset", 142: "Beagle",
        143: "Bloodhound", 144: "Bluetick", 145: "Black-and-tan Coonhound", 146: "Walker Hound",
        147: "English Foxhound", 148: "Redbone", 149: "Borzoi", 150: "Irish Wolfhound"
    }
    
    return animal_classes

def map_imagenet_to_animals(predictions, top_k=5):
    """
    Map ImageNet predictions to animal names
    
    Args:
        predictions: Model predictions array
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples (animal_name, confidence_percentage)
    """
    animal_classes = get_animal_classes()
    
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k * 3]  # Get more to filter animals
    
    animal_predictions = []
    for idx in top_indices:
        if idx in animal_classes:
            confidence = predictions[0][idx] * 100
            animal_predictions.append((animal_classes[idx], confidence))
        
        if len(animal_predictions) >= top_k:
            break
    
    return animal_predictions
