#include "yolo_trt.h"

// 定义在本文件末尾
extern const std::vector<std::string> cls_classes;
extern const std::vector<std::string> det_classes;
extern const std::vector<std::string> obb_classes;
extern const std::vector<std::string> pose_classes;
extern const std::vector<std::string> seg_classes;
extern const std::vector<std::vector<float>> color_list;

// Pose 骨架
const std::vector<std::pair<int, int>> skeleton_pairs = {
    {0, 1}, {0, 2},  {0, 5}, {0, 6},  {1, 2},   {1, 3},   {2, 4},   {5, 6},   {5, 7},  {5, 11},
    {6, 8}, {6, 12}, {7, 9}, {8, 10}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    ModelParams clsParams  = {224, 1000, cls_classes, 0.0f, 0.0f, 5, 0.0f, 0, 0.0f};
    ModelParams detParams  = {640, 80, det_classes, 0.25f, 0.45f, 300, 0.0f, 0, 0.0f};
    ModelParams obbParams  = {1024, 15, obb_classes, 0.25f, 0.45f, 300, 0.0f, 0, 0.0f};
    ModelParams poseParams = {640, 1, pose_classes, 0.25f, 0.45f, 300, 0.0f, 17, 0.5f};
    ModelParams segParams  = {640, 80, seg_classes, 0.25f, 0.45f, 300, 0.5f, 0, 0.0f};
    
    std::string model_type, model_path, image_path;
    std::unique_ptr<YOLODetector> model;
    ModelParams model_params;

    std::cout << "Enter the type of the model file (cls/det/obb/pose/seg): ";
    std::cin >> model_type;
    if (model_type != "cls" && model_type != "det" && model_type != "obb" && model_type != "pose" && model_type != "seg") {
        std::cerr << "Unsupported model type!" << std::endl;
        return -1;
    }

    std::cout << "Enter the path to the model file: ";
    std::cin >> model_path;

    if (model_type == "cls") {
        std::cout << "Loading CLS model..." << std::endl;
        model_params = clsParams;
        model = std::make_unique<YOLOCls>(model_path, clsParams);
    }
    else if (model_type == "det") {
        std::cout << "Loading DET model..." << std::endl;
        model_params = detParams;
        model = std::make_unique<YOLODet>(model_path, detParams);
    }
    else if (model_type == "obb") {
        std::cout << "Loading OBB model..." << std::endl;
        model_params = obbParams;
        model = std::make_unique<YOLOObb>(model_path, obbParams);
    }
    else if (model_type == "pose") {
        std::cout << "Loading POSE model..." << std::endl;
        model_params = poseParams;
        model = std::make_unique<YOLOPose>(model_path, poseParams);
    }
    else if (model_type == "seg") {
        std::cout << "Loading SEG model..." << std::endl;
        model_params = segParams;
        model = std::make_unique<YOLOSeg>(model_path, segParams);
    }

    while (true) {
        std::cout << "Enter the path to the image file: ";
        std::cin >> image_path;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            return -1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model->infer(image);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end - start;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
        std::cout << "Detect " << result.size() << " objects." << std::endl;
        // 绘制 detections 在图像上
        #ifdef IS_DEBUG
        
        cv::Point default_pos = cv::Point(10, 10);
        cv::Point label_pos(image.cols, image.rows);
        cv::Mat mask = image.clone();
        for (const auto& det : result) {
            if(!det.box.empty()) {
                // HBB box
                cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
                label_pos = cv::Point(det.box.x, det.box.y);
            }
            else if(!det.rotatedBox.size.empty()) {
                // OBB box
                cv::Point2f vertices[4];
                det.rotatedBox.points(vertices);
                for (int j = 0; j < 4; j++) {
                    cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
                    if (label_pos.y > vertices[j].y) {
                        // 更新为最靠上的点
                        label_pos.x = static_cast<int>(vertices[j].x);
                        label_pos.y = static_cast<int>(vertices[j].y);
                    }
                }
            }
            else {
                // CLS 默认位置
                label_pos = default_pos;
                default_pos.y += 20;
            }
            std::string label = "ID: " + model_params.class_names[det.class_id] + " Conf: " + cv::format("%.4f", det.confidence);
            cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
            cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            // 绘制关键点
            bool is_pose = false;
            for (const auto& kp : det.keyPoints) {
                if (kp.x >= 0 && kp.y >= 0) { // 仅绘制有效关键点，后处理时将低于置信度的点置为负数，也可以用 kp.confidence 判断
                    cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 3, cv::Scalar(0, 0, 255), -1);
                }
                is_pose = true;
            }
            if (is_pose) {
                // 绘制骨架连接线
                for (const auto& bone : skeleton_pairs) {
                    int kp1_idx = bone.first;
                    int kp2_idx = bone.second;
                    if (det.keyPoints[kp1_idx].confidence > 0.5 && det.keyPoints[kp2_idx].confidence > 0.5) {
                        cv::Point p1((int)det.keyPoints[kp1_idx].x, (int)det.keyPoints[kp1_idx].y);
                        cv::Point p2((int)det.keyPoints[kp2_idx].x, (int)det.keyPoints[kp2_idx].y);
                        cv::line(image, p1, p2, cv::Scalar(0, 0x27, 0xC1), 2);
                    }
                }
            }
            // 绘制 mask
            if (!det.boxMask.empty()) {
                // Choose the color
                int colorIndex = det.class_id % color_list.size(); // We have only defined 80 unique colors
                cv::Scalar color = cv::Scalar(color_list[colorIndex][0], color_list[colorIndex][1], color_list[colorIndex][2]);

                // Add the mask for said object
                mask(det.box).setTo(color * 255, det.boxMask);
            }
        }
        // cv::imshow("Detections", image);
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
        cv::imwrite("output.jpg", image);
        // cv::waitKey(0);
        #endif
    }
    return 0;
}

const std::vector<std::string> cls_classes {"tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock", "quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern", "crane (bird)", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniels", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland", "Pyrenean Mountain Dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf", "Alaskan tundra wolf", "red wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "stick insect", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral", "ringlet", "monarch butterfly", "small white", "sulfur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn sheep", "Alpine ibex", "hartebeest", "impala", "gazelle", "dromedary", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek", "eel", "coho salmon", "rock beauty", "clownfish", "sturgeon", "garfish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "waste container", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military cap", "beer bottle", "beer glass", "bell-cot", "bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "bow", "bow tie", "brass", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "coil", "combination lock", "computer keyboard", "confectionery store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire engine", "fire screen sheet", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "barrette", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "jack-o'-lantern", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "pulled rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "speaker", "loupe", "sawmill", "magnetic compass", "mail bag", "mailbox", "tights", "tank suit", "manhole cover", "maraca", "marimba", "mask", "match", "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "square academic cap", "mosque", "mosquito net", "scooter", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "packet", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "pitcher", "hand plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "billiard table", "soda bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler", "running shoe", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swimsuit", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "yawl", "yurt", "website", "comic book", "crossword", "traffic sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice pop", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potato", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "shoal", "seashore", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star", "hen-of-the-woods", "bolete", "ear", "toilet paper"};

const std::vector<std::string> det_classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

const std::vector<std::string> obb_classes {"plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"};

const std::vector<std::string> pose_classes {"person"};

const std::vector<std::string> seg_classes = det_classes;

const std::vector<std::vector<float>> color_list = {{1, 1, 1},
                                                    {0.098f, 0.325f, 0.850f},
                                                    {0.125f, 0.694f, 0.929f},
                                                    {0.556f, 0.184f, 0.494f},
                                                    {0.188f, 0.674f, 0.466f},
                                                    {0.933f, 0.745f, 0.301f},
                                                    {0.184f, 0.078f, 0.635f},
                                                    {0.300f, 0.300f, 0.300f},
                                                    {0.600f, 0.600f, 0.600f},
                                                    {0.000f, 0.000f, 1.000f},
                                                    {0.000f, 0.500f, 1.000f},
                                                    {0.000f, 0.749f, 0.749f},
                                                    {0.000f, 1.000f, 0.000f},
                                                    {1.000f, 0.000f, 0.000f},
                                                    {1.000f, 0.000f, 0.667f},
                                                    {0.000f, 0.333f, 0.333f},
                                                    {0.000f, 0.667f, 0.333f},
                                                    {0.000f, 1.000f, 0.333f},
                                                    {0.000f, 0.333f, 0.667f},
                                                    {0.000f, 0.667f, 0.667f},
                                                    {0.000f, 1.000f, 0.667f},
                                                    {0.000f, 0.333f, 1.000f},
                                                    {0.000f, 0.667f, 1.000f},
                                                    {0.000f, 1.000f, 1.000f},
                                                    {0.500f, 0.333f, 0.000f},
                                                    {0.500f, 0.667f, 0.000f},
                                                    {0.500f, 1.000f, 0.000f},
                                                    {0.500f, 0.000f, 0.333f},
                                                    {0.500f, 0.333f, 0.333f},
                                                    {0.500f, 0.667f, 0.333f},
                                                    {0.500f, 1.000f, 0.333f},
                                                    {0.500f, 0.000f, 0.667f},
                                                    {0.500f, 0.333f, 0.667f},
                                                    {0.500f, 0.667f, 0.667f},
                                                    {0.500f, 1.000f, 0.667f},
                                                    {0.500f, 0.000f, 1.000f},
                                                    {0.500f, 0.333f, 1.000f},
                                                    {0.500f, 0.667f, 1.000f},
                                                    {0.500f, 1.000f, 1.000f},
                                                    {1.000f, 0.333f, 0.000f},
                                                    {1.000f, 0.667f, 0.000f},
                                                    {1.000f, 1.000f, 0.000f},
                                                    {1.000f, 0.000f, 0.333f},
                                                    {1.000f, 0.333f, 0.333f},
                                                    {1.000f, 0.667f, 0.333f},
                                                    {1.000f, 1.000f, 0.333f},
                                                    {1.000f, 0.000f, 0.667f},
                                                    {1.000f, 0.333f, 0.667f},
                                                    {1.000f, 0.667f, 0.667f},
                                                    {1.000f, 1.000f, 0.667f},
                                                    {1.000f, 0.000f, 1.000f},
                                                    {1.000f, 0.333f, 1.000f},
                                                    {1.000f, 0.667f, 1.000f},
                                                    {0.000f, 0.000f, 0.333f},
                                                    {0.000f, 0.000f, 0.500f},
                                                    {0.000f, 0.000f, 0.667f},
                                                    {0.000f, 0.000f, 0.833f},
                                                    {0.000f, 0.000f, 1.000f},
                                                    {0.000f, 0.167f, 0.000f},
                                                    {0.000f, 0.333f, 0.000f},
                                                    {0.000f, 0.500f, 0.000f},
                                                    {0.000f, 0.667f, 0.000f},
                                                    {0.000f, 0.833f, 0.000f},
                                                    {0.000f, 1.000f, 0.000f},
                                                    {0.167f, 0.000f, 0.000f},
                                                    {0.333f, 0.000f, 0.000f},
                                                    {0.500f, 0.000f, 0.000f},
                                                    {0.667f, 0.000f, 0.000f},
                                                    {0.833f, 0.000f, 0.000f},
                                                    {1.000f, 0.000f, 0.000f},
                                                    {0.000f, 0.000f, 0.000f},
                                                    {0.143f, 0.143f, 0.143f},
                                                    {0.286f, 0.286f, 0.286f},
                                                    {0.429f, 0.429f, 0.429f},
                                                    {0.571f, 0.571f, 0.571f},
                                                    {0.714f, 0.714f, 0.714f},
                                                    {0.857f, 0.857f, 0.857f},
                                                    {0.741f, 0.447f, 0.000f},
                                                    {0.741f, 0.717f, 0.314f},
                                                    {0.000f, 0.500f, 0.500f}};