from PIL import Image
from torchvision.transforms import transforms

# NUM_PRIVATE_PROPS = 3
# NUM_PRIVATE_PROPS = 1
# MIX_RATIO = [1.0, 0.8, 0.6, 0.4, 0]
# MIX_RATIO = [1.0]
# celeba_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# celeba_transform = transforms.Compose([
#             transforms.Resize(84),
#             transforms.CenterCrop(84),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
class NumpyToPIL:
    def __call__(self, numpy_array):
        pil_image = Image.fromarray(numpy_array)
        return pil_image

celeba_transform = transforms.Compose([
    NumpyToPIL(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# celeba_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])
# celeba_transform = transforms.Compose([
#             transforms.Resize(84),
#             transforms.CenterCrop(84),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
# INPUT_SIZE = [112, 112]
# RGB_MEAN = [0.5, 0.5, 0.5]
# RGB_STD = [0.5, 0.5, 0.5]
# EMBEDDING_SIZE = 512
# WEIGHT_DECAY = 5e-4
# LR = 0.01
# MOMENTUM = 0.9
# NUM_CLASSES = 50
# DROP_LAST = True
# celeba_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=RGB_MEAN,
#                          std=RGB_STD),
# ])

# POSITIVE_ITERS = 50
# NEGATIVE_ITERS = POSITIVE_ITERS * len([item for item in MIX_RATIO if item > 0])
# PID_2_NAME = {0: 'Gender', 1: 'Smiling'}
PID_2_NAME = \
    {0: '5_o_Clock_Shadow',
     1: 'Arched_Eyebrows',
     2: 'Attractive',
     3: 'Bags_Under_Eyes',
     4: 'Bald',
     5: 'Bangs',
     6: 'Big_Lips',
     7: 'Big_Nose',
     8: 'Black_Hair',
     9: 'Blond_Hair',
     10: 'Blurry',
     11: 'Brown_Hair',
     12: 'Bushy_Eyebrows',
     13: 'Chubby',
     14: 'Double_Chin',
     15: 'Eyeglasses',
     16: 'Goatee',
     17: 'Gray_Hair',
     18: 'Heavy_Makeup',
     19: 'High_Cheekbones',
     20: 'Male',
     21: 'Mouth_Slightly_Open',
     22: 'Mustache',
     23: 'Narrow_Eyes',
     24: 'No_Beard',
     25: 'Oval_Face',
     26: 'Pale_Skin',
     27: 'Pointy_Nose',
     28: 'Receding_Hairline',
     29: 'Rosy_Cheeks',
     30: 'Sideburns',
     31: 'Smiling',
     32: 'Straight_Hair',
     33: 'Wavy_Hair',
     34: 'Wearing_Earrings',
     35: 'Wearing_Hat',
     36: 'Wearing_Lipstick',
     37: 'Wearing_Necklace',
     38: 'Wearing_Necktie',
     39: 'Young'}