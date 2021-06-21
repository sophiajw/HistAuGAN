
import random
import torch

from torchvision import transforms

from mdmm.model import MD_multi

class RandomRotate90:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)
    
# no scaling/deformation yet
basic_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
    transforms.RandomApply((transforms.ColorJitter(brightness=0.1, contrast=0.1), ), p=0.5),
#     transforms.Normalize(mean, std),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

mdmm_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90([0, 90, 180, 270]),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(),
    ])


geom_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90([0, 90, 180, 270]),
#         transforms.Normalize(mean, std),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(),
    ])


color_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90([0, 90, 180, 270]),
        transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
        transforms.RandomApply((transforms.ColorJitter(brightness=0.1, contrast=0.1), ), p=0.5),
        transforms.RandomApply((transforms.ColorJitter(saturation=0.5, hue=0.5), ), p=0.5),
#         transforms.Normalize(mean, std),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(),
    ])

color_augmentations_light = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90([0, 90, 180, 270]),
        transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
        transforms.RandomApply((transforms.ColorJitter(brightness=0.1, contrast=0.1), ), p=0.5),
        transforms.RandomApply((transforms.ColorJitter(saturation=0.1, hue=0.1), ), p=0.5),
#         transforms.Normalize(mean, std),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(),
    ])

no_augmentations = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def normalization(center):
    assert center in range(5), 'center not valid, should be in range(5)'
    mean = [
        [0.6710, 0.5327, 0.6448],
        [0.6475, 0.5139, 0.6222],
        [0.7875, 0.6251, 0.7567],
        [0.4120, 0.3270, 0.3959],
        [0.7324, 0.5814, 0.7038]
    ]
    std = [
        [0.2083, 0.2294, 0.1771],
        [0.2060, 0.2261, 0.1754],
        [0.2585, 0.2679, 0.2269],
        [0.2605, 0.2414, 0.2394],
        [0.2269, 0.2450, 0.1950]
    ]

    return mean[center], std[center]


# options for the model, default arguments + commandline arguments
class Args:
    concat = 0
    crop_size = 216 # only used as an argument for training
    dis_norm = None
    dis_scale = 3
    dis_spectral_norm = False
    dataroot ='data'
    gpu = 1
    input_dim = 3
    isDcontent = False
    nThreads = 4
    num_domains = 5
    nz = 8
    resume = False
#     resume ='/home/haicu/sophia.wagner/projects/stain_color/mdmm_model/model_file.pth'
opts = Args()

mean_domains = [
torch.tensor([ 0.3020, -2.6476, -0.9849, -0.7820, -0.2746,  0.3361,  0.1694, -1.2148]),
torch.tensor([ 0.1453, -1.2400, -0.9484,  0.9697, -2.0775,  0.7676, -0.5224, -0.2945]),
torch.tensor([ 2.1067, -1.8572,  0.0055,  1.2214, -2.9363,  2.0249, -0.4593, -0.9771]),
torch.tensor([ 0.8378, -2.1174, -0.6531,  0.2986, -1.3629, -0.1237, -0.3486, -1.0716]),
torch.tensor([ 1.6073,  1.9633, -0.3130, -1.9242, -0.9673,  2.4990, -2.2023, -1.4109]),
]

std_domains = [
torch.tensor([0.6550, 1.5427, 0.5444, 0.7254, 0.6701, 1.0214, 0.6245, 0.6886]),
torch.tensor([0.4143, 0.6543, 0.5891, 0.4592, 0.8944, 0.7046, 0.4441, 0.3668]),
torch.tensor([0.5576, 0.7634, 0.7875, 0.5220, 0.7943, 0.8918, 0.6000, 0.5018]),
torch.tensor([0.4157, 0.4104, 0.5158, 0.3498, 0.2365, 0.3612, 0.3375, 0.4214]),
torch.tensor([0.6154, 0.3440, 0.7032, 0.6220, 0.4496, 0.6488, 0.4886, 0.2989]),
]

def generate_new_stain(img, img_domain, new_domain_idx, random_attribute=True):
    """
    Generates a new stain color for the input image img.
    
    :img: input image of shape (3, 216, 216)
    :img_domain: vector of shape (1, opts.num_domain) indicating the domain of img
    :new_domain_idx: int in range(opts.num_domain) that indicates the domain where img should projecte to
    :random_attribute: default True, indicates whether original attribute vector should be used or a 
                      randomly drawn attribute vector
    """
    # create attribute encodings
    z_random = get_z_random(1, opts.nz, 'gauss')
    z_random = z_random * std_domains[new_domain_idx] + mean_domains[new_domain_idx]
    z_attr = z_random
    
    # domain vector
    domain = torch.eye(5)[new_domain_idx].unsqueeze(0)
    
    # create content encoding
    z_content = model.enc_c.forward(img.unsqueeze(0))
    
    out = model.gen.forward(z_content, z_attr, domain)
    
    return out

def generate_hist_augs(img, img_domain, model, z_content=None, same_attribute=False, new_domain=None, stats=None):
    """
    Generates a new stain color for the input image img.
    
    :img: input image of shape (3, 216, 216) [type: torch.Tensor]
    :img_domain: int in range(5)
    :model: HistAuGAN model
    :z_content: content encoding, if None this will be computed from img
    :same_attribute: [type: bool] indicates whether the attribute encoding of img or a randomly generated attribute are used
    :new_domain: either int in range(5) or torch.Tensor of shape (1, 5)
    :stats: (mean, std dev) of the latent space of HistAuGAN
    """
    # compute content vector
    if z_content is None:
        z_content = model.enc_c(img.sub(0.5).mul(2).unsqueeze(0))
    
    # compute attribute 
    if same_attribute:
        mu, logvar = model.enc_a.forward(img.sub(0.5).mul(2).unsqueeze(0), torch.eye(5)[img_domain].unsqueeze(0))
        std = logvar.mul(0.5).exp_()
        eps = torch.randn((std.size(0), std.size(1)))
        z_attr = eps.mul(std).add_(mu)    
    elif same_attribute == False and stats is not None:
        z_attr = torch.randn((1, 8, )) * stats[1][i] + stats[0][i]
    else:
        z_attr = torch.randn((1, 8, ))
    
    # determine new domain vector
    if isinstance(new_domain, int) and new_domain in range(5):
        new_domain = torch.eye(5)[new_domain].unsqueeze(0)
    elif isinstance(new_domain, torch.Tensor) and new_domain.shape == (1, 5):
        new_domain = new_domain
    else:
        new_domain = torch.eye(5)[np.random.randint(5)].unsqueeze(0)
    
    # generate new histology image with same content as img
    out = model.gen(z_content, z_attr, new_domain).detach().squeeze(0) # in range [-1, 1]

    return out