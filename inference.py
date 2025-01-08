import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from core.utils.misc import process_cfg
from utils import flow_viz

from core.Networks import build_network
from utils.utils import InputPadder, forward_interpolate
import imageio

print("torch.cuda.is_available() ", torch.cuda.is_available())

def prepare_image(seq_dir, batch_size):
    """
    Prépare les images par lots.
    """
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    image_list = sorted([
        f for f in os.listdir(seq_dir)
        if os.path.isfile(os.path.join(seq_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])  
    # print("images_foun" , image_list)  
    images = []

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)

        # Si on atteint la taille du batch, on le renvoie
        if len(images) == batch_size:
            yield torch.stack(images)  # Renvoie un batch
            images = []

    # Renvoyer les images restantes si elles ne forment pas un batch complet
    if images:
        yield torch.stack(images)

def vis_pre(flow_pre, vis_dir):
    """
    Sauvegarde les visualisations de flux optique.
    """
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]
    # print("N = ", N)
    # print("flow_pre.shape = ", flow_pre.shape)
    for idx in range(N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save(f'{vis_dir}/flow_{idx+1:04}_to_{idx+2:04}.png')
    # for idx in range(N // 2):
    #     flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
    #     image = Image.fromarray(flow_img)
    #     image.save(f'{vis_dir}/flow_{idx+2:04}_to_{idx+3:04}.png')
    
    # for idx in range(N // 2, N):
    #     flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
    #     image = Image.fromarray(flow_img)
    #     image.save(f'{vis_dir}/flow_{idx-N//2+2:04}_to_{idx-N//2+1:04}.png')

@torch.no_grad()
def MOF_inference(model, cfg, batch_size=4):
    """
    Inférence en mode MOF, avec traitement par lots.
    """
    model.eval()
    all_flows = []

    # Charger les images par lots
    for batch in prepare_image(cfg.seq_dir, batch_size=batch_size):
        if batch.shape[0] < batch_size:  # Vérifier si le batch est incomplet
            print(f"Ignoring last batch of size {batch.shape[0]}")
            continue  # Ignorer ce batch

        batch = batch[None].cuda()  # Ajouter une dimension pour le batch global
        padder = InputPadder(batch.shape)
        batch = padder.pad(batch)
        # print(batch.shape)

        # Faire une inférence sur le lot
        flow_pre, _ = model(batch, {})
        flow_pre = padder.unpad(flow_pre[0]).cpu()

        # Ajouter les résultats à la liste
        all_flows.append(flow_pre)

    # Concaténer tous les flux optiques calculés
    return torch.cat(all_flows, dim=0)

@torch.no_grad()
def BOF_inference(model, cfg, batch_size=4):
    """
    Inférence en mode BOF, avec traitement par lots.
    """
    model.eval()
    all_flows = []

    for batch in prepare_image(cfg.seq_dir, batch_size=batch_size):
        batch = batch[None].cuda()
        padder = InputPadder(batch.shape)
        batch = padder.pad(batch)
        flow_pre, _ = model(batch, {})
        flow_pre = padder.unpad(flow_pre[0]).cpu()

        all_flows.append(flow_pre)

    return torch.cat(all_flows, dim=0)

def count_parameters(model):
    """
    Compte les paramètres d'un modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    parser.add_argument('--batch_size', type=int, default=4, help='Taille des lots (batch size)')
    
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    with torch.no_grad():
        if args.mode == 'MOF':
            flow_pre = MOF_inference(model.module, cfg, batch_size=args.batch_size)
        elif args.mode == 'BOF':
            flow_pre = BOF_inference(model.module, cfg, batch_size=args.batch_size)
    
    vis_pre(flow_pre, cfg.vis_dir)




# import sys
# sys.path.append('core')

# from PIL import Image
# import argparse
# import os
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from core.utils.misc import process_cfg
# from utils import flow_viz

# from core.Networks import build_network

# from utils import frame_utils
# from utils.utils import InputPadder, forward_interpolate
# import itertools
# import imageio

# print("torch.cuda.is_available() " , torch.cuda.is_available() )
# def prepare_image(seq_dir):
#     print(f"preparing image...")
#     print(f"Input image sequence dir = {seq_dir}")

#     images = []

#     image_list = sorted(os.listdir(seq_dir))

#     for fn in image_list:
#         img = Image.open(os.path.join(seq_dir, fn))
#         img = np.array(img).astype(np.uint8)[..., :3]
#         img = torch.from_numpy(img).permute(2, 0, 1).float()
#         images.append(img)
    
#     return torch.stack(images)

# def vis_pre(flow_pre, vis_dir):

#     if not os.path.exists(vis_dir):
#         os.makedirs(vis_dir)

#     N = flow_pre.shape[0]

#     for idx in range(N//2):
#         flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
#         image = Image.fromarray(flow_img)
#         image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
#     for idx in range(N//2, N):
#         flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
#         image = Image.fromarray(flow_img)
#         image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

# @torch.no_grad()
# def MOF_inference(model, cfg):

#     model.eval()

#     input_images = prepare_image(cfg.seq_dir)
#     input_images = input_images[None].cuda()
#     padder = InputPadder(input_images.shape)
#     input_images = padder.pad(input_images)
#     flow_pre, _ = model(input_images, {})
#     flow_pre = padder.unpad(flow_pre[0]).cpu()

#     return flow_pre

# @torch.no_grad()
# def BOF_inference(model, cfg):

#     model.eval()

#     input_images = prepare_image(cfg.seq_dir)
#     input_images = input_images[None].cuda()
#     padder = InputPadder(input_images.shape)
#     input_images = padder.pad(input_images)
#     flow_pre, _ = model(input_images, {})
#     flow_pre = padder.unpad(flow_pre[0]).cpu()

#     return flow_pre

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', default='MOF')
#     parser.add_argument('--seq_dir', default='default')
#     parser.add_argument('--vis_dir', default='default')
    
#     args = parser.parse_args()

#     if args.mode == 'MOF':
#         from configs.multiframes_sintel_submission import get_cfg
#     elif args.mode == 'BOF':
#         from configs.sintel_submission import get_cfg

#     cfg = get_cfg()
#     cfg.update(vars(args))

#     model = torch.nn.DataParallel(build_network(cfg))
#     model.load_state_dict(torch.load(cfg.model))

#     model.cuda()
#     model.eval()

#     print(cfg.model)
#     print("Parameter Count: %d" % count_parameters(model))
    
#     with torch.no_grad():
#         if args.mode == 'MOF':
#             from configs.multiframes_sintel_submission import get_cfg
#             flow_pre = MOF_inference(model.module, cfg)
#         elif args.mode == 'BOF':
#             from configs.sintel_submission import get_cfg
#             flow_pre = BOF_inference(model.module, cfg)
    
#     vis_pre(flow_pre, cfg.vis_dir)



