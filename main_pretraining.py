import argparse
import os, time

import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

from episode_creation import Episode_Transformations, collate_function_notaskid, DeterministicEpisodes, ImageFolderDetEpisodes

from models.view_encoder import *
from models.action_encoder import Action_Encoder_Network
from models.seek import Seek_Network
from models.bind import Bind_Network
from models.generator import Generator_Network
from models.classifier import Classifier_Network

from utils import MetricLogger, accuracy, time_duration_print, build_stratified_indices, make_plot_batch, make_param_group, extract_crop_params_from_raw
from utils import seed_everything_with_fabric

import numpy as np
import json
from PIL import Image

from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Pre-training Dream Replay Bind')
### Dataset parameters
parser.add_argument('--data_path', type=str, default='/data/datasets/caltech256/256_ObjectCategories_splits')
parser.add_argument('--val_episode_seed', type=int, default=12345)
parser.add_argument('--num_classes', type=int, default=256)
parser.add_argument('--mean', type=list, default=[0.5, 0.5, 0.5])
parser.add_argument('--std', type=list, default=[0.5, 0.5, 0.5])
parser.add_argument('--channels', type=int, default=4)
parser.add_argument('--only_crop', action='store_true', default=False)
### View encoder parameters
parser.add_argument('--venc_model', type=str, default='deit_tiny_patch16_LS')
parser.add_argument('--lr_venc', type=float, default=0.0008)
parser.add_argument('--wd_venc', type=float, default=0.05)
parser.add_argument('--venc_drop_path', type=float, default=0.0125) # 0.0125 for tiny, 0.05 for small, 0.2 for base
### Action encoder parameters
parser.add_argument('--lr_act_enc', type=float, default=0.0008)
parser.add_argument('--wd_act_enc', type=float, default=0)
parser.add_argument('--act_enc_dim', type=int, default=64)
parser.add_argument('--act_enc_n_layers', type=int, default=2)
parser.add_argument('--act_enc_n_heads', type=int, default=4)
### Seek parameters
parser.add_argument('--lr_seek', type=float, default=0.0008)
parser.add_argument('--wd_seek', type=float, default=0)
parser.add_argument('--seek_dim', type=int, default=192)
parser.add_argument('--seek_n_layers', type=int, default=8)
parser.add_argument('--seek_n_heads', type=int, default=8)
parser.add_argument('--seek_dropout', type=float, default=0)
parser.add_argument('--seek_gain_fields', action='store_true', default=False)
### Bind parameters
parser.add_argument('--lr_bind', type=float, default=0.0008)
parser.add_argument('--wd_bind', type=float, default=0.001)
parser.add_argument('--bind_dim', type=int, default=128)
parser.add_argument('--bind_n_layers', type=int, default=2)
parser.add_argument('--bind_n_heads', type=int, default=4)
parser.add_argument('--bind_dropout', type=float, default=0)
### Generator parameters
parser.add_argument('--lr_gen', type=float, default=0.0008)
parser.add_argument('--wd_gen', type=float, default=0)
parser.add_argument('--gen_num_Blocks', type=list, default=[1,1,1,1])
### Classifier parameters
parser.add_argument('--lr_classifier', type=float, default=0.01)
parser.add_argument('--wd_classifier', type=float, default=0)
### Training parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--episode_batch_size', type=int, default=96)
parser.add_argument('--num_views', type=int, default=4)
parser.add_argument('--coeff_mse', type=float, default=1.0)
parser.add_argument('--coeff_bce', type=float, default=1.0)
parser.add_argument('--freeze_view_encoder', action='store_true', default=False)
parser.add_argument('--ema_momentum_view_encoder', type=float, default=None) # None for no EMA
### Other parameters
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--save_dir', type=str, default="output/Pretraining/run_debug")
parser.add_argument('--print_frequency', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--zoom_min', type=float, default=0.08)
parser.add_argument('--zoom_max', type=float, default=0.5)

@torch.no_grad()
def update_ema(student, teacher, momentum: float):
    """EMA update: teacher = m * teacher + (1 - m) * student."""
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

def main():

    ### Parse arguments
    args = parser.parse_args()

    ### Rise error if EMA momentum is not None and freeze view encoder is True
    if args.ema_momentum_view_encoder is not None and args.freeze_view_encoder:
        raise ValueError("EMA momentum is not None and freeze view encoder is True. This is not allowed.")

    ### Create save dir folder
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    ### Define loggers
    tb_logger = TensorBoardLogger(root_dir=os.path.join(args.save_dir, "logs"), name="tb_logs")
    csv_logger = CSVLogger(root_dir=os.path.join(args.save_dir, "logs"), name="csv_logs",  flush_logs_every_n_steps=1)

    ### Define Fabric and launch it
    fabric = Fabric(accelerator="gpu", strategy="ddp", devices="auto", precision="bf16-mixed", loggers=[tb_logger, csv_logger])
    # fabric = Fabric(accelerator="gpu", strategy="auto", devices=1, precision="bf16-mixed", loggers=[tb_logger, csv_logger])
    fabric.launch()

    ### Seed everything
    seed_everything_with_fabric(args.seed, fabric)

    ### If 4 channel, add a 4th dimension to mean and std with 0 and 1 respectively
    if args.channels == 4:
        args.mean = args.mean + [0.0]
        args.std = args.std + [1.0]

    ### Print args
    fabric.print(args)

    ### Save args
    if fabric.is_global_zero:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    args.workers = int(args.workers / fabric.world_size)
    args.episode_batch_size = int(args.episode_batch_size / fabric.world_size)

    ### Load Training data
    fabric.print('\n==> Preparing Training data...')
    traindir = os.path.join(args.data_path, 'train')
    train_transform = Episode_Transformations(num_views = args.num_views, mean = args.mean, std = args.std, channels=args.channels,
                                              zoom_range = (args.zoom_min, args.zoom_max), only_crop=args.only_crop)
    train_dataset = ImageFolder(traindir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True,
                                               collate_fn=collate_function_notaskid)
    train_loader = fabric.setup_dataloaders(train_loader)

    ### Load Validation data
    fabric.print('\n==> Preparing Validation data...')
    valdir = os.path.join(args.data_path, 'val')
    val_base_transform = Episode_Transformations(num_views=args.num_views, mean=args.mean, std=args.std, channels=args.channels,
                                                 zoom_range = (args.zoom_min, args.zoom_max), only_crop=args.only_crop)
    val_transform = DeterministicEpisodes(val_base_transform, base_seed=args.val_episode_seed)
    val_dataset = ImageFolderDetEpisodes(valdir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.episode_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, persistent_workers=True,
                                             collate_fn=collate_function_notaskid)
    val_loader = fabric.setup_dataloaders(val_loader)

    ### Define models
    fabric.print('\n==> Prepare models...')
    view_encoder = eval(args.venc_model)(drop_path_rate=args.venc_drop_path, in_chans=args.channels)
    action_encoder = Action_Encoder_Network(d_model=args.act_enc_dim, 
                                            n_layers=args.act_enc_n_layers, 
                                            n_heads=args.act_enc_n_heads,
                                            only_crop=args.only_crop)
    seek = Seek_Network(d_model=args.seek_dim,
                        imgfttok_dim=view_encoder.embed_dim, 
                        acttok_dim=args.act_enc_dim,
                        num_layers=args.seek_n_layers, 
                        nhead=args.seek_n_heads,
                        dropout=args.seek_dropout,
                        use_gain_fields=args.seek_gain_fields)
    bind = Bind_Network(d_model=args.bind_dim,
                        num_queries=view_encoder.num_patches, # 14x14 canvas for deit_tiny_patch16_LS (196 patches)
                        imgfttok_dim=view_encoder.embed_dim,
                        acttok_dim=args.act_enc_dim,
                        num_layers=args.bind_n_layers,
                        nhead=args.bind_n_heads,
                        dropout=args.bind_dropout)
    generator = Generator_Network(in_planes=view_encoder.embed_dim, 
                                  num_Blocks=args.gen_num_Blocks, 
                                  nc=args.channels)
    classifier = Classifier_Network(input_dim=args.bind_dim, num_classes=args.num_classes)

    if args.ema_momentum_view_encoder is not None:
        ### Define EMA view encoder (no gradients, used only to define targets)
        view_encoder_ema = copy.deepcopy(view_encoder)
        for p in view_encoder_ema.parameters():
            p.requires_grad = False
        view_encoder_ema.eval()
        # EMA view encoder lives on device but is not wrapped with DDP or optimizer
        view_encoder_ema.to(fabric.device)

                                                  
    ### Print models
    fabric.print('\nView encoder')
    fabric.print(view_encoder)
    fabric.print('\nAction encoder')
    fabric.print(action_encoder)
    fabric.print('\nSeek')
    fabric.print(seek)
    fabric.print('\nBind')
    fabric.print(bind)
    fabric.print('\nGenerator')
    fabric.print(generator)
    fabric.print('\nClassifier')
    fabric.print(classifier)
    fabric.print('\n')

    ### Compile models
    view_encoder = torch.compile(view_encoder)
    # action_encoder = torch.compile(action_encoder) # It uses lists inside. It is tricky to compile.
    # seek = torch.compile(seek) # Having variable masking ratio is tricky to compile.
    bind = torch.compile(bind)
    # generator = torch.compile(generator) # Can't compile the upsampling layers.
    classifier = torch.compile(classifier)

    ### Setup models
    view_encoder = fabric.setup_module(view_encoder)
    action_encoder = fabric.setup_module(action_encoder)
    seek = fabric.setup_module(seek)
    bind = fabric.setup_module(bind)
    generator = fabric.setup_module(generator)
    classifier = fabric.setup_module(classifier)

    ### Define optimizers
    view_encoder_param_group = make_param_group(view_encoder, lr=args.lr_venc, weight_decay=args.wd_venc, no_weight_decay_list=getattr(view_encoder, "no_weight_decay", lambda: set())())
    action_encoder_param_group = make_param_group(action_encoder, lr=args.lr_act_enc, weight_decay=args.wd_act_enc, no_weight_decay_list=getattr(action_encoder, "no_weight_decay", lambda: set())())
    seek_param_group = make_param_group(seek, lr=args.lr_seek, weight_decay=args.wd_seek, no_weight_decay_list=getattr(seek, "no_weight_decay", lambda: set())())
    bind_param_group = make_param_group(bind, lr=args.lr_bind, weight_decay=args.wd_bind, no_weight_decay_list=getattr(bind, "no_weight_decay", lambda: set())())
    generator_param_group = make_param_group(generator, lr=args.lr_gen, weight_decay=args.wd_gen, no_weight_decay_list=getattr(generator, "no_weight_decay", lambda: set())())
    classifier_param_group = make_param_group(classifier, lr=args.lr_classifier, weight_decay=args.wd_classifier, no_weight_decay_list=getattr(classifier, "no_weight_decay", lambda: set())())
    
    param_groups = view_encoder_param_group + action_encoder_param_group + seek_param_group + bind_param_group + generator_param_group + classifier_param_group
    
    optimizer = torch.optim.AdamW(param_groups, lr=0, weight_decay=0)
    
    ### Setup optimizers
    optimizer = fabric.setup_optimizers(optimizer)

    ### Define schedulers
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    # 2 LR drops at ~60% and ~85% of training (in steps, measured AFTER warmup starts)
    # m1 = max(int(0.60 * total_steps) - warmup_steps, 1)
    # m2 = max(int(0.85 * total_steps) - warmup_steps, m1 + 1)
    # multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_scheduler, multi_step_scheduler], milestones=[warmup_steps])

    ### Save one batch for plot purposes
    seed_everything_with_fabric(args.seed, fabric)  # Reset seed to ensure reproducibility for the plot batch
    if fabric.is_global_zero:
        PLOT_N = 16
        # Training batch
        plot_indices = build_stratified_indices(train_dataset, PLOT_N)
        episodes_plot_train, _ = make_plot_batch(train_dataset, plot_indices, collate_function_notaskid)
        # Validation batch
        plot_indices = build_stratified_indices(val_dataset, PLOT_N)
        episodes_plot_val, _ = make_plot_batch(val_dataset, plot_indices, collate_function_notaskid)
        # Train batch plot: Quickly plot the first episode to see if it is correct (not generated images, just plot directly the images)
        episode_0_imgs = episodes_plot_train[0][0]
        episode_0_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_0_imgs]
        episode_0_imgs = torch.stack(episode_0_imgs, dim=0)
        if args.channels == 4:
            episode_0_imgs = episode_0_imgs[:, :3, :, :] # (V, 3, H, W)
        grid = torchvision.utils.make_grid(episode_0_imgs, nrow=args.num_views)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(os.path.join(args.save_dir, 'episode_0_imgs_train.png'))
        # Validation batch plot: Quickly plot the first episode to see if it is correct (not generated images, just plot directly the images)
        episode_0_imgs = episodes_plot_val[0][0]
        episode_0_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_0_imgs]
        episode_0_imgs = torch.stack(episode_0_imgs, dim=0)
        if args.channels == 4:
            episode_0_imgs = episode_0_imgs[:, :3, :, :] # (V, 3, H, W)
        grid = torchvision.utils.make_grid(episode_0_imgs, nrow=args.num_views)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = Image.fromarray(grid)
        grid.save(os.path.join(args.save_dir, 'episode_0_imgs_val.png'))
    fabric.barrier()


    #### Train and Validation loop ####
    fabric.print('\n==> Training and Validating model')
    init_time = time.time()

    # Seed right before training and validation starts
    seed_everything_with_fabric(args.seed, fabric)

    for epoch in range(args.epochs):
        start_time = time.time()
        fabric.print(f'\n==> Epoch {epoch}/{args.epochs}')

        ##################
        ### Train STEP ###
        ##################

        train_loss_BCE = MetricLogger('Train Loss BCE')
        train_loss_MSE_1 = MetricLogger('Train Loss MSE 1')
        train_loss_MSE_2 = MetricLogger('Train Loss MSE 2')
        train_loss_MSE_total = MetricLogger('Train Loss MSE Total')
        train_loss_total = MetricLogger('Train Loss Total')
        train_acc1 = MetricLogger('Train Top1 ACC')
        train_acc5 = MetricLogger('Train Top5 ACC')

        # if epoch < 0:
        if args.freeze_view_encoder:
            view_encoder.eval()
            for param in view_encoder.parameters():
                param.requires_grad = False
        else:
            view_encoder.train()
        action_encoder.train()
        seek.train()
        bind.train()
        generator.train()
        classifier.train()

        for i, (batch_episodes, batch_labels) in enumerate(train_loader):
            batch_episodes_imgs = batch_episodes[0] # (B, V, C, H, W)
            batch_episodes_actions = batch_episodes[1] # (B, V, A)
            B, V, C, H, W = batch_episodes_imgs.shape
            flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)

            if args.ema_momentum_view_encoder is not None:
                # EMA View Encoder forward pass – uses this as targets for the MSE loss
                with torch.no_grad():
                    flat_imgfttoks_ema_targets, _ = view_encoder_ema(flat_imgs)
                noflat_imgfttoks_ema_targets = flat_imgfttoks_ema_targets.view(B, V, flat_imgfttoks_ema_targets.size(1), -1)

            # View Encoder forward pass
            flat_imgfttoks, flat_ret2D = view_encoder(flat_imgs) # (B*V, Timg, D)
            noflat_imgfttoks = flat_imgfttoks.view(B, V, flat_imgfttoks.size(1), -1) # (B, V, Timg, Dimg)
            noflat_ret2D = flat_ret2D.view(B, V, flat_ret2D.size(1), -1) # (B, V, Timg, 2)

            # Action Encoder forward pass
            flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)] # list length B*V
            flat_acttok = action_encoder(flat_actions) # (B*V, 1, D)
            noflat_acttok = flat_acttok.view(B, V, flat_acttok.size(1), -1) # (B, V, 1, D)

            # Seek forward pass (first view output is a bunch of zeros here. We are not predicting it. It is always available inside the transformer)
            if args.seek_gain_fields: crop_bv = extract_crop_params_from_raw(batch_episodes_actions, device=noflat_imgfttoks.device) # (B,V,4)
            else: crop_bv = None
            noflat_PRED_imgfttoks, mask_indices = seek(noflat_acttok, noflat_imgfttoks, noflat_ret2D, crop_bv) # (B, V, Timg, Dimg), (Timg)

            if fabric.is_global_zero and ((i % args.print_frequency) == 0):
                for j in range(generator.alpha.squeeze().size(0)):
                    fabric.log(name=f'Generator Alpha dim {j+1}', value=generator.alpha.squeeze()[j].item(), step=epoch*len(train_loader)+i)
                    fabric.log(name=f'Generator Gamma dim {j+1}', value=generator.gamma.squeeze()[j].item(), step=epoch*len(train_loader)+i)

            # Generator + View Encoder forward pass
            noflat_PRED_imgs = generator(noflat_PRED_imgfttoks)
            flat_PRED_imgs = noflat_PRED_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
            if args.ema_momentum_view_encoder is not None:
                flat_PRED2_imgfttoks, _ = view_encoder_ema(flat_PRED_imgs) # (B*V, Timg, D) # This one uses the EMA view encoder
            else:
                flat_PRED2_imgfttoks, _ = view_encoder(flat_PRED_imgs) # (B*V, Timg, D) # This one uses the student view encoder
            noflat_PRED2_imgfttoks = flat_PRED2_imgfttoks.view(B, V, flat_PRED2_imgfttoks.size(1), -1) # (B, V, Timg, Dimg)

            # Bind forward pass
            canvas = bind(noflat_acttok, noflat_imgfttoks, noflat_ret2D, batch_episodes_actions) # (B, num_queries, Dhidden)

            # Classifier forward pass
            logits = classifier(canvas) # (B, K) -> It outputs a logit per episode.

            with fabric.autocast(): # Run losses calculations in mixed precision (models already run in mixed precision)
                if args.ema_momentum_view_encoder is not None:
                    # Reconstruction losses in latent space (with ema)
                    noflat_imgfttoks_ema_targets_detach = noflat_imgfttoks_ema_targets.detach()
                    loss_mse_1 = F.mse_loss(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_ema_targets_detach[:, :, mask_indices, :])
                    loss_mse_2 = F.mse_loss(noflat_PRED2_imgfttoks, noflat_imgfttoks_ema_targets_detach)
                else:
                    # Reconstruction losses in latent space (no ema)
                    noflat_imgfttoks_detach = noflat_imgfttoks.detach()
                    loss_mse_1 = F.mse_loss(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                    loss_mse_2 = F.mse_loss(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)

                # BCE loss
                batch_labels_onehot = F.one_hot(batch_labels, num_classes=args.num_classes).float() # (B, K)
                pos_weight = torch.full((args.num_classes,), fill_value=args.num_classes-1, dtype=torch.float32, device=logits.device) # Weight for the positive class
                loss_bce = F.binary_cross_entropy_with_logits(logits, batch_labels_onehot, pos_weight=pos_weight)

                # Calculate Total loss for the batch
                loss_mse_total = loss_mse_1 + loss_mse_2
                loss_total = args.coeff_mse * loss_mse_total + args.coeff_bce * loss_bce

                # Classification accuracy (works for a non multi-label classification task)
                acc1, acc5 = accuracy(logits, batch_labels, topk=(1, 5))

            ## Backward pass with clip norm
            optimizer.zero_grad()
            fabric.backward(loss_total)
            fabric.clip_gradients(view_encoder, optimizer, max_norm=1.0)
            fabric.clip_gradients(action_encoder, optimizer, max_norm=1.0)
            fabric.clip_gradients(seek, optimizer, max_norm=1.0)
            fabric.clip_gradients(bind, optimizer, max_norm=1.0)
            fabric.clip_gradients(generator, optimizer, max_norm=1.0)
            fabric.clip_gradients(classifier, optimizer, max_norm=1.0)
            optimizer.step()

            if args.ema_momentum_view_encoder is not None:
                ## Update EMA view encoder
                update_ema(view_encoder, view_encoder_ema, args.ema_momentum_view_encoder)

            ## Update scheduler
            scheduler.step()

            ## Track metrics
            B_allranks = fabric.all_reduce(torch.tensor(B, device=fabric.device), reduce_op="sum").item()
            train_loss_total.update(fabric.all_reduce(loss_total.detach(), reduce_op="mean").item(), B_allranks)
            train_loss_MSE_total.update(fabric.all_reduce(loss_mse_total.detach(), reduce_op="mean").item(), B_allranks)
            train_loss_BCE.update(fabric.all_reduce(loss_bce.detach(), reduce_op="mean").item(), B_allranks)
            train_loss_MSE_1.update(fabric.all_reduce(loss_mse_1.detach(), reduce_op="mean").item(), B_allranks)
            train_loss_MSE_2.update(fabric.all_reduce(loss_mse_2.detach(), reduce_op="mean").item(), B_allranks)
            train_acc1.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), B_allranks)
            train_acc5.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), B_allranks)

            ## Log and print training metrics per batch
            if fabric.is_global_zero and ((i % args.print_frequency) == 0):
                fabric.log(name=f'Loss Total', value=train_loss_total.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE Total', value=train_loss_MSE_total.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss BCE', value=train_loss_BCE.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE 1', value=train_loss_MSE_1.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Loss MSE 2', value=train_loss_MSE_2.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Top1 ACC', value=train_acc1.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'Top5 ACC', value=train_acc5.val, step=epoch*len(train_loader)+i)
                fabric.log(name=f'lr_venc', value=scheduler.get_last_lr()[0], step=epoch*len(train_loader)+i)
                fabric.print(
                    f'Epoch [{epoch}] [{i}/{len(train_loader)}] -- '
                    f'Loss Total: {train_loss_total.val:.6f} -- ' 
                    f'Loss MSE Total: {train_loss_MSE_total.val:.6f} -- '
                    f'Loss BCE: {train_loss_BCE.val:.6f} -- '
                    f'Loss MSE 1: {train_loss_MSE_1.val:.6f} -- '
                    f'Loss MSE 2: {train_loss_MSE_2.val:.6f} -- '
                    f'Top1 ACC: {train_acc1.val:.3f} -- '
                    f'Top5 ACC: {train_acc5.val:.3f} -- '
                    f'lr_venc: {scheduler.get_last_lr()[0]:.6f}'
                    )

        ## Log and print training metrics per epoch
        fabric.log(name=f'Loss Total (per epoch)', value=train_loss_total.avg, step=epoch)
        fabric.log(name=f'Loss MSE Total (per epoch)', value=train_loss_MSE_total.avg, step=epoch)
        fabric.log(name=f'Loss BCE (per epoch)', value=train_loss_BCE.avg, step=epoch)
        fabric.log(name=f'Loss MSE 1 (per epoch)', value=train_loss_MSE_1.avg, step=epoch)
        fabric.log(name=f'Loss MSE 2 (per epoch)', value=train_loss_MSE_2.avg, step=epoch)
        fabric.log(name=f'Top1 ACC (per epoch)', value=train_acc1.avg, step=epoch)
        fabric.log(name=f'Top5 ACC (per epoch)', value=train_acc5.avg, step=epoch)
        fabric.print(
            f'Epoch [{epoch}] Train --> Loss Total: {train_loss_total.avg:.6f} -- '
            f'Loss MSE Total: {train_loss_MSE_total.avg:.6f} -- '
            f'Loss BCE: {train_loss_BCE.avg:.6f} -- '
            f'Loss MSE 1: {train_loss_MSE_1.avg:.6f} -- '
            f'Loss MSE 2: {train_loss_MSE_2.avg:.6f} -- '
            f'Top1 ACC: {train_acc1.avg:.3f} -- '
            f'Top5 ACC: {train_acc5.avg:.3f}'
            )

        ## Wait for all processes to finish the training step
        fabric.barrier()  # Wait for all processes to finish the training step



        #######################
        ### Validation STEP ###
        #######################

        val_loss_BCE = MetricLogger('Val Loss BCE')
        val_loss_MSE_1 = MetricLogger('Val Loss MSE 1')
        val_loss_MSE_2 = MetricLogger('Val Loss MSE 2')
        val_loss_MSE_total = MetricLogger('Val Loss MSE Total')
        val_loss_total = MetricLogger('Val Loss Total')
        val_acc1 = MetricLogger('Val Top1 ACC')
        val_acc5 = MetricLogger('Val Top5 ACC')

        view_encoder.eval()
        action_encoder.eval()
        seek.eval()
        bind.eval()
        generator.eval()
        classifier.eval()

        with torch.no_grad():
            for j, (batch_episodes, batch_labels) in enumerate(val_loader):
                batch_episodes_imgs = batch_episodes[0] # (B, V, C, H, W)
                batch_episodes_actions = batch_episodes[1]  # list of lists (B,V,ops)
                B, V, C, H, W = batch_episodes_imgs.shape

                # View Encoder
                flat_imgs = batch_episodes_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_imgfttoks, flat_ret2D = view_encoder(flat_imgs)  # (B*V, Timg, D)
                noflat_imgfttoks = flat_imgfttoks.view(B, V, flat_imgfttoks.size(1), -1) # (B, V, Timg, Dimg)
                noflat_ret2D = flat_ret2D.view(B, V, flat_ret2D.size(1), -1) # (B, V, Timg, 2)

                # Action Encoder
                flat_actions = [batch_episodes_actions[b][v] for b in range(B) for v in range(V)] # list length B*V
                flat_acttok = action_encoder(flat_actions) # (B*V, 1, D)
                noflat_acttok = flat_acttok.view(B, V, flat_acttok.size(1), -1) # (B, V, 1, D)

                # Seek: predict IMG tokens for each view
                if args.seek_gain_fields: crop_bv = extract_crop_params_from_raw(batch_episodes_actions, device=noflat_imgfttoks.device) # (B,V,4)
                else: crop_bv = None
                noflat_PRED_imgfttoks, mask_indices = seek(noflat_acttok, noflat_imgfttoks, noflat_ret2D, crop_bv) # (B, V, Timg, Dimg), (Timg)

                # Generator + Encoder (re-encode predicted images)
                noflat_PRED_imgs = generator(noflat_PRED_imgfttoks)
                flat_PRED_imgs = noflat_PRED_imgs.reshape(B * V, C, H, W) # (B*V, C, H, W)
                flat_PRED2_imgfttoks, _ = view_encoder(flat_PRED_imgs) # (B*V, Timg, D)
                noflat_PRED2_imgfttoks = flat_PRED2_imgfttoks.view(B, V, flat_PRED2_imgfttoks.size(1), -1) # (B, V, Timg, Dimg)

                ##############################################################################################################
                # On every first validation batch, let's plot mean token and std token (for patch and for action)
                if j==0 and fabric.is_global_zero:
                    # Plot patch tokens
                    tok_means = flat_imgfttoks.mean(dim=(0, 1)).detach().float().cpu().numpy()  # (D,)
                    tok_stds = flat_imgfttoks.std(dim=(0, 1), unbiased=False).detach().float().cpu().numpy()  # (D,)
                    tok_mean_abs = torch.abs(flat_imgfttoks).mean(dim=(0, 1)).detach().float().cpu().numpy()  # (D,)
                    dims = np.arange(tok_means.shape[0])
                    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
                    axs[0].bar(dims, tok_means)
                    axs[0].set_title('PatchToken (View encoder) mean per dimension')
                    axs[0].set_xlabel('Dimension')
                    axs[0].set_ylabel('Mean')
                    axs[1].bar(dims, tok_stds, color='orange')
                    axs[1].set_title('PatchToken (View encoder) std per dimension')
                    axs[1].set_xlabel('Dimension')
                    axs[1].set_ylabel('Std')
                    axs[2].bar(dims, tok_mean_abs)
                    axs[2].set_title('PatchToken (View encoder) mean absolute value per dimension')
                    axs[2].set_xlabel('Dimension')
                    axs[2].set_ylabel('Mean Absolute Value')
                    save_plot_dir = os.path.join(args.save_dir, 'Token_stats_val')
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    fig.savefig(os.path.join(save_plot_dir, f'epoch{epoch}_patchtoken_stats_viewencoder.png'))
                    plt.close(fig)
                    if epoch==0 or ((epoch+1) % 10 == 0):
                        # Also save the patch tokens as a numpy array
                        np.save(os.path.join(save_plot_dir, f'epoch{epoch}_patchtokens_tensor_viewencoder.npy'), flat_imgfttoks.detach().float().cpu().numpy())
                
                    # Plot patch tokens after seek
                    tok_means = noflat_PRED_imgfttoks.mean(dim=(0, 1, 2)).detach().float().cpu().numpy()  # (D,)
                    tok_stds = noflat_PRED_imgfttoks.std(dim=(0, 1, 2), unbiased=False).detach().float().cpu().numpy()  # (D,)
                    tok_mean_abs = torch.abs(noflat_PRED_imgfttoks).mean(dim=(0, 1, 2)).detach().float().cpu().numpy()  # (D,)
                    dims = np.arange(tok_means.shape[0])
                    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
                    axs[0].bar(dims, tok_means)
                    axs[0].set_title('PatchToken (Seek) mean per dimension')
                    axs[0].set_xlabel('Dimension')
                    axs[0].set_ylabel('Mean')
                    axs[1].bar(dims, tok_stds, color='orange')
                    axs[1].set_title('PatchToken (Seek) std per dimension')
                    axs[1].set_xlabel('Dimension')
                    axs[1].set_ylabel('Std')
                    axs[2].bar(dims, tok_mean_abs)
                    axs[2].set_title('PatchToken (Seek) mean absolute value per dimension')
                    axs[2].set_xlabel('Dimension')
                    axs[2].set_ylabel('Mean Absolute Value')
                    save_plot_dir = os.path.join(args.save_dir, 'Token_stats_val')
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    fig.savefig(os.path.join(save_plot_dir, f'epoch{epoch}_patchtoken_stats_seek.png'))
                    plt.close(fig)
                    if epoch==0 or ((epoch+1) % 10 == 0):
                        # Also save the patch tokens as a numpy array
                        np.save(os.path.join(save_plot_dir, f'epoch{epoch}_patchtokens_tensor_seek.npy'), noflat_PRED_imgfttoks.detach().float().cpu().numpy())
                    
                    # Plot patch token after generatorviewencoder
                    tok_means = noflat_PRED2_imgfttoks.mean(dim=(0, 1, 2)).detach().float().cpu().numpy()  # (D,)
                    tok_stds = noflat_PRED2_imgfttoks.std(dim=(0, 1, 2), unbiased=False).detach().float().cpu().numpy()  # (D,)
                    tok_mean_abs = torch.abs(noflat_PRED2_imgfttoks).mean(dim=(0, 1, 2)).detach().float().cpu().numpy()  # (D,)
                    dims = np.arange(tok_means.shape[0])
                    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
                    axs[0].bar(dims, tok_means)
                    axs[0].set_title('PatchToken (Generator+View encoder) mean per dimension')
                    axs[0].set_xlabel('Dimension')
                    axs[0].set_ylabel('Mean')
                    axs[1].bar(dims, tok_stds, color='orange')
                    axs[1].set_title('PatchToken (Generator+View encoder) std per dimension')
                    axs[1].set_xlabel('Dimension')
                    axs[1].set_ylabel('Std')
                    axs[2].bar(dims, tok_mean_abs)
                    axs[2].set_title('PatchToken (Generator+View encoder) mean absolute value per dimension')
                    axs[2].set_xlabel('Dimension')
                    axs[2].set_ylabel('Mean Absolute Value')
                    save_plot_dir = os.path.join(args.save_dir, 'Token_stats_val')
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    fig.savefig(os.path.join(save_plot_dir, f'epoch{epoch}_patchtoken_stats_generatorviewencoder.png'))
                    plt.close(fig)
                    if epoch==0 or ((epoch+1) % 10 == 0):
                        # Also save the patch tokens as a numpy array
                        np.save(os.path.join(save_plot_dir, f'epoch{epoch}_patchtokens_tensor_generatorviewencoder.npy'), noflat_PRED2_imgfttoks.detach().float().cpu().numpy())

                    # Plot action tokens
                    tok_means = flat_acttok.mean(dim=(0, 1)).detach().float().cpu().numpy()  # (D,)
                    tok_stds = flat_acttok.std(dim=(0, 1), unbiased=False).detach().float().cpu().numpy()  # (D,)
                    tok_mean_abs = torch.abs(flat_acttok).mean(dim=(0, 1)).detach().float().cpu().numpy()  # (D,)
                    dims = np.arange(tok_means.shape[0])
                    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
                    axs[0].bar(dims, tok_means)
                    axs[0].set_title('ActionToken (Action encoder) mean per dimension')
                    axs[0].set_xlabel('Dimension')
                    axs[0].set_ylabel('Mean')
                    axs[1].bar(dims, tok_stds, color='orange')
                    axs[1].set_title('ActionToken (Action encoder) std per dimension')
                    axs[1].set_xlabel('Dimension')
                    axs[1].set_ylabel('Std')
                    axs[2].bar(dims, tok_mean_abs)
                    axs[2].set_title('ActionToken (Action encoder) mean absolute value per dimension')
                    axs[2].set_xlabel('Dimension')
                    axs[2].set_ylabel('Mean Absolute Value')
                    save_plot_dir = os.path.join(args.save_dir, 'Token_stats_val')
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    fig.savefig(os.path.join(save_plot_dir, f'epoch{epoch}_actiontoken_stats_actionencoder.png'))
                    plt.close(fig)
                    if epoch==0 or ((epoch+1) % 10 == 0):
                        # Also save the action tokens as a numpy array
                        np.save(os.path.join(save_plot_dir, f'epoch{epoch}_actiontokens_tensor_actionencoder.npy'), flat_acttok.detach().float().cpu().numpy())


                # Bind forward pass
                canvas = bind(noflat_acttok, noflat_imgfttoks, noflat_ret2D, batch_episodes_actions) # (B, num_queries, Dhidden)

                # Classifier forward pass
                logits = classifier(canvas) # (B, K) -> It outputs a logit per episode.

                with fabric.autocast(): # Run losses calculations in mixed precision (models already run in mixed precision)
                    # Reconstruction losses in latent space
                    noflat_imgfttoks_detach = noflat_imgfttoks #.detach() No need to detach because it is for validation with torch.no_grad()
                    loss_mse_1 = F.mse_loss(noflat_PRED_imgfttoks[:, :, mask_indices, :], noflat_imgfttoks_detach[:, :, mask_indices, :])
                    loss_mse_2 = F.mse_loss(noflat_PRED2_imgfttoks, noflat_imgfttoks_detach)

                    # BCE loss
                    batch_labels_onehot = F.one_hot(batch_labels, num_classes=args.num_classes).float() # (B, K)
                    pos_weight = torch.full((args.num_classes,), fill_value=args.num_classes-1, dtype=torch.float32, device=logits.device) # Weight for the positive class
                    loss_bce = F.binary_cross_entropy_with_logits(logits, batch_labels_onehot, pos_weight=pos_weight)

                    # Calculate Total loss for the batch
                    loss_mse_total = loss_mse_1 + loss_mse_2
                    loss_total = args.coeff_mse * loss_mse_total + args.coeff_bce * loss_bce
                    # loss_total = args.coeff_mse * loss_mse_total

                    # Classification accuracy
                    acc1, acc5 = accuracy(logits, batch_labels, topk=(1, 5))

                ## Track metrics
                B_allranks = fabric.all_reduce(torch.tensor(B, device=fabric.device), reduce_op="sum").item()
                val_loss_total.update(fabric.all_reduce(loss_total.detach(), reduce_op="mean").item(), B_allranks)
                val_loss_MSE_total.update(fabric.all_reduce(loss_mse_total.detach(), reduce_op="mean").item(), B_allranks)
                val_loss_BCE.update(fabric.all_reduce(loss_bce.detach(), reduce_op="mean").item(), B_allranks)
                val_loss_MSE_1.update(fabric.all_reduce(loss_mse_1.detach(), reduce_op="mean").item(), B_allranks)
                val_loss_MSE_2.update(fabric.all_reduce(loss_mse_2.detach(), reduce_op="mean").item(), B_allranks)
                val_acc1.update(fabric.all_reduce(acc1.detach(), reduce_op="mean").item(), B_allranks)
                val_acc5.update(fabric.all_reduce(acc5.detach(), reduce_op="mean").item(), B_allranks)

        ## Log and print validation metrics per epoch
        fabric.log(name=f'Val Loss Total (per epoch)', value=val_loss_total.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE Total (per epoch)', value=val_loss_MSE_total.avg, step=epoch)
        fabric.log(name=f'Val Loss BCE (per epoch)', value=val_loss_BCE.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE 1 (per epoch)', value=val_loss_MSE_1.avg, step=epoch)
        fabric.log(name=f'Val Loss MSE 2 (per epoch)', value=val_loss_MSE_2.avg, step=epoch)
        fabric.log(name=f'Val Top1 ACC (per epoch)', value=val_acc1.avg, step=epoch)
        fabric.log(name=f'Val Top5 ACC (per epoch)', value=val_acc5.avg, step=epoch)
        fabric.print(
                f'Epoch [{epoch}] Val --> Loss Total: {val_loss_total.avg:.6f} -- '
                f'Loss MSE Total: {val_loss_MSE_total.avg:.6f} -- '
                f'Loss BCE: {val_loss_BCE.avg:.6f} -- '
                f'Loss MSE 1: {val_loss_MSE_1.avg:.6f} -- '
                f'Loss MSE 2: {val_loss_MSE_2.avg:.6f} -- '
                f'Top1 ACC: {val_acc1.avg:.3f} -- '
                f'Top5 ACC: {val_acc5.avg:.3f}'
            )

        ## Wait for all processes to finish the validation step
        fabric.barrier()  # Wait for all processes to finish the validation step

        ### Save models ###
        if (((epoch+1) % 10) == 0) or epoch==0: 
            state={
                "view_encoder": view_encoder,
                "action_encoder": action_encoder,
                "seek": seek,
                "bind": bind,
                "generator": generator,
                "classifier": classifier, 
                }
            fabric.save(os.path.join(args.save_dir, f'models_checkpoint_epoch{epoch}.pth'), state=state)

        ## Wait for all processes to finish the save models step
        fabric.barrier()  # Wait for all processes to finish the save models step


        ### Plot reconstructions examples ###
        if fabric.is_global_zero:
            if (epoch+1) % 1 == 0 or epoch==0:
                # We only need networks to produce generated images
                # No need to use bind and classifier here
                view_encoder.eval()
                action_encoder.eval()
                seek.eval()
                generator.eval()
                N = PLOT_N

                ## Train batch plot
                episodes_plot_imgs = episodes_plot_train[0][:N].to(fabric.device) # (N, V, C, H, W)
                episodes_plot_actions = episodes_plot_train[1][:N] # (N, V, A)
                _, V, C, H, W = episodes_plot_imgs.shape
                with torch.no_grad():
                    # View Encoder forward pass
                    flat_imgfttoks, flat_ret2D = view_encoder(episodes_plot_imgs.reshape(N * V, C, H, W))
                    noflat_imgfttoks = flat_imgfttoks.reshape(N, V, flat_imgfttoks.size(1), -1) # (N, V, Timg, Dimg)
                    noflat_ret2D = flat_ret2D.reshape(N, V, flat_ret2D.size(1), -1) # (N, V, Timg, 2)
                    # Action Encoder forward pass
                    flat_actions = [episodes_plot_actions[b][v] for b in range(N) for v in range(V)] # list length N*V
                    flat_acttok = action_encoder(flat_actions) # (N*V, 1, D)
                    noflat_acttok = flat_acttok.view(N, V, flat_acttok.size(1), -1) # (N, V, 1, D)
                    # Seek forward pass
                    if args.seek_gain_fields: crop_bv = extract_crop_params_from_raw(episodes_plot_actions, device=noflat_imgfttoks.device) # (N,V,4)
                    else: crop_bv = None
                    noflat_PRED_imgfttoks, mask_indices = seek(noflat_acttok, noflat_imgfttoks, noflat_ret2D, crop_bv) # (N, V, Timg, Dimg), (Timg)
                    # Generator forward pass
                    noflat_PRED_imgs = generator(noflat_PRED_imgfttoks) # (N, V, C, H, W)
                episodes_plot_gen_imgs = noflat_PRED_imgs.detach().cpu() # (N, V, C, H, W)
                episodes_plot_imgs = episodes_plot_imgs.detach().cpu() # (N, V, C, H, W)
                # plot each episode
                for i in range(N):
                    episode_i_imgs = episodes_plot_imgs[i]
                    episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
                    episode_i_imgs = torch.stack(episode_i_imgs, dim=0) # (V, C, H, W)
                    episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
                    episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0) # (V, C, H, W)
                    episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]
                    if args.channels == 4:
                        episode_i_gen_imgs_4th_channel = episode_i_gen_imgs[:, 3:, :, :] # (V, 1, H, W)
                        # repeat the 4th channel to have also a 3 channel image so we can plot it together with the other images
                        episode_i_gen_imgs_4th_channel = episode_i_gen_imgs_4th_channel.repeat(1, 3, 1, 1) # (V, 3, H, W)
                        episode_i_imgs = episode_i_imgs[:, :3, :, :] # (V, 3, H, W)
                        episode_i_gen_imgs = episode_i_gen_imgs[:, :3, :, :] # (V, 3, H, W) 
                        grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs, episode_i_gen_imgs_4th_channel], dim=0), nrow=args.num_views)
                    else:
                        grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
                    grid = grid.permute(1, 2, 0).cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grid = Image.fromarray(grid)
                    image_name = f'epoch{epoch}_episode{i}.png'
                    save_plot_dir = os.path.join(args.save_dir, 'gen_plots_train')
                    # create folder if it doesn't exist
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    grid.save(os.path.join(save_plot_dir, image_name))

                ## Validation batch plot
                episodes_plot_imgs = episodes_plot_val[0][:N].to(fabric.device) # (N, V, C, H, W)
                episodes_plot_actions = episodes_plot_val[1][:N] # (N, V, A)
                _, V, C, H, W = episodes_plot_imgs.shape
                with torch.no_grad():
                    # View Encoder forward pass
                    flat_imgfttoks, flat_ret2D = view_encoder(episodes_plot_imgs.reshape(N * V, C, H, W))
                    noflat_imgfttoks = flat_imgfttoks.reshape(N, V, flat_imgfttoks.size(1), -1) # (N, V, Timg, Dimg)
                    noflat_ret2D = flat_ret2D.reshape(N, V, flat_ret2D.size(1), -1) # (N, V, Timg, 2)
                    # Action Encoder forward pass
                    flat_actions = [episodes_plot_actions[b][v] for b in range(N) for v in range(V)] # list length N*V
                    flat_acttok = action_encoder(flat_actions) # (N*V, 1, D)
                    noflat_acttok = flat_acttok.view(N, V, flat_acttok.size(1), -1) # (N, V, 1, D)
                    # Seek forward pass
                    if args.seek_gain_fields: crop_bv = extract_crop_params_from_raw(episodes_plot_actions, device=noflat_imgfttoks.device) # (N,V,4)
                    else: crop_bv = None
                    noflat_PRED_imgfttoks, mask_indices = seek(noflat_acttok, noflat_imgfttoks, noflat_ret2D, crop_bv) # (N, V, Timg, Dimg), (Timg)
                    # Generator forward pass
                    noflat_PRED_imgs = generator(noflat_PRED_imgfttoks) # (N, V, C, H, W)
                episodes_plot_gen_imgs = noflat_PRED_imgs.detach().cpu() # (N, V, C, H, W)
                episodes_plot_imgs = episodes_plot_imgs.detach().cpu() # (N, V, C, H, W)
                # plot each episode
                for i in range(N):
                    episode_i_imgs = episodes_plot_imgs[i]
                    episode_i_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_imgs]
                    episode_i_imgs = torch.stack(episode_i_imgs, dim=0) # (V, C, H, W)
                    episode_i_gen_imgs = episodes_plot_gen_imgs[i]
                    episode_i_gen_imgs = [torchvision.transforms.functional.normalize(img, [-m/s for m, s in zip(args.mean, args.std)], [1/s for s in args.std]) for img in episode_i_gen_imgs]
                    episode_i_gen_imgs = torch.stack(episode_i_gen_imgs, dim=0) # (V, C, H, W)
                    episode_i_gen_imgs = torch.clamp(episode_i_gen_imgs, 0, 1) # Clip values to [0, 1]
                    if args.channels == 4:
                        episode_i_gen_imgs_4th_channel = episode_i_gen_imgs[:, 3:, :, :] # (V, 1, H, W)
                        # repeat the 4th channel to have also a 3 channel image so we can plot it together with the other images
                        episode_i_gen_imgs_4th_channel = episode_i_gen_imgs_4th_channel.repeat(1, 3, 1, 1) # (V, 3, H, W)
                        episode_i_imgs = episode_i_imgs[:, :3, :, :] # (V, 3, H, W)
                        episode_i_gen_imgs = episode_i_gen_imgs[:, :3, :, :] # (V, 3, H, W)
                        grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs, episode_i_gen_imgs_4th_channel], dim=0), nrow=args.num_views)
                    else:
                        grid = torchvision.utils.make_grid(torch.cat([episode_i_imgs, episode_i_gen_imgs], dim=0), nrow=args.num_views)
                    grid = grid.permute(1, 2, 0).cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grid = Image.fromarray(grid)
                    image_name = f'epoch{epoch}_episode{i}.png'
                    save_plot_dir = os.path.join(args.save_dir, 'gen_plots_val')
                    # create folder if it doesn't exist
                    if not os.path.exists(save_plot_dir):
                        os.makedirs(save_plot_dir)
                    grid.save(os.path.join(save_plot_dir, image_name))

        epoch_time = time.time() - start_time
        elapsed_time = time.time() - init_time
        fabric.print(f"Epoch [{epoch}] Epoch Time: {time_duration_print(epoch_time)} -- Elapsed Time: {time_duration_print(elapsed_time)}")

        # if epoch==30: # Break at epoch = 50 to save time for debugging
        #     break

    return None

if __name__ == '__main__':
    main()