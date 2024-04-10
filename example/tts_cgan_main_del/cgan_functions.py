# Conditional GAN training  

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from gesture.DA.tts_cgan_main.utils import save_image
from tqdm import tqdm

logger = logging.getLogger(__name__)

def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        # if search_iter < self.grow_step1:
        #     return 0
        # elif self.grow_step1 <= search_iter < self.grow_step2:
        #     return 1
        # else:
        #     return 2
        # for idx, grow_step in enumerate(args.grow_steps):
        #     if iter < grow_step:
        #         return idx
        # return len(args.grow_steps)
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx

# y: torch.Size([32, 1]); x: torch.Size([32, 1, 1, 187])
def gradient_penalty(y, x, args):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda(args.gpu, non_blocking=True)
    dydx = torch.autograd.grad(outputs=y, # torch.Size([32, 1, 1, 187])
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.reshape(dydx.size(0), -1) #dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1)) # torch.Size([32])
    return torch.mean((dydx_l2norm-1)**2)    
    
def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train_del(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    cls_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.BCEWithLogitsLoss()
    lambda_cls = 1
    lambda_gp = 10
    gen_net.train()
    dis_net.train()

    r_correct_cls_sum = 0
    f_correct_cls_sum = 0
    r_correct_adv_sum = 0
    f_correct_adv_sum = 0

    for iter_idx, (real_imgs, r_label_cls) in enumerate(tqdm(train_loader)):
        r_correct_cls = 0
        r_correct_adv = 0
        f_correct_cls = 0
        f_correct_adv = 0
        r_label_adv = torch.ones((r_label_cls.shape)).to('cuda')
        f_label_adv = torch.zeros((r_label_cls.shape)).to('cuda')
        r_label_cls = r_label_cls.type(torch.LongTensor).to('cuda')  # torch.Size([64])
        f_label_cls = torch.randint(0, 5, (real_imgs.shape[0],)).to('cuda')  # torch.Size([64])

        real_imgs = real_imgs.type(torch.cuda.FloatTensor).to('cuda')  # torch.Size([64, 1, 1, 187])
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).to(
            'cuda')  # torch.Size([64, 128])
        fake_imgs = gen_net(noise, f_label_cls)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_net.zero_grad()
        # forward pass
        r_out_adv, r_out_cls = dis_net(real_imgs)
        f_out_adv, f_out_cls = dis_net(fake_imgs)  # torch.Size([64, 1, 1, 187])
        # real classification loss
        r_cls_loss = cls_criterion(r_out_cls, r_label_cls)
        f_cls_loss = cls_criterion(f_out_cls, f_label_cls)

        if args.loss_metric == 'wgangp':
            # Compute loss for gradient penalty.
            alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to('cuda')  # bh, C, H, W
            x_hat = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
            out_src, _ = dis_net(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat,
                                         args)  # tensor(0.9068, device='cuda:0', grad_fn=<MeanBackward0>)

            r_adv_loss = -torch.mean(r_out_adv)  # tensor(0.5191, device='cuda:0', grad_fn=<NegBackward>)
            f_adv_loss = torch.mean(f_out_adv)  # tensor(0.0812, device='cuda:0', grad_fn=<MeanBackward0>)
            adv_loss = r_adv_loss + f_adv_loss  # tensor(0.6003, device='cuda:0', grad_fn=<AddBackward0>)

            d_loss = adv_loss + lambda_cls * r_cls_loss + lambda_gp * d_loss_gp
        elif args.loss_metric == 'standard':
            r_adv_loss = adv_criterion(r_out_adv.squeeze(), r_label_adv.squeeze())
            f_adv_loss = adv_criterion(f_out_adv.squeeze(), f_label_adv.squeeze())
            adv_loss = r_adv_loss + f_adv_loss
            # d_loss = adv_loss + lambda_cls * r_cls_loss
            d_loss = adv_loss + lambda_cls * (r_cls_loss + f_cls_loss)

        # train D&G equally for 15 epochs, then train G more
        if (epoch < 5) or (epoch >= 5 and (iter_idx % 11 == 0)):  # and writer_dict['f_acc_adv'] < 0.9
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), writer_dict['train_global_steps'])

        # classification error
        r_preds_cls = r_out_cls.argmax(dim=1, keepdim=True)
        f_preds_cls = f_out_cls.argmax(dim=1, keepdim=True)
        r_correct_cls = torch.sum(r_preds_cls.squeeze() == r_label_cls.squeeze())
        f_correct_cls = torch.sum(f_preds_cls.squeeze() == f_label_cls.squeeze())
        r_correct_cls_sum = r_correct_cls_sum + r_correct_cls
        f_correct_cls_sum = f_correct_cls_sum + f_correct_cls
        r_acc_cls = r_correct_cls / args.batch_size
        f_acc_cls = f_correct_cls / args.batch_size

        # adv error
        if args.loss_metric == 'standard':
            r_preds_adv = torch.sigmoid(r_out_adv) > 0.5  # r_out_adv.argmax(dim=1, keepdim=True)
            f_preds_adv = torch.sigmoid(f_out_adv) < 0.5  # f_out_adv.argmax(dim=1, keepdim=True)
            r_correct_adv = sum(
                torch.sigmoid(r_out_adv) > 0.5)  # sum(torch.sum(r_preds_adv.squeeze() == r_label_adv.squeeze())
            f_correct_adv = sum(
                torch.sigmoid(f_out_adv) < 0.5)  # torch.sum(f_preds_adv.squeeze() == f_label_adv.squeeze())
            r_acc_adv = r_correct_adv / args.batch_size
            f_acc_adv = f_correct_adv / args.batch_size
            r_correct_adv_sum = r_correct_adv_sum + r_correct_adv
            f_correct_adv_sum = f_correct_adv_sum + f_correct_adv
            writer_dict['f_acc_adv'] = f_acc_adv  # pause the D training if it > 0.9
            writer.add_scalars('monitor', {'r_acc_adv': r_acc_adv,  # will create a new folder: monitor
                                           'f_acc_adv': f_acc_adv,
                                           'r_acc_cls': r_acc_cls,
                                           'f_acc_cls': f_acc_cls},
                               writer_dict['train_global_steps'])

        # -----------------
        #  Train Generator
        # -----------------

        gen_net.zero_grad()

        # data should be around zero
        gen_imgs = gen_net(noise, f_label_cls)
        g_out_adv, g_out_cls = dis_net(gen_imgs)
        mean_value = torch.mean(gen_imgs, 3, True)  # should be around zero
        amp = torch.sum(mean_value.abs())

        if args.loss_metric == 'wgangp':
            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = cls_criterion(g_out_cls.squeeze(), f_label_cls.squeeze())  # why try to class fake data?
            g_loss = g_adv_loss + lambda_cls * g_cls_loss
        elif args.loss_metric == 'standard':
            g_loss = adv_criterion(g_out_adv.squeeze(), r_label_adv.squeeze())
        if amp > 2:
            g_loss = g_loss + amp
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        # torch.nn.utils.clip_grad_norm_(gen_net.convlayer.parameters(), 1.)
        gen_optimizer.step()

        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(writer_dict['train_global_steps'])
            d_lr = dis_scheduler.step(writer_dict['train_global_steps'])
            writer.add_scalar('LR/g_lr', g_lr, writer_dict['train_global_steps'])
            writer.add_scalar('LR/d_lr', d_lr, writer_dict['train_global_steps'])

        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.dis_batch_size * args.world_size * writer_dict['train_global_steps']
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
            del cpu_p

        writer.add_scalar('g_loss', g_loss.item(), writer_dict['train_global_steps'])
        writer_dict['train_global_steps'] += 1

        # verbose
        if writer_dict['train_global_steps'] and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(),
                 ema_beta))

        if args.debug == True:
            break
    r_accuracy_cls = r_correct_cls_sum / (len(train_loader) * args.batch_size)
    f_accuracy_cls = f_correct_cls_sum / (len(train_loader) * args.batch_size)
    r_accuracy_adv = r_correct_adv_sum / (len(train_loader) * args.batch_size)
    f_accuracy_adv = f_correct_adv_sum / (len(train_loader) * args.batch_size)

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    cls_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.BCEWithLogitsLoss()
    lambda_cls = 1
    lambda_gp = 10
    gen_net.train()
    dis_net.train()

    r_correct_cls_sum=0
    f_correct_cls_sum=0
    r_correct_adv_sum=0
    f_correct_adv_sum=0

    for iter_idx, (real_imgs, r_label_cls) in enumerate(tqdm(train_loader)):
        r_correct_cls = 0
        r_correct_adv = 0
        f_correct_cls = 0
        f_correct_adv = 0
        r_label_adv=torch.ones((r_label_cls.shape)).to('cuda')
        f_label_adv=torch.zeros((r_label_cls.shape)).to('cuda')
        r_label_cls = r_label_cls.type(torch.LongTensor).to('cuda')  # torch.Size([64])

        real_imgs = real_imgs.type(torch.cuda.FloatTensor).to('cuda') # torch.Size([64, 1, 1, 187])


        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_net.zero_grad()
        # real forward
        r_out_adv, r_out_cls = dis_net(real_imgs)
        # fake forward
        f_label_cls = torch.randint(0, 5, (real_imgs.shape[0],)).to('cuda')  # torch.Size([64])
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).to('cuda')  # torch.Size([64, 128])
        fake_imgs = gen_net(noise, f_label_cls)
        f_out_adv, f_out_cls = dis_net(fake_imgs) # torch.Size([64, 1, 1, 187])
        # real classification loss
        r_cls_loss = cls_criterion(r_out_cls, r_label_cls)
        f_cls_loss = cls_criterion(f_out_cls, f_label_cls)

        if args.loss_metric=='wgangp':
            # Compute loss for gradient penalty.
            alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to('cuda')  # bh, C, H, W
            x_hat = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
            out_src, _ = dis_net(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, args) # tensor(0.9068, device='cuda:0', grad_fn=<MeanBackward0>)

            r_adv_loss = -torch.mean(r_out_adv) # tensor(0.5191, device='cuda:0', grad_fn=<NegBackward>)
            f_adv_loss = torch.mean(f_out_adv) #tensor(0.0812, device='cuda:0', grad_fn=<MeanBackward0>)
            adv_loss = r_adv_loss + f_adv_loss # tensor(0.6003, device='cuda:0', grad_fn=<AddBackward0>)

            d_loss = adv_loss + lambda_cls * r_cls_loss + lambda_gp * d_loss_gp

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()


        elif args.loss_metric == 'standard':
            r_adv_loss = adv_criterion(r_out_adv.squeeze(), r_label_adv.squeeze())
            f_adv_loss = adv_criterion(f_out_adv.squeeze(), f_label_adv.squeeze())
            adv_loss = r_adv_loss + f_adv_loss
            #d_loss = adv_loss + lambda_cls * r_cls_loss
            d_loss = adv_loss + lambda_cls * (r_cls_loss + f_cls_loss)

            # train D&G equally for 15 epochs, then train G more
            if (epoch < 5) or (epoch >= 5 and (iter_idx % 11 == 0)): #  and writer_dict['f_acc_adv'] < 0.9
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
                dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), writer_dict['train_global_steps'])

        # classification error
        r_preds_cls = r_out_cls.argmax(dim=1, keepdim=True)
        f_preds_cls = f_out_cls.argmax(dim=1, keepdim=True)
        r_correct_cls = torch.sum(r_preds_cls.squeeze() == r_label_cls.squeeze())
        f_correct_cls = torch.sum(f_preds_cls.squeeze() == f_label_cls.squeeze())
        r_correct_cls_sum = r_correct_cls_sum + r_correct_cls
        f_correct_cls_sum = f_correct_cls_sum + f_correct_cls
        r_acc_cls=r_correct_cls/args.batch_size
        f_acc_cls = f_correct_cls / args.batch_size

        # log the error
        if args.loss_metric == 'standard':
            r_preds_adv = torch.sigmoid(r_out_adv)>0.5 #r_out_adv.argmax(dim=1, keepdim=True)
            f_preds_adv = torch.sigmoid(f_out_adv)<0.5 #f_out_adv.argmax(dim=1, keepdim=True)
            r_correct_adv = sum(torch.sigmoid(r_out_adv)>0.5) # sum(torch.sum(r_preds_adv.squeeze() == r_label_adv.squeeze())
            f_correct_adv = sum(torch.sigmoid(f_out_adv)<0.5) # torch.sum(f_preds_adv.squeeze() == f_label_adv.squeeze())
            r_acc_adv = r_correct_adv / args.batch_size
            f_acc_adv = f_correct_adv / args.batch_size
            r_correct_adv_sum = r_correct_adv_sum + r_correct_adv
            f_correct_adv_sum = f_correct_adv_sum + f_correct_adv
            writer_dict['f_acc_adv']=f_acc_adv # pause the D training if it > 0.9
            writer.add_scalars('monitor', {'r_acc_adv': r_acc_adv, # will create a new folder: monitor
                                           'f_acc_adv': f_acc_adv,
                                           'r_acc_cls': r_acc_cls,
                                           'f_acc_cls': f_acc_cls},
                               writer_dict['train_global_steps'])
        else:
            pass # do not log for wgangp
        # -----------------
        #  Train Generator
        # -----------------
        
        gen_net.zero_grad()

        # data should be around zero
        gen_imgs = gen_net(noise, f_label_cls)
        g_out_adv, g_out_cls = dis_net(gen_imgs)
        mean_value=torch.mean(gen_imgs,3,True) # should be around zero
        amp=torch.sum(mean_value.abs())

        if args.loss_metric=='wgangp':
            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = cls_criterion(g_out_cls.squeeze(), f_label_cls.squeeze()) # why try to class fake data?
            g_loss = g_adv_loss + lambda_cls * g_cls_loss
        elif args.loss_metric=='standard':
            g_loss=adv_criterion(g_out_adv.squeeze(),r_label_adv.squeeze())
        if amp > 2:
            pass
            #g_loss=g_loss+amp
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        #torch.nn.utils.clip_grad_norm_(gen_net.convlayer.parameters(), 1.)
        gen_optimizer.step()

        # adjust learning rate
        schedulers=False # disable it
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(writer_dict['train_global_steps'])
            d_lr = dis_scheduler.step(writer_dict['train_global_steps'])
            writer.add_scalar('LR/g_lr', g_lr, writer_dict['train_global_steps'])
            writer.add_scalar('LR/d_lr', d_lr, writer_dict['train_global_steps'])

        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.dis_batch_size * args.world_size * writer_dict['train_global_steps']
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
            del cpu_p

        writer.add_scalar('g_loss', g_loss.item(), writer_dict['train_global_steps'])
        writer_dict['train_global_steps'] += 1

        # verbose
        if writer_dict['train_global_steps'] and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), ema_beta))

        if args.debug==True:
            break
    r_accuracy_cls = r_correct_cls_sum / (len(train_loader) * args.batch_size)
    f_accuracy_cls = f_correct_cls_sum / (len(train_loader) * args.batch_size)
    r_accuracy_adv = r_correct_adv_sum / (len(train_loader) * args.batch_size)
    f_accuracy_adv = f_correct_adv_sum / (len(train_loader) * args.batch_size)

def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('calculate Inception score...')
    mean, std = get_inception_score(img_list)

    return mean        
        
def save_samples(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        batch_size = fixed_z.size(0)
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i:(i+1)], epoch)
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(f"./samples/{args.exp_name}", exist_ok=True)
        save_image(sample_imgs, f'./samples/{args.exp_name}/sampled_images_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(args.num_candidate, with_hidden=True, prev_archs=prev_archs,
                                             prev_hiddens=prev_hiddens)
    hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten