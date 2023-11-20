import os
import pickle
from typing import Dict, Union
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
# Self-written modules
from gesture.DA.timegan.models.dataset import TimeGANDataset

def embedding_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the embedding and recovery functions
    """  
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
    emb_losses_recon = []
    emb_losses_sup = []
    for epoch in logger:
        for iter_index, (X_mb, T_mb) in enumerate(dataloader):
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            _, E_loss0, E_loss_T0,G_loss_S = model(X=X_mb, Z=None, obj="autoencoder")
            E_loss0.backward()
            loss = np.sqrt(E_loss_T0.item())
            emb_losses_recon.append(E_loss_T0.item()) # no sqrt
            emb_losses_sup.append(G_loss_S.item())

            e_opt.step()
            r_opt.step()

            writer.add_scalar("Train_emb/E_loss0", E_loss0, epoch * len(dataloader) + iter_index)
            writer.flush()
        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")

    args.losses['emb_losses_recon'] = emb_losses_recon
    args.losses['emb_losses_sup'] = emb_losses_sup

def supervisor_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None
) -> None:
    """The training loop for the supervisor function
    """
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
    sup_losses_sup=[]
    for epoch in logger:
        for iter_index, (X_mb, T_mb) in enumerate(dataloader):
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, Z=None, obj="supervisor")
            S_loss.backward()
            s_opt.step()

            loss = np.sqrt(S_loss.item())
            sup_losses_sup.append(S_loss.item())
            writer.add_scalar("Train_sup/S_loss", S_loss, epoch * len(dataloader) + iter_index)
            writer.flush()

        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")

    args.losses['sup_losses_sup'] = sup_losses_sup

def joint_trainer(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    e_opt: torch.optim.Optimizer, 
    r_opt: torch.optim.Optimizer, 
    s_opt: torch.optim.Optimizer, 
    g_opt: torch.optim.Optimizer, 
    d_opt: torch.optim.Optimizer, 
    args: Dict, 
    writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)]=None, 
) -> None:
    """The training loop for training the model altogether
    """
    #logger = trange(args.joint_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")
    joint_losses_gen_sum=[]
    joint_losses_gen_gen = []
    joint_losses_dis=[]
    joint_losses_emb=[]
    joint_losses_sup=[]
    D_loss=0
    G_loss_sum=0
    for epoch in range(args.joint_epochs):
        for iter_index,(X_mb, T_mb) in enumerate(dataloader):
            if iter_index % 1 == 0:
                ## Discriminator Training
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
                model.zero_grad()
                D_loss = model(X=X_mb, Z=Z_mb, obj="discriminator")
                if D_loss > args.dis_thresh:
                    D_loss.backward()
                    d_opt.step()
                D_loss = D_loss.item()
                joint_losses_dis.append(D_loss)
                writer.add_scalar("Train_joint/D_loss", D_loss, epoch * len(dataloader) + iter_index)
                writer.flush()

            ## Generator Training
            if epoch > 0 and iter_index % 1 == 0:
                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim)) # torch.Size([32, 500, 190])

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss_sum, G_loss_gen,G_loss_sup = model(X=X_mb, Z=Z_mb, obj="generator")
                G_loss_sum.backward()
                G_loss = np.sqrt(G_loss_sum.item())
                joint_losses_gen_sum.append(G_loss_sum.item())
                joint_losses_gen_gen.append(G_loss_gen.item())
                g_opt.step()
                #s_opt.step()
                writer.add_scalar("Train_joint/G_loss_sum", G_loss_sum, epoch*len(dataloader)+iter_index)
                writer.flush()

                '''
                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0,G_loss_S = model(X=X_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                #E_loss = np.sqrt(E_loss.item())
                joint_losses_emb.append(E_loss.item())
                joint_losses_sup.append(G_loss_S.item())
                
                # Update model parameters
                e_opt.step()
                r_opt.step()
                '''
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
                epoch, args.joint_epochs, iter_index, len(dataloader), D_loss, G_loss_sum))

    args.losses['joint_losses_gen_sum']= joint_losses_gen_sum
    args.losses['joint_losses_gen_gen'] = joint_losses_gen_gen
    args.losses['joint_losses_emb'] = joint_losses_emb
    args.losses['joint_losses_dis'] = joint_losses_dis


def timegan_trainer(model, dataloader, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    model.to(args.device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"{args.tensorboard_path}"))

    print("\nStart Embedding Network Training")
    embedding_trainer(
        model=model, 
        dataloader=dataloader, 
        e_opt=e_opt, 
        r_opt=r_opt, 
        args=args, 
        writer=writer
    )

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        args=args,
        writer=writer
    )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        args=args,
        writer=writer,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}")

def timegan_generator(model, args):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        args = torch.load(fb)
    model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
    
    print("\nGenerating Data...")
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((args.gen_trials, args.max_seq_len, args.Z_dim)) # torch.Size([32, 500, 190])
        generated_data = model(X=None, Z=Z, obj="inference")

    return generated_data.numpy() # (batch_size, wind, chn_number)
