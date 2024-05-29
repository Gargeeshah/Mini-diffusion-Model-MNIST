import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import os
import math
from model import MNISTDiffusion
from ema import ExponentialMovingAverage

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):

    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)


"""
CONFIG SECTION - CHANGE AS PER REQUIREMENT
"""
lr = 0.001
batch_size=128
epochs=100
ckpt=''
n_samples=36
model_base_dim=64
timesteps=1000
model_ema_steps=10
model_ema_decay=0.995
log_freq=10
no_clip=True
cpu=False
"""
END OF CONFIG SECTION
"""
device="cpu" if cpu else "cuda"
train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=batch_size,image_size=28)
model=MNISTDiffusion(timesteps=timesteps,
            image_size=28,
            in_channels=1,
            base_dim=model_base_dim,
            dim_mults=[2,4]).to(device)

adjust = 1* batch_size * model_ema_steps / epochs
alpha = 1.0 - model_ema_decay
alpha = min(1.0, alpha * adjust)
model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

optimizer=AdamW(model.parameters(),lr=lr)
scheduler=OneCycleLR(optimizer,lr,total_steps=epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
loss_fn=nn.MSELoss(reduction='mean')

if ckpt:
    ckpt=torch.load(ckpt)
    model_ema.load_state_dict(ckpt["model_ema"])
    model.load_state_dict(ckpt["model"])

global_steps=0
for i in range(epochs):
    model.train()
    for j,(image,target) in enumerate(train_dataloader):
        noise=torch.randn_like(image).to(device)
        image=image.to(device)
        pred=model(image,noise)
        loss=loss_fn(pred,noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if global_steps%model_ema_steps==0:
            model_ema.update_parameters(model)
        global_steps+=1
        if j%log_freq==0:
            print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,epochs,j,len(train_dataloader),
                                                                loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
    ckpt={"model":model.state_dict(),
            "model_ema":model_ema.state_dict()}

    os.makedirs("models",exist_ok=True)
    torch.save(ckpt,"models/steps_{:0>8}.pt".format(global_steps))

    model_ema.eval()
    samples=model_ema.module.sampling(n_samples,clipped_reverse_diffusion=not no_clip,device=device)
    save_image(samples,"models/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(n_samples)))