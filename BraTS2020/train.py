import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os
import json
logdir = "/kaggle/working/logs_brats/diffusion_seg_all_loss_embed/"
model_save_path = os.path.join(logdir, "model")
'----------------------------------------------------'
data_dir = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
kaggle_dir = '/kaggle/working/logs_brats/diffusion_seg_all_loss_embed/model'


#env = "DDP" # or env = "pytorch" if you only have one gpu.
env = "pytorch" # or env = "pytorch" if you only have one gpu.
max_epoch = 1
batch_size = 1
val_every =1
num_gpus = 1
device = "cuda:0"

number_modality = 4
number_targets = 3 ## WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):

        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.25)

        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=30,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
       
        label = label.float()
        return image, label 

    def validation_step(self, batch,idx,epoch):
        best_model_path = "/kaggle/input/diff-unet-wieghts/best_model_0.8124.pt"
        trainer.load_state_dict(best_model_path)
        #print('weights loaded secsesfully')
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()

        try:
            os.mkdir('/kaggle/working/seg')
        except:
            pass
        out_sv_dir = f"/kaggle/working/seg/{all_ids[idx]}_out.npz"
        label_sv_dir = f"/kaggle/working/seg/{all_ids[idx]}_seg.npz"
        np.savez_compressed(out_sv_dir,output)
        np.savez_compressed(label_sv_dir,label)
        #print('-------------------- saved ----------------')


        target = label.cpu().numpy()
        o = output[:, 1]
        t = target[:, 1] # ce
        wt = dice(o, t)
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        # active
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)
        
        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", (wt+tc+et)/3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")

if __name__ == "__main__":

    train_and_val_directories = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    def pathListIntoIds(dirList):
        x = []
        for i in range(0,len(dirList)):
            x.append(dirList[i][dirList[i].rfind('/')+1:])
        return x

    train_and_test_ids = pathListIntoIds(train_and_val_directories)
    train_test_split_brats20 = {}

    train_test_split_brats20['test'] = ["BraTS20_Training_254", "BraTS20_Training_068", "BraTS20_Training_137", "BraTS20_Training_106", "BraTS20_Training_146", "BraTS20_Training_102", "BraTS20_Training_034", "BraTS20_Training_242", "BraTS20_Training_114", "BraTS20_Training_105", "BraTS20_Training_006", "BraTS20_Training_210", "BraTS20_Training_149", "BraTS20_Training_074", "BraTS20_Training_115", "BraTS20_Training_185", "BraTS20_Training_053", "BraTS20_Training_246", "BraTS20_Training_338", "BraTS20_Training_099", "BraTS20_Training_316", "BraTS20_Training_131", "BraTS20_Training_162", "BraTS20_Training_177", "BraTS20_Training_104", "BraTS20_Training_113", "BraTS20_Training_268", "BraTS20_Training_266", "BraTS20_Training_049", "BraTS20_Training_056", "BraTS20_Training_223", "BraTS20_Training_226", "BraTS20_Training_132", "BraTS20_Training_175", "BraTS20_Training_166", "BraTS20_Training_080", "BraTS20_Training_042", "BraTS20_Training_201", "BraTS20_Training_298", "BraTS20_Training_328", "BraTS20_Training_163", "BraTS20_Training_291", "BraTS20_Training_111", "BraTS20_Training_069", "BraTS20_Training_215", "BraTS20_Training_033", "BraTS20_Training_311", "BraTS20_Training_151", "BraTS20_Training_019", "BraTS20_Training_247", "BraTS20_Training_248", "BraTS20_Training_229", "BraTS20_Training_305", "BraTS20_Training_351", "BraTS20_Training_030", "BraTS20_Training_008", "BraTS20_Training_356", "BraTS20_Training_205", "BraTS20_Training_170", "BraTS20_Training_269", "BraTS20_Training_341", "BraTS20_Training_292", "BraTS20_Training_138", "BraTS20_Training_260", "BraTS20_Training_057", "BraTS20_Training_263", "BraTS20_Training_366", "BraTS20_Training_212", "BraTS20_Training_174", "BraTS20_Training_090", "BraTS20_Training_359", "BraTS20_Training_118", "BraTS20_Training_219", "BraTS20_Training_076"]
    train_test_split_brats20['train'] = []
    for id in np.sort(train_and_test_ids):
        if id not in train_test_split_brats20['test']:
            train_test_split_brats20['train'].append(id)
    all_ids = train_test_split_brats20['train'] + train_test_split_brats20['test']
    "--------------- setup dataset dir to be upladed later ---------------------"

    os.makedirs(kaggle_dir, exist_ok=True)
    API = {"username":"salahpsg","key":"b4e0ab6c595cd6615cf39b847adff51c"}
    os.environ['KAGGLE_USERNAME'] = API["username"]
    os.environ['KAGGLE_KEY'] = API["key"]

    dataset_name = 'Diff-UNet-wieghts'


    with open(os.path.join(kaggle_dir , 'dataset-metadata.json'), 'w') as f:
            json.dump({
                  "title": dataset_name,
                  "id": os.environ['KAGGLE_USERNAME']+"/"+dataset_name,
                  "licenses": [
                    {
                      "name": "CC0-1.0"
                    }
                  ]
                },
              f)
    '---------------------------------------------------------------------------'
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)
    try:
        best_model_path = "/kaggle/input/diff-unet-wieghts/best_model_0.8124.pt"
        trainer.load_state_dict(best_model_path)
        print('weights loaded secsesfully')
    except:
        print("couldn't loaded weights")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
