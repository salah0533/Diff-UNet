import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml 
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder

set_determinism(123)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
data_dir = "./datasets/brats2020/MICCAI_BraTS2020_TrainingData/"

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 2
device = "cuda:0"
thres = 0.7

def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

def parse_config(config_path):

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, 7, 3, [64, 64, 128, 256, 512, 64])
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denose":
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 6
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, 3, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, 3, 96, 96, 96))
            step = torch.arange(1, 11)
            step = torch.softmax(torch.log(step), dim=0)

            for index in range(10):
# 
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))
              
                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

            return sample_return

        elif pred_type == "seg":
            return self.seg_unet(image)[0]
        elif pred_type == "embedding":
            return self.seg_unet(image)

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.5)
        
        self.model = DiffUNet()

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
       
        label = label.float()
        return image, label 

    def convert_label(self, output):
        b, d, w, h = output.shape[0], output.shape[2], output.shape[3], output.shape[4]
        show_output = np.zeros((b, d, w, h))
        wt = output[:, 1] == 1
        tc = output[:, 0] == 1
        et = output[:, 2] == 1
        show_output[wt] = 1
        show_output[tc] = 2
        show_output[et] = 3
        return show_output

    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        o = output[:, 1]
        t = target[:, 1] # ce
        wt = dice(o, t)
        wt_hd = hausdorff_distance_95(o, t)
        wt_recall = recall(o, t)
        wt_f1 = 1 if wt > thres else 0
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        tc_hd = hausdorff_distance_95(o, t)
        tc_recall = recall(o, t)
        tc_f1 = 1 if tc > thres else 0

        # active
        o = output[:, 2]
        t = target[:, 2]
       
        et = dice(o, t)
        et_hd = hausdorff_distance_95(o, t)
        et_recall = recall(o, t)
        et_f1 = 1 if et > thres else 0

        print(f"wt is {wt}, tc is {tc}, et is {et}")
        return [wt, tc, et, wt_hd, tc_hd, et_hd, wt_recall, tc_recall, et_recall, wt_f1, tc_f1, et_f1]

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__)

    logdir = "./logs_brats/diffusion_seg_all_loss_embed/model/final_model_0.8696.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")