from utils.data_utils import load_config
import torch
from models.base_module import BaseLitModule
from models.SpatioTemporalLSTMCell_v2 import PredRNN
from models.backbones.unet import MoE

class UNetWrapper(torch.nn.Module):
    def __init__(self, input_channels, output_channels, nb_filter=None):
        super().__init__()
        self.input_channels = input_channels
        from models.backbones.unet import UNet
        self.model = UNet(input_channels=input_channels, num_classes=output_channels, nb_filter=nb_filter)

    def forward(self, x):
        img_w = x.shape[-2]
        img_h = x.shape[-1]

        pw = (32 - img_w % 32) // 2
        ph = (32 - img_h % 32) // 2

        x = x.reshape(-1, self.input_channels, img_w, img_h)
        x = torch.nn.functional.pad(x, (pw, pw, ph, ph), mode="replicate")  # 252x252 -> 256x256
        x, f = self.model(x)
        x = x.unsqueeze(1)  # add back channel dim
        x = x[..., pw:-pw, ph:-ph]  # back to 252x252
        return x, f


def crop_slice(img_size=252, scale_ratio=2/12):
    padding = int(img_size * (1 - scale_ratio) // 2)
    return ..., slice(padding, img_size-padding), slice(padding, img_size-padding)


class PhyDNetWrapper(torch.nn.Module):
    def __init__(self, config_path, ckpt_path=None):
        super().__init__()
        phydnet_config = load_config(config_path)
        from models.backbones.phydnet import PhyDNet
        if ckpt_path:
            self.phydnet = PhyDNet.load_from_checkpoint(ckpt_path, config=phydnet_config)
        else:
            self.phydnet = PhyDNet(phydnet_config)

    def forward(self, x):
        return self.phydnet(x)



# ---------------------------------
# LIGHTNING MODULES
# ---------------------------------



class UNetCropUpscale(BaseLitModule):
    def __init__(self, config):
        super().__init__(config)

        self.unet = UNetWrapper(
            input_channels=11 * config["dataset"]["len_seq_in"],
            output_channels=config["dataset"]["len_seq_predict"],
        )
        self.upscale = torch.nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)

    def forward(self, x, crop=True, upscale=True):
        x = self.unet(x)

        if crop:
            x = x[crop_slice()]

        if upscale:
            x = self.upscale(x[:, 0]).unsqueeze(1)

        return x


class WeatherFusionNet_stage2(BaseLitModule):
    def __init__(self, UNet_params, config):
        super().__init__(UNet_params, config)
        self.phydnet = PhyDNetWrapper("models/configurations/phydnet.yaml", ckpt_path="/path/to/your/pretrained/weights/sat-phydnet.ckpt")
        for param in self.phydnet.parameters():
            param.requires_grad = False
        self.sat2rad = UNetWrapper(input_channels=11, output_channels=1)
        self.sat2rad.load_state_dict(torch.load("/path/to/your/pretrained/weights/sat2rad-unet.pt"))
        self.unet = UNetWrapper(input_channels=11 * (4 + 10) + 4, output_channels=32)
        self.upscale = torch.nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)
        self.predRNN = PredRNN(in_channels=1, num_hidden=1, width=252, filter_size=3)
        self.MoE = MoE(num_classes=32)

    def forward(self, x, return_inter=False):
        self.sat2rad.eval()
        with torch.no_grad():
            sat2rad_out = self.sat2rad(x.swapaxes(1, 2))[0].reshape(x.shape[0], 4, x.shape[-2], x.shape[-1])

        self.phydnet.eval()
        with torch.no_grad():
            phydnet_out = self.phydnet(x.swapaxes(1, 2)).flatten(1, 2)

        x = torch.concat([x.flatten(1, 2), phydnet_out, sat2rad_out], dim=1)
        self.unet.eval()
        with torch.no_grad():
            x, f = self.unet(x)

        x = 0
        Xs = self.MoE(f)
        for x_i in Xs:
            x_i = x_i[crop_slice()]
            x_i = self.upscale(x_i[:, 0]).unsqueeze(1)
            x_i = self.predRNN(x_i)
            x += x_i

        x /= len(Xs)

        if return_inter:
            return sat2rad_out, phydnet_out, x
        return x


    def configure_optimizers(self):
        params = [
            {'params': self.MoE.parameters()},
            {'params': self.predRNN.parameters(),'lr': 0.01*float(self.config["lr"])},]

        optimizer = torch.optim.AdamW(params, lr=float(self.config["lr"]),
                                      weight_decay=float(self.config["weight_decay"]))
        return optimizer
    

class WeatherFusionNet_stage3(BaseLitModule):
    def __init__(self, UNet_params, config):
        super().__init__(UNet_params, config)
        self.phydnet = PhyDNetWrapper("models/configurations/phydnet.yaml",
                                      ckpt_path="/path/to/your/pretrained/weights/sat-phydnet.ckpt")
        self.sat2rad = UNetWrapper(input_channels=11, output_channels=1)
        self.sat2rad.load_state_dict(
            torch.load("/path/to/your/pretrained/weights/sat2rad-unet.pt"))
        self.unet_ptr = UNetWrapper(input_channels=11 * (4 + 10) + 4, output_channels=32)
        self.unet_ptr.load_state_dict(torch.load("/path/to/your/pretrained/weights/unet.pt"))
        self.unet = UNetWrapper(input_channels=11 * (4 + 10) + 4, output_channels=32)
        self.upscale = torch.nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)
        self.predRNN = PredRNN(in_channels=1, num_hidden=1, width=252, filter_size=3)
        self.predRNN2 = PredRNN(in_channels=5, num_hidden=5, width=252, filter_size=3)

        self.sigmoid = torch.nn.Sigmoid()
        self.MoE = MoE(num_classes=32)

    def forward(self, x, return_inter=False):
        self.sat2rad.eval()
        with torch.no_grad():
            sat2rad_out = self.sat2rad(x.swapaxes(1, 2))[0].reshape(x.shape[0], 4, x.shape[-2], x.shape[-1])

        self.phydnet.eval()
        with torch.no_grad():
            phydnet_out = self.phydnet(x.swapaxes(1, 2)).flatten(1, 2)

        x = torch.concat([x.flatten(1, 2), phydnet_out, sat2rad_out], dim=1)

        self.unet_ptr.eval()
        with torch.no_grad():
            x_ptr, _ = self.unet_ptr(x)
            x_ptr = x_ptr[crop_slice()]
            x_ptr = self.upscale(x_ptr[:, 0]).unsqueeze(1)

        x_ptr_5 = repeat(x_ptr, 'b c t w h -> b (rp c) t w h', rp=5)
        x_ptr_5 = self.predRNN2(x_ptr_5)
        x_ptr_s_5 = self.sigmoid(x_ptr_5)

        self.unet.eval()
        with torch.no_grad():
            _, f = self.unet(x)

        x = 0.0
        self.MoE.eval()
        self.predRNN.eval()
        with torch.no_grad():
            Xs = self.MoE(f)
        for i, x_i in enumerate(Xs):
            x_tpr_s = x_ptr_s_5[:, i, :, :, :].unsqueeze(1)
            x_i = x_i[crop_slice()]
            x_i = self.upscale(x_i[:, 0]).unsqueeze(1)
            with torch.no_grad():
                x_i = self.predRNN(x_i)
            x += x_i * x_tpr_s

        if return_inter:
            return sat2rad_out, phydnet_out, x
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.predRNN2.parameters(), lr=float(self.config["lr"]),
                                      weight_decay=float(self.config["weight_decay"]))
        return optimizer
