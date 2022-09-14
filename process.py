
import os
import time

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax

class Hybrid_cnn():
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.pretrained_weights_path = '/opt/algorithm/checkpoints/hybrid_cnn/'
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/result'
        self.nii_seg_file = 'TCIA_001.nii.gz'
        self.npz_seg_file = 'TCIA_001.npz'

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
        img = sitk.ReadImage(nii_input_path)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                # os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))    # in my nnUnet, 1 for pet, 0 for ct
                                os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz')) 
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                # os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
                                os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz')) 
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.result_path, self.nii_seg_file), os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict_ssl(self):
        """
        Your algorithm goes here
        """
        print("ssl segmentation starting!")

        # one channel image        
        img_pet = sitk.ReadImage(os.path.join(self.nii_path, 'TCIA_001_0000.nii.gz'))
        pet_volume = sitk.GetArrayFromImage(img_pet)

        img_ct = sitk.ReadImage(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz'))
        ct_volume = sitk.GetArrayFromImage(img_ct)

        ct_volume_normalized = (ct_volume - (-800)) / (400 - (-800))
        ct_volume_normalized[ct_volume_normalized > 1] = 1.
        ct_volume_normalized[ct_volume_normalized < 0] = 0.

        pet_volume_normalized = (pet_volume - 0) / (0.95 * 15 - 0)
        pet_volume_normalized[pet_volume_normalized > 1] = 1.
        pet_volume_normalized[pet_volume_normalized < 0] = 0.

        start = ct_volume_normalized.shape[1] // 2 - 224 // 2
        end = start + 224
        pet_cropped = pet_volume_normalized[:, start:end, start:end]
        ct_cropped = ct_volume_normalized[:, start:end, start:end]

        model = ResNet(Bottleneck, [3, 4, 6, 3])
        num_channels = model.layer4[2].bn3.weight.shape[0]
        seg_decoder = Unet_Decoder(n_channels=num_channels, n_classes=2)

        model_stage_2 = UNet(n_channels=5, n_classes=2, bilinear=False)

        result = torch.empty([3, pet_cropped.shape[0], 2, 224, 224])
        for fold in range(3):

            # open checkpoint file
            checkpoint = torch.load(os.path.join(self.pretrained_weights_path, 'fold_' + str(fold+1) + '_best_checkpoint.pth.tar'), map_location="cpu")
            msg_1 = model.load_state_dict(checkpoint['en_state_dict'], strict=False)
            msg_2 = seg_decoder.load_state_dict(checkpoint['de_state_dict'], strict=False)
            # print('Pretrained weights found at {} and loaded with msg: {} and {}'.format(os.path.join(self.pretrained_weights_path, sorted(os.listdir(self.pretrained_weights_path))[int(fold)]), msg_1, msg_2))

            checkpoint = torch.load(os.path.join(self.pretrained_weights_path, 'fold_' + str(fold+1) + '_best_checkpoint_2nd_stage.pth.tar'), map_location="cpu")
            msg_3 = model_stage_2.load_state_dict(checkpoint['state_dict'], strict=False)
            # print('Pretrained weights for model_stage_2 found at {} and loaded with msg: {}'.format(os.path.join(self.pretrained_weights_path, 'hybrid_cnn/fold_' + str(fold+1) + '_best_checkpoint_2nd_stage.pth.tar'), msg_3))

            model.cuda()
            seg_decoder.cuda()
            model_stage_2.cuda()
            model.eval()
            seg_decoder.eval()
            model_stage_2.eval()

            pred_volume = torch.empty([pet_cropped.shape[0], 2, 224, 224])

            for i in range(pet_cropped.shape[0]):
                ct_slice = ct_cropped[i, :, :].astype(np.float32)
                ct_slice = (ct_slice -  0.2617) /  0.3239
                ct_slice = ct_slice.reshape((1,) + ct_slice.shape)
                ct_slice = torch.from_numpy(ct_slice.astype(np.float32)).cuda(non_blocking=True)
                ct_slice = ct_slice.repeat(3, 1, 1)

                pet_slice = pet_cropped[i, :, :].astype(np.float32)
                pet_slice = (pet_slice -  0.0456) /  0.0855
                pet_slice = pet_slice.reshape((1,) + pet_slice.shape)
                pet_slice = torch.from_numpy(pet_slice.astype(np.float32)).cuda(non_blocking=True)
                pet_slice = pet_slice.repeat(3, 1, 1)
                # tensor shape NxCxHxW, 2:5 means 1 channel CT + 2 channel PET
                input_images = torch.cat((torch.unsqueeze(ct_slice, dim=0), torch.unsqueeze(pet_slice, dim=0)), dim=1)[:,2:5,:,:]

                with torch.no_grad():
                    encoder_4_layers_features = model(input_images)
                    output = seg_decoder(input_images, encoder_4_layers_features)
                    _, pred_result = torch.max(output, dim=1)
                    # pred_result = pred_result.cpu().data.numpy()
                    # pred_prob = torch.softmax(output, dim=1)

                    input_images = torch.cat((torch.unsqueeze(ct_slice, dim=0), torch.unsqueeze(pet_slice, dim=0), output, torch.unsqueeze(pred_result, dim=0)), dim=1)[:,[2,5,6,7,8],:,:]
                    pred_prob = model_stage_2(input_images)

                pred_volume[i] = pred_prob[0]
        
            # result[i] shape Nx2xHxW, the 2 is the probability of the 2 classes
            result[fold] = pred_volume
        
        final_pred = torch.softmax(torch.mean(result,dim=0), dim=1)
        pred_result = final_pred.cpu().data.numpy()        
        pred_result[pet_cropped.shape[0] - 15:,:,:] = 0 # remove the last few slices that are in the brain region
        pred_result[:15,:,:] = 0 # remove the first few slices that are in the brain region
        pred_pad_volume=np.transpose(np.pad(pred_result, ((0,0), (0,0), (start, ct_volume_normalized.shape[1] - end), (start, ct_volume_normalized.shape[1] - end)), 'constant'), (1, 0, 2, 3))
        
        # combine with nnUnet outcome
        os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 001 -m 3d_fullres --save_npz')
        if not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('waiting for ssl segmentation to be created')
        while not os.path.exists(os.path.join(self.result_path, self.nii_seg_file)):
            print('.', end='')
            time.sleep(5)
        print('Prediction finished')

        pred_nnunet = np.load(os.path.join(self.result_path, self.npz_seg_file))['softmax']
        pred_sum = softmax(pred_pad_volume * 0.65 + pred_nnunet * 0.35, axis=0)
        pred_sum_result = np.argmax(pred_sum, axis=0).astype(np.uint8)
        # pred_sum_result = np.argmax(pred_pad_volume, axis=0).astype(np.uint8)

        pred_save_image = sitk.GetImageFromArray(pred_sum_result)
        pred_save_image.SetSpacing(img_pet.GetSpacing())
        pred_save_image.SetOrigin(img_pet.GetOrigin())
        pred_save_image.SetDirection(img_pet.GetDirection())
        sitk.WriteImage(pred_save_image, os.path.join(self.result_path, self.nii_seg_file))

        print("ssl segmentation done!")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()

        print('Start prediction')
        self.predict_ssl()      # get ssl output

        print('Start output writing')
        self.write_outputs(uuid)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # we return the output tokens from the encoder blocks
        return [x0, x2, x3, x4, x5]


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self._init_weight()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x_prob = self.softmax(x)
        return x_prob
 
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Unet_Decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(n_channels, (n_channels // 2) // factor, bilinear)
        self.up2 = Up(n_channels // 2, (n_channels // 4) // factor, bilinear)
        self.up3 = Up(n_channels // 4, (n_channels // 8) // factor, bilinear)
        self.up4_1 = nn.ConvTranspose2d(n_channels // 8, n_channels // 16, kernel_size=2, stride=2)
        self.up4_2 = nn.Sequential(
            nn.Conv2d(n_channels // 16 + 64, n_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 16),
            nn.ReLU(inplace=True)
        )
        self.up5_1 = nn.ConvTranspose2d(n_channels // 16, n_channels // 32, kernel_size=2, stride=2)
        self.up5_2 = nn.Sequential(
            nn.Conv2d(n_channels // 32 + 3, n_channels // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 32, n_channels // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 32),
            nn.ReLU(inplace=True)
        )
            
        self.outc = OutConv(n_channels // 32, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, encoder_feas):
        x = self.up1(encoder_feas[4], encoder_feas[3])
        x = self.up2(x, encoder_feas[2])
        x = self.up3(x, encoder_feas[1])
        x = self.up4_1(x)
        x = torch.cat([encoder_feas[0], x], dim=1)
        x = self.up4_2(x)
        x = self.up5_1(x)
        x = torch.cat([input, x], dim=1)
        x = self.up5_2(x)
        x = self.outc(x)
        x_prob = self.softmax(x)
        return x_prob


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        ###### Usually the bias is removed in conv layers before a batch norm layer, as the batch norm’s beta parameter (bias of nn.BatchNorm) will have the same effect 
        ###### and the bias of the conv layer might be canceled out by the mean subtraction.
        ###### https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067/2
        ###### http://proceedings.mlr.press/v37/ioffe15.html 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



if __name__ == "__main__":
    print("START")
    Hybrid_cnn().process()
