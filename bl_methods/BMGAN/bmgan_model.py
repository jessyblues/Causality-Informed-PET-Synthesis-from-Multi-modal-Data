import monai.networks
import torch
import os
import monai
import torch.nn as nn
from monai.networks.blocks import ConvDenseBlock, ResidualUnit
import sys
sys.path.append('/home1/yujiali/diffusion')
from monai_diffusion.generative.networks.nets import PatchDiscriminator
import pdb

def get_dense_block(input_c, output_c):
    
    layers = [ConvDenseBlock(spatial_dims=3, in_channels=input_c,  
                            channels=[output_c], num_res_units=1,
                            act=("leakyrelu", {"negative_slope": 0.2})),
                nn.Conv3d(input_c+output_c, output_c, 3, padding=1), nn.InstanceNorm3d(output_c), nn.LeakyReLU(0.2),
                ConvDenseBlock(spatial_dims=3, in_channels=output_c,  
                            channels=[output_c], num_res_units=1,
                            act=("leakyrelu", {"negative_slope": 0.2})),
                nn.Conv3d(output_c+output_c, output_c, 3, padding=1), nn.InstanceNorm3d(output_c), nn.LeakyReLU(0.2)]
    
    return layers

class dense_unet_generator(nn.Module):
    
    def __init__(self, input_channel = 9, input_conv_channel=64, output_conv_channel=64,
                 down_layers = 5, down_channels=[128, 256, 256, 512],
                 middle_layers = 1, middle_channels = [512],
                 up_layers = 6, up_channels = [512, 256, 256, 256, 128]):
        super().__init__()
        
    
        self.input_layer = nn.Sequential(
            nn.Conv3d(input_channel, input_conv_channel, 3, padding=1), nn.InstanceNorm3d(input_conv_channel), nn.LeakyReLU(0.2), 
            nn.Conv3d(input_conv_channel, input_conv_channel, 3, padding=1), nn.InstanceNorm3d(input_conv_channel), nn.LeakyReLU(0.2),
            nn.Conv3d(input_conv_channel, input_conv_channel, 3, padding=1, stride=2), nn.InstanceNorm3d(input_conv_channel), nn.LeakyReLU(0.2)
        )
        
        self.down_layers = nn.ModuleList([])
        current_channel = input_conv_channel
        
        for i, c in enumerate(down_channels):
            
            self.down_layers.append(nn.Sequential(
                *(get_dense_block(current_channel, c)+[nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1),
                                    nn.InstanceNorm3d(c),
                                    nn.LeakyReLU(0.2)])
            ))
            
            current_channel = c

        self.middle_layers = nn.Sequential(*get_dense_block(current_channel, middle_channels[-1]))
        current_channel = middle_channels[-1]
        
        self.up_layers = nn.ModuleList([])
        for i, c in enumerate(up_channels):
            self.up_layers.append(
                nn.Sequential(*(get_dense_block(current_channel+([input_conv_channel]+down_channels)[-1-i], c)+
                  [nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm3d(c),
                    nn.LeakyReLU(0.2)])))
            
            current_channel = c
        
        self.output_layer = nn.Sequential(
            nn.Conv3d(current_channel, output_conv_channel, 3, padding=1), nn.InstanceNorm3d(output_conv_channel), nn.LeakyReLU(0.2), 
            nn.Conv3d(output_conv_channel, output_conv_channel, 3, padding=1), nn.InstanceNorm3d(output_conv_channel), nn.LeakyReLU(0.2),
            nn.Conv3d(output_conv_channel, 1, 3, padding=1), nn.Tanh()
        )
        
    
    def forward(self, x:torch.Tensor, sampled_latent_vector:torch.Tensor):
        
        batch_size = x.shape[0]
        sampled_latent_vector = sampled_latent_vector.view(batch_size, -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).\
                                                                            expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        
        concate_input = torch.cat([x, sampled_latent_vector], dim=1)
        
        feature = self.input_layer(concate_input)
        residual_features = [feature]
        for i, downsampled_block in enumerate(self.down_layers):
            #pdb.set_trace()
            feature = downsampled_block(feature)
            residual_features.append(feature)
            #pdb.set_trace()
            #print(feature.shape)
            
            
        feature = self.middle_layers(feature)
        
        for i, upsampled_block in enumerate(self.up_layers):
            #pdb.set_trace()
            feature = torch.cat([feature, residual_features[-1-i]], dim=1)
            feature = upsampled_block(feature)
            
            #print(feature.shape)
        output = self.output_layer(feature)
        
        return output
    
class ResNet_encoder(nn.Module):
    
    def __init__(self, input_layer_channel=32, channels=[64, 128, 128, 128, 128, 128]):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv3d(1, input_layer_channel, 3, padding=1), nn.InstanceNorm3d(input_layer_channel), nn.ReLU()
        )
        
        self.resblocks = nn.ModuleList([])
        current_channel = input_layer_channel
        for i, c in enumerate(channels):
            self.resblocks.append(ResidualUnit(3, current_channel, c, strides=2, padding=1))
            
            current_channel = c
    
        self.linear1 = nn.Linear(128*8, 8)
        self.linear2 = nn.Linear(128*8, 8)
        
    def forward(self, x):
        
        output = self.input_layer(x)
        
        for i, resblock in enumerate(self.resblocks):
            output = resblock(output)
        
        output = nn.Flatten()(output)
        return self.linear1(output), self.linear2(output)


class patch_discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
    
        self.patch_d = PatchDiscriminator(
            3, 32, 1, num_layers_d=4
        )
        
    def forward(self, x):
        
        return self.patch_d(x)[-1]
    

if __name__ == '__main__':
    
    t1 = torch.rand([1, 1, 96, 128, 96], device=torch.device('cuda:7'))
    vector_ = torch.randn([1, 8], device=torch.device('cuda:7'))
    model = dense_unet_generator().to(device=torch.device('cuda:7'))
    encoder = ResNet_encoder().to(device=torch.device('cuda:7'))
    d = patch_discriminator().to(device=torch.device('cuda:7'))
    
    result = model(t1, vector_)
    result1 = d(t1)
    result2 = encoder(t1)
    #pdb.set_trace()
    
    print(result2[0].shape, result2[1].shape)
            
            
        
        
        
        
            
    