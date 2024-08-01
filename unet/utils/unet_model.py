import torch
import torch.nn as nn
import functools

class UnetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False): # TODO
        super(UnetGenerator3d, self).__init__()
        #self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True) 
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout) 
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer) 
        if num_downs>=5:
            unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, unet_block, norm_layer=norm_layer) 
            unet_block = UnetSkipConnectionBlock3d(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        else:
            unet_block = UnetSkipConnectionBlock3d(output_nc, ngf*2, unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    #pdb.set_trace()
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock3d(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock3d, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            #upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1)
            
            upsample = nn.Upsample(scale_factor=2)
            conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)

            down = [downconv]
            up = [uprelu, upsample, conv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            #upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)
                        
            upsample = nn.Upsample(scale_factor=2)
            conv = nn.Conv3d(inner_nc, outer_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, conv, upnorm]
            model = down + up
        else:
            #upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                            kernel_size=4, stride=2,
            #                            padding=1, bias=use_bias)
            upsample = nn.Upsample(scale_factor=2)
            conv = nn.Conv3d(inner_nc*2, outer_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, conv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            #pdb.set_trace()
            return torch.cat([self.model(x), x], 1)

if __name__ == '__main__':
    
    img = torch.ones((1, 1, 96, 128, 96))
    model = UnetGenerator3d(input_nc=1, output_nc=1, num_downs=4)
    img1 = model(img)
    
    print(img1.shape)