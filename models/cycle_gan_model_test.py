import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from vggloss import VGGLoss,Perc_out
from util import util, html
import numpy as np
from unet_test.nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
class CycleGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_P', type=float, default=0.1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_S', type=float, default=0.5, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_C', type=float, default=3, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['cycle_A', 'G_B', 'D_B' ,'cycle_B', 'idt_C', 'con', 'task']
        # self.loss_names = ['cycle_A', 'G_B', 'D_B', 'cycle_B', 'idt_C', 'perc_B', 'perc_A', 'con', 'seg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['fake_B']
        visual_names_B = []
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_B.append('idt_C')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['E_A','E_U', 'G_B','D_B','E_B', 'G_A', 'PFC', 'SEG']
        else:  # during test time, only load Gs
            self.model_names = ['E_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netE_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'E_A', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netE_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'E_A', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netE_U = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'E_U', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'G_A', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'G_B', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSEG = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'SEG', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netPFC = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'PFC', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionPerc = VGGLoss()
            # self.criterionSeg = SEGLoss()
            self.criterionCon = torch.nn.TripletMarginLoss(margin=1, p=2)
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE_A.parameters(),self.netG_B.parameters(),self.netE_B.parameters(),self.netE_U.parameters(),
                                self.netG_A.parameters(), self.netPFC.parameters(), self.netSEG.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()), lr=opt.lr/2, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.label_A = input['C' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths']

    def preprocess_input(self, inputs):
        MEANS = (104, 117, 123)
        return inputs - MEANS

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A: underwater domain
        # B: air domain
        self.f_real_A = self.netE_A(self.real_A)
        self.fake_B = self.netG_B(self.f_real_A)  # G_A(A)


