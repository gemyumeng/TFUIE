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
            # parser.add_argument('--lambda_P', type=float, default=0.1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_S', type=float, default=0.2, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_C', type=float, default=5, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['cycle_A', 'G_B', 'D_B' ,'cycle_B', 'idt_C', 'task', 'con']
        # self.loss_names = ['cycle_A', 'D_B', 'G_B', 'cycle_B', 'perc_A', 'perc_B', 'idt_C', 'con', 'task']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)

            visual_names_B.append('idt_C')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['E_A','E_U', 'G_B','D_B','E_B', 'G_A', 'SEG', 'UNET']
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
        self.netUNET = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'UNET', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators
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
            self.criterionCon = torch.nn.TripletMarginLoss(margin=1, p=2)
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE_A.parameters(),self.netG_B.parameters(),self.netE_B.parameters(),self.netE_U.parameters(),
                                self.netG_A.parameters(),  self.netSEG.parameters(), self.netUNET.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.label_A = input['C' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def preprocess_input(self, inputs):
        MEANS = (104, 117, 123)
        return inputs - MEANS

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A: underwater domain
        # B: air domain

        self.f_real_A = self.netE_A(self.real_A)
        self.u_real_A = self.netE_U(self.real_A)

        self.seg_out = self.netSEG(self.f_real_A)

        self.f_real_B = self.netE_B(self.real_B)

        self.fake_B = self.netG_B(self.f_real_A)  # G_A(A)
        self.unet_out = self.netUNET(self.fake_B)

        self.fake_A = self.netG_A(self.f_real_B, self.u_real_A)  # G_B(B)

        self.f_fake_A = self.netE_A(self.fake_A)
        self.u_fake_A = self.netE_U(self.fake_A)
        self.f_fake_B = self.netE_B(self.fake_B)

        self.rec_B = self.netG_B(self.f_fake_A)   # G_A(G_B(B))
        self.rec_A = self.netG_A(self.f_fake_B, self.u_fake_A)  # G_B(G_A(A))

        self.perc_real_A = self.netPFC(self.real_A)
        self.perc_real_B = self.netPFC(self.real_B)
        self.perc_fake_B = self.netPFC(self.fake_B)
        self.perc_fake_A = self.netPFC(self.fake_A)

    def backward_D_basic(self, netD, real, fake, epoch, total_iters):

        lambda_G = self.opt.lambda_G
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * lambda_G / 2.0
        if (total_iters-1) % 5 == 0:
            loss_D.backward()
        return loss_D

    def backward_D_A(self,epoch, total_iters):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B,epoch, total_iters)

    def backward_D_B(self,epoch, total_iters):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A,epoch, total_iters)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_P = self.opt.lambda_P
        lambda_S = self.opt.lambda_S
        lambda_G = self.opt.lambda_G
        lambda_C = self.opt.lambda_C
        # Identity loss
        if lambda_idt > 0:
            self.f_real_B = self.netE_B(self.real_B)
            self.u_real_B = self.netE_U(self.real_B)
            self.idt_C = self.netG_A(self.f_real_B, self.u_real_B)
            self.loss_idt_C = self.criterionIdt(self.idt_C, self.real_B) * lambda_A * lambda_idt
        else:
            self.loss_idt_C = 0
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * lambda_G
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # VGGLOSS
        # self.loss_perc_A = self.criterionPerc(self.real_A, self.fake_B) * lambda_P
        # self.loss_perc_B = self.criterionPerc(self.real_B, self.fake_A) * lambda_P

        self.loss_task = CE_Loss(self.seg_out, self.label_A, 8) * lambda_S  + CE_Loss(self.unet_out, self.label_A, 8) * lambda_S

        self.loss_con = self.criterionCycle(self.perc_fake_B, self.perc_real_A) / self.criterionCycle(self.perc_fake_B,
                                                                                                  self.perc_real_B) * lambda_C + \
                        self.criterionCycle(self.perc_fake_A, self.perc_real_B) / self.criterionCycle(self.perc_fake_A,
                                                                                                  self.perc_real_A) * lambda_C
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B  + self.loss_idt_C + self.loss_task
        self.loss_G.backward()


    def optimize_parameters(self, total_iters,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  #
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()


        # D_A and D_B
        if (epoch-1) % 1 == 0:
            self.set_requires_grad([self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # # self.backward_D_A()      # calculate gradients for D_A
        # self.backward_D_A(epoch, total_iters)
        self.backward_D_B(epoch, total_iters)  # calculate graidents for D_B
        if (epoch - 1) % 1 == 0:
            self.optimizer_D.step()  # update D_A and D_B's weights