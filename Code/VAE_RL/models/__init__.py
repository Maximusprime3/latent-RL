from .base import *
from .mssim_vae import MSSIMVAE
from .betatc_vae import *
from .beta_vae import *


vae_models = {'MSSIMVAE':MSSIMVAE, 'BetaTCVAE':BetaTCVAE, 'BetaVAE':BetaVAE}