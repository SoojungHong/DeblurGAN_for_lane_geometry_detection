import numpy as np
import torch.nn as nn
#ORG from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM #SJ_TEMP_FIX

from util.metrics import PSNR

import logging #SJ_TEST
from datetime import datetime #SJ_TEST 
import sys #sj_TEST

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()
        #for handler in logging.root.handlers[:]: #SJ_TEST
        #  logging.root.removeHandler(handler) #SJ_TEST
        #import logging
        
        """
        logger = logging.getLogger()
        fhandler = logging.FileHandler(filename='data_monitoring.log', mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.INFO)
        logger.removeHandler(sys.stderr)
        """
          
        #logging.basicConfig(filename='data_monitoring.log', level=logging.INFO) #SJ_TEST

    def get_input(self, data):
        img = data['a']
        #logging.info(str(datetime.now())+" \n [debug] len(data[a]) : \n"+ str(len(img)) ) #SJ_TEST
        #logging.info(str(datetime.now())+"img :\n"+ str(img)) #SJ_TEST
        inputs = img
        targets = data['b']
        #logging.info(str(datetime.now())+" \n [debug] len(data[b]) : \n"+ str(len(targets)) ) #SJ_TEST
        #logging.info(str(datetime.now())+"targets :\n"+ str(targets)) #SJ_TEST
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        #ORG ssim = SSIM(fake, real, multichannel=True)
        ssim = SSIM(fake, real, multichannel=True) # SJ_FIX
        
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
