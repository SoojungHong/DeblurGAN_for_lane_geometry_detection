import logging
from functools import partial

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from schedulers import LinearDecay, WarmRestart

import logging #SJ_TEST
from datetime import datetime #SJ_TEST
import sys #SJ_TEST

cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']

    def train(self):
        
        self._init_params()
        
         #SJ_TEST_BEGIN
        if config['continue_train'] == True:
          saved_model_filename = 'last_{}.h5'.format(config['experiment_desc'])
          self.netG.load_state_dict(torch.load(saved_model_filename), strict=False)
          
          #saved_optimizer_filename = 'last_optimizer_{}.h5'.format(config['experiment_desc'])
          #self.optimizer_G.load_state_dict(torch.load(saved_optimizer_filename)) #, strict=False)
          
          self.resume_epoch = config['continue_epoch']
        else:
            self.resume_epoch = 0
        #SJ_TEST_END
        
        #ORG for epoch in range(0, config['num_epochs']):
        for epoch in range(self.resume_epoch, config['num_epochs']): #SJ_TEMP_FIX
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                #logging.info("epoch is "+str(epoch)+" "+str(datetime.now())) #SJ_TEST
                #logging.info("netG.module.unfreeze() is called and epoch is "+str(epoch)+" "+str(datetime.now())) #SJ_TEST
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                #logging.info("best epoch is : "+str(epoch)+" "+str(datetime.now()) ) #SJ_TEST
                torch.save({
                    'model': self.netG.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
                #logging.info("current best model epoch : "+str(epoch)+" "+str(datetime.now()) ) #SJ_TEST
                #torch.save({ #SJ_TEST
                #    'optimizer': self.optimizer_G.state_dict() #SJ_TEST
                #}, 'best_optimizer_{}.h5'.format(self.config['experiment_desc'])) #SJ_TEST
                
            torch.save({
              'model': self.netG.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            
            #torch.save({ #SJ_TEST
            #  'optimizer': self.optimizer_G.state_dict() #SJ_TEST
            #}, 'last_optimizer_{}.h5'.format(self.config['experiment_desc'])) #SJ_TEST
                
            #logging.info("last model epoch : "+str(epoch)+" "+str(datetime.now()) ) #SJ_TEST
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        #logging.info("epoch size : "+str(epoch_size)+" "+ str(datetime.now()) ) #SJ_TEST
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        #logging.info("i is :"+str(i)+" "+ str(datetime.now()))  #SJ_TEST
        for data in tq:
            #logging.info("len(data) in for data in tq : "+ str(len(data)) +' '+str(datetime.now()) ) #SJ_TEST
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)
            
            loss_D = self._update_d(outputs, targets)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item(), loss_content.item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                #logging.info("i is bigger than epoch_size and i is "+ str(i) +"and epoch_size is "+ str(epoch_size)+" "+str(datetime.now()) ) #SJ_TEST
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.item(), loss_content.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                #logging.info("in validate() i is bigger than epoch_size and i is "+ str(i) +"and epoch_size is "+ str(epoch_size) +" "+ str(datetime.now()) ) #SJ_TEST
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        self.netG.cuda()
        
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)


if __name__ == '__main__':
    import torch #sj_TEST
    print(torch.__version__) #SJ_TEST
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f)
        
    #for handler in logging.root.handlers[:]: #SJ_TEMP_FIX
    #  logging.root.removeHandler(handler)    #SJ_TEMP_FIX
    
    #print("log file creation")
    #logging.basicConfig(filename='train_log.log', level=logging.INFO) #SJ_TEST
    
    """
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='train_log.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)  
    logger.removeHandler(sys.stderr)
    """
    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
     
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    
    trainer.train()
