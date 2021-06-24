from __future__ import print_function
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image
#ORG from ssim.ssimlib import SSIM
from skimage.metrics import structural_similarity as SSIM #SJ_TEMP_FIX
from skimage import measure #SJ_FIX

from models.networks import get_generator


def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--img_folder', required=True, help='GoPRO Folder')
	parser.add_argument('--weights_path', required=True, help='Weights path')

	return parser.parse_args()


def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def get_gt_image(path):
  dir, filename = os.path.split(path)
  base, seq = os.path.split(dir)
  print('base : ', base)
  print('seq : ', seq) 
  print('filename :', filename)
  #ORG base, _ = os.path.split(base)
  #ORG img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename)), cv2.COLOR_BGR2RGB)
  img = cv2.cvtColor(cv2.imread(os.path.join(base, 'labels', filename)), cv2.COLOR_BGR2RGB)  # SJ_FIX
  return img


def test_image(model, image_path):
	print('image_path')
	print(image_path)
	img_transforms = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
	size_transform = Compose([
		#ORG PadIfNeeded(736, 1280)
		PadIfNeeded(255, 255) #SJ_TEST
	])
	#ORG crop = CenterCrop(720, 1280)
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_s = size_transform(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		result_image = model(img_tensor)
	result_image = result_image[0].cpu().float().numpy() #it is ORG, even though cpu()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	#ORG result_image = crop(image=result_image)['image']
	result_image = result_image.astype('uint8')
  # Convert to PIL Image
	pImg = Image.fromarray(result_image, mode='RGB') #SJ_TEST
	basedir, seqname = os.path.split(image_path) 
	print('basedir :',basedir)
	print('seqname :',seqname) 
	save_path = basedir + '/generated/'+seqname
	pImg.save(save_path) #SJ_TEST	
	gt_image = get_gt_image(image_path)
	_, filename = os.path.split(image_path)
	psnr = PSNR(result_image, gt_image)
	   
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(gt_image)
	#ORG ssim = SSIM(pilFake).cw_ssim_value(pilReal)
	from SSIM_PIL import compare_ssim #SJ_TEST
	ssim = compare_ssim(pilFake, pilReal) #SJ_TEST
 
	if psnr < 20 and ssim < 0.90: #SJ_TEST
		print('psnr : ', psnr)
		print('ssim : ', ssim)  
		new_path = basedir + '/psnr_below20_ssim90/'+seqname
		pImg.save(new_path) #SJ_TEST	
	return psnr, ssim



def test(model, files):
	psnr = 0
	ssim = 0
	
	for file in tqdm.tqdm(files):
		#print('debug : file')
		print(file)
		cur_psnr, cur_ssim = test_image(model, file)
		psnr += cur_psnr
		ssim += cur_ssim
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))


if __name__ == '__main__':
	args = get_args()
	with open('config/config.yaml') as cfg:
		config = yaml.load(cfg)
	
	model = get_generator(config['model'])
	model.load_state_dict(torch.load(args.weights_path)['model'])
	model = model.cuda()
	#ORG filenames = sorted(glob.glob(args.img_folder + '/test' + '/blur/**/*.png', recursive=True))
	args.img_folder = '/home/shong/SFO_l26_filtered/test/' 
	#ORG filenames = sorted(glob.glob(args.img_folder + '/images' + '**/*.png', recursive=True)) #SJ_TEST
	filenames = sorted(glob.glob(config['test_dir'] + '/images' + '**/*.png', recursive=True)) #SJ_TEST 
	test(model, filenames)
