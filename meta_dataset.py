
# %%

import os, sys
import pickle as pk
import json
import pandas as pd
import torch
import torch.nn as nn
from zipfile import ZipFile, BadZipFile
from PIL import Image
from glob import glob
import io
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as tf
from torchvision.transforms.transforms import ToPILImage

gray2rgb = tf.Lambda(lambda x: x.expand(3,-1,-1) )

img2tensor = tf.Compose([
	tf.ToTensor(),
])

# applicable to data that load as grayscale images
default_transform_gray = tf.Compose([
	tf.ToPILImage(),
	tf.Resize(256),
	tf.CenterCrop(256),
	tf.ToTensor(),
	gray2rgb
])

default_transform_rgb = tf.Compose([
	tf.ToPILImage(),
	tf.Resize(256),
	tf.CenterCrop(256),
	tf.ToTensor()
])

def default_open_json(jsonname):
	with open(jsonname) as fl:
		blob = json.load(fl)
	return blob

def default_open_imageZip_inmem(zipname):
	ims = []
	try:
		with ZipFile(zipname) as blob:
			imglist = blob.namelist()

			for imname in imglist:
				imgdata = blob.read(imname)
				img = Image.open(io.BytesIO(imgdata))
				ims += [img2tensor(img)]
	except BadZipFile:
		print()
		print('Invalid zipfile:', zipname)
		return None

	return ims

def default_open_npz_images(fname):
	blob = np.load(fname)

	# NOTE: preferabbly using modular metadata files we don't have to specify this
	vol = blob['oct_volume']

	imgs = [img2tensor(Image.fromarray(mat)) for mat in vol]

	return imgs

class MetaDataset(Dataset):
	def __init__(self,
			metafile,
			transform=default_transform_gray,
			load_metadata=lambda fname: pd.read_csv(fname),
			data_format='npz', # or zip
			get_samples=lambda meta: meta['filename'].values,
			post_format_fn=None,
		):

		# with open(metadata) as fl:
		# 	metadata = json.load(fl)

		metadata = load_metadata(metafile)

		self.samples = get_samples(metadata)

		self.t = transform

		self.post_format_fn = post_format_fn
		self.data_reader = dict(
			zip=default_open_imageZip_inmem,
			npz=default_open_npz_images
		)[data_format]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]

		# torch tensor (image) or list of tensors (volume)
		imgs = self.data_reader(sample)

		print(len(imgs), imgs[0].shape)

		if self.post_format_fn is not None:
			imgs = self.post_format_fn(imgs)

		if self.t is not None:
			imgs = [self.t(im) for im in imgs]

		t_imgs = torch.stack(imgs)

		return t_imgs, sample


# %%
if __name__ == '__main__':
	pass

	# %%

	npzExample = MetaDataset(
		metafile='/data1/Ophthalmology/OCT/EMMES/EMMES_oct_meta.csv',
		# NOTE: do we want the file format specified in the metadata file?
		data_format='npz',
		# NOTE: do we want file extensions in the metadata files?
		get_samples=lambda df: [(fname+'.npz') for fname in df[df['num_slices'] >= 90]['filename'].values]
	)

	print(len(npzExample))
	vol, _ = npzExample[0]
	print(vol.shape)

	# %%

	ds_params = dict(n_slices=97)

	def safe_read_nslice_zipped_images(imgs):
		# Some jobs need more logic than default behavior (e.g. ukbb jobf)
		# 1. choose center slices
		# 2. check if zip file was corrupted (unless we can check this beforehand)
		# 3. check for any other issues

		n_slices = ds_params['n_slices']

		if imgs is None:
			# corrupted: return empty volume
			return [torch.zeros(3, 256, 256) for _ in range(n_slices)]
		else:
			# choose which slices to focus on
			m = len(imgs)//2
			imgs = imgs[m-n_slices//2:m+1+n_slices//2]

			try:
				assert len(imgs) == n_slices
			except:
				raise BaseException('# Slices Mismatch:', len(imgs), n_slices)

			return imgs

	ukbbExample = MetaDataset(
		# NOTE: json or pandas?
		metafile='/data1/Ophthalmology/OCT/ukbb/oct_right_flat.json',
		load_metadata=default_open_json,
		get_samples=lambda meta: meta['samples'],

		data_format='zip',

		post_format_fn=safe_read_nslice_zipped_images,
	)

	print(len(ukbbExample))
	vol, _ = ukbbExample[0]
	print(vol.shape)


# %%
