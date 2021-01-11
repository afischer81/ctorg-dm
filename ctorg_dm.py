# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:16:35 2016

@author: Alexander
"""

from __future__ import print_function

import datetime
import logging
import os
import platform
import random
import re
import sys
import time

import numpy
import scipy
from sklearn.model_selection import train_test_split

from misc_utils import *
#from progressbar import *
from sitk_utils import *

class MlData(object):
    """
    """

    def __init__(self, rootDir='', tmpDir='', logger=None):
        self.rootDir = rootDir
        self.tmpDir = ''
        if self.rootDir and not tmpDir:
            self.tmpDir = os.path.join(self.rootDir, 'tmp')
        self.data = {}
        self.channels = [ 'CT' ]
        if logger == None:
            self.log = logging.getLogger(__name__)
            self.log.setLevel(logging.INFO)
        else:
            self.log = logging.getLogger()
        self.target_voxel_size = 1.0

    def getFileList(self, dir):
        """
        Get the list of files in a directory (including subdirs).
        """

        fileList = []
        for root, dirs, files in os.walk(dir):
            if len(files) > 0:
                for f in files:
                    fileList.append(os.path.join(root,f))
        return fileList

    def readData(self, subDir):
        self.log.info('readData() start')
        self.data = {}
        for f in self.getFileList(os.path.join(self.rootDir, subDir)):
            if not f.endswith('nii.gz'):
                continue
            basename = os.path.basename(f)
            if basename.startswith('volume-'):
                patient = basename.replace('volume-', '').replace('.nii.gz', '')
                seq = { 'Type' : 'CT', 'FileName' : f }
            elif basename.startswith('labels-'):
                patient = basename.replace('labels-', '').replace('.nii.gz', '')
                seq = { 'Type' : 'Labels', 'FileName' : f }
            if not patient in self.data.keys():
                self.data[patient] = { 'Name' : patient, 'Sequences' : [ ] }
            self.data[patient]['Sequences'].append(seq)
            self.log.info('  ' + patient)
        self.log.info('readData() finish')

    def preprocess1(self, numPatients=0, bodyThreshold=-150, force=False):
        """
        image prepreprocessing stage 1: body mask
        """
        log.info('preprocess1() start')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        for p in patientList:
            log.info('  patient ' + p)
            pDir = os.path.join(self.tmpDir, p)
            if not os.path.exists(pDir):
                os.makedirs(pDir)
            maskFileName = os.path.join(self.tmpDir, p, 'mask.nii.gz')
            if force or not os.path.exists(maskFileName):
                s = FindData(self.data[p]['Sequences'], { 'Type' : 'CT' })
                img = sitk.ReadImage(s[0]['FileName'])
                size = img.GetSize()
                # find a point on the skin
                startPos = [ 0, size[1] // 2, size[2] // 2 ]
                seedPoint = None
                for y in range(startPos[1], startPos[1] + size[1] // 4, size[1] // 16):
                    for x in range(startPos[0], size[0] // 2):
                        value = img.GetPixel(x, y, startPos[2])
                        if value >= bodyThreshold:
                            seedPoint = [ x, y, startPos[2] ]
                            break
                log.info('    SeedPoint = {} {}'.format(seedPoint, value))
                mask = sitk.ConnectedThreshold(img, seedList=[seedPoint], lower=bodyThreshold, upper=5000)
                # closing
                mask = grayscaleClose(mask, radius=[ 2, 2, 2 ])
                # directional opening (table)
                mask = grayscaleOpen(mask, radius=[ 0, 2, 0 ])
                # small object removal
                mask = relabelComponent(mask, 10)
                # bottom and top slice closing/hole filling
                mask_bottom = extract(mask, [0, 0, 0 ] , [ size[0], size[1], 0 ] )
                log.debug('mask_bottom {}'.format(numpy.count_nonzero(sitk.GetArrayFromImage(mask_bottom))))
                mask_bottom = binaryFillHole(mask_bottom)
                log.debug('mask_bottom {}'.format(numpy.count_nonzero(sitk.GetArrayFromImage(mask_bottom))))
                mask_top = extract(mask, [0, 0, size[2]-1] , [ size[0], size[1], 0 ] )
                log.debug('mask_top {}'.format(numpy.count_nonzero(sitk.GetArrayFromImage(mask_top))))
                mask_top = binaryFillHole(mask_top)
                log.debug('mask_top {}'.format(numpy.count_nonzero(sitk.GetArrayFromImage(mask_top))))
                # copy filtered 2D mask slices back to 3D mask
                for y in range(size[1]):
                    for x in range(size[0]):
                        mask.SetPixel(x, y, 0, mask_bottom.GetPixel(x, y))
                        mask.SetPixel(x, y, size[2] - 1, mask_top.GetPixel(x, y))
                # hole filling 3D
                mask = binaryFillHole(mask)
                # final dilation
                mask = grayscaleDilate(mask, radius=[ 2, 2, 2 ])
                sitk.WriteImage(mask, maskFileName)
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Mask' })
            if len(s) == 0:
                self.data[p]['Sequences'].append({ 'Type' : 'Mask', 'FileName' : maskFileName })
        log.info('preprocess1() finish')

    def preprocess2(self, numPatients=0, force=False):
        """
        preprocessing stage 2: : label splitting
        """
        labels = { 1: 'Liver', 2: 'Bladder', 3: 'Lungs', 4: 'Kidneys', 5: 'Bone', 6: 'Brain' }

        log.info('preprocess2() start')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        for p in patientList:
            log.info('  patient ' + p)
            pDir = os.path.join(self.tmpDir, p)
            if not os.path.exists(pDir):
                os.makedirs(pDir)
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Labels' })
            imgFileName = s[0]['FileName']
            img = None
            labelData = None
            for l in range(1, 7):
                labelFileName = os.path.join(self.tmpDir, p, labels[l] + '.nii.gz')
                if force or not os.path.exists(labelFileName):
                    if img is None:
                        img = sitk.ReadImage(imgFileName)
                        labelData = sitk.GetArrayFromImage(img)
                    data = (labelData == l).astype(numpy.uint8)
                    labelImg = sitk.GetImageFromArray(data)
                    labelImg.CopyInformation(img)
                    n = numpy.count_nonzero(data)
                    if n > 0:
                        sitk.WriteImage(labelImg, labelFileName)
                        log.debug('{} active voxels in label {}'.format(n, labels[l]))
                    else:
                        log.warning('label {} empty'.format(labels[l]))
                if os.path.exists(labelFileName):
                    self.log.info('label {} {}'.format(l, labels[l]))
                    s = FindData(self.data[p]['Sequences'], { 'Type' : 'Label_' + labels[l] })
                    if len(s) == 0:
                        self.data[p]['Sequences'].append({ 'Type' : 'Label_' + labels[l], 'FileName' : labelFileName })
                else:
                    self.log.warning('no label {} {}'.format(l, labels[l]))
        log.info('preprocess2() finish')

    def preprocess3(self, label, crop_boundary, global_align=False, numPatients=0, force=False):
        """
        preprocessing stage 3: crop images and resample to uniform resolution
        """

        def crop_and_pad_and_resample(img_in, img_crop, img_pad, target_size, f_out, is_label=False, force=False):
            if not os.path.exists(f_out) or force:
                if type(img_in) == type('') or type(img_in) == type(u''):
                    img = sitk.ReadImage(img_in)
                else:
                    img = img_in
                img = crop(img, tuple(map(int, img_crop[0])), tuple(map(int, img_crop[1])))
                pad_constant = -1024
                if is_label:
                    pad_constant = 0
                if img_pad[0][0] > 0 or img_pad[0][1] > 0 or img_pad[0][2] > 0 or img_pad[1][0] > 0 or img_pad[1][1] > 0 or img_pad[1][2] > 0:
                    img = pad(img, tuple(map(int, img_pad[0])), tuple(map(int, img_pad[1])), pad_constant)
                img = resample(img, is_label=is_label)
                img_size = img.GetSize()
                if img_size[0] != target_size[0] or img_size[1] != target_size[1] or img_size[2] != target_size[2]:
                    p = numpy.abs(target_size - numpy.array(img_size))
                    img = pad(img, (0, 0, 0), tuple(map(int, p)), pad_constant)
                self.log.info('cropped/padded/resampled {} spacing {}'.format(img.GetSize(), img.GetSpacing()))
                if is_label:
                    img = sitk.Cast(img, sitk.sitkUInt8)
                sitk.WriteImage(img, f_out)

        log.info('preprocess3() start')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        object_size = numpy.array([ 0.0, 0.0, 0.0 ])
        if global_align:
            #
            # identify the physical size of the largest object to be extracted
            #
            for p in patientList:
                log.info('  patient ' + p)
                s = FindData(self.data[p]['Sequences'], { 'Type' : 'Label_' + label })
                if len(s) == 0:
                    log.info('no label {} found for patient {}'.format(label, p))
                    continue
                mask_file_name = s[0]['FileName']
                mask = sitk.ReadImage(mask_file_name)
                voxel_size = mask.GetSpacing()
                bbox = boundingBox(mask)
                size = (numpy.array(bbox[1][1::2]) - numpy.array(bbox[1][::2])) * numpy.array(voxel_size)
                log.info('  bbox {} size {}'.format(bbox[1], size))
                for i in range(3):
                    object_size[i] = max(size[i], object_size[i])
        object_volume = numpy.prod(object_size)
        if object_volume > 0.0:
            log.info('physical object size {} volume {}'.format(object_size, object_volume))
        #
        # crop the identified object size relative to the center of the label object plus the margin
        # pad, if the image is exceeded
        #
        for p in patientList:
            log.info('  patient ' + p)
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Label_' + label })
            if len(s) == 0:
                log.info('no label {} found for patient {}'.format(label, p))
                continue
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'CT' })
            img_file_name = s[0]['FileName']
            label_file_name = s[0]['FileName']
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Mask' })
            mask_file_name = s[0]['FileName']

            lbl = sitk.ReadImage(label_file_name)
            mask = sitk.ReadImage(mask_file_name)
            lbl_array = sitk.GetArrayViewFromImage(lbl)
            mask_array = sitk.GetArrayViewFromImage(mask)
            overlap = numpy.count_nonzero(numpy.logical_and(lbl_array, mask_array))
            if overlap == 0:
                log.warning('mask and label do not overlap')
                continue

            img = sitk.ReadImage(img_file_name)
            img_size = numpy.array(img.GetSize())
            voxel_size = numpy.array(img.GetSpacing())

            # normalize to mean 0 std dev 1.0
            img = normalize(sitk.Cast(img, sitk.sitkFloat32), mask)
            # crop and pad parameters
            bbox = boundingBox(lbl)
            crop_center = (numpy.array(bbox[1][::2]) + numpy.array(bbox[1][1::2])) / 2 * voxel_size
            if object_volume > 0.0:
                crop_size = (object_size + 2 * crop_boundary) / voxel_size
            else:
                crop_size = numpy.array(bbox[1][1::2]) - numpy.array(bbox[1][::2]) + 2 * crop_boundary / voxel_size
                object_size = (numpy.array(bbox[1][1::2]) - numpy.array(bbox[1][::2])) * voxel_size
            crop_lower = numpy.round((crop_center - object_size / 2 - crop_boundary) / voxel_size).astype(numpy.int16)
            crop_size = (4 * numpy.round(crop_size / 4 + 0.5)).astype(numpy.int16)
            log.info('crop {} - {} in {}'.format(crop_lower, crop_lower + crop_size, img_size))
            crop_upper = crop_lower + crop_size
            pad_lower = numpy.array([ 0, 0, 0 ])
            pad_upper = numpy.array([ 0, 0, 0 ])
            for i in range(3):
                if crop_lower[i] < 0:
                    pad_lower[i] = -crop_lower[i]
                    crop_lower[i] = 0
                crop_upper[i] = img_size[i] - crop_upper[i]
                if crop_upper[i] < 0:
                    pad_upper[i] = -crop_upper[i]
                    crop_upper[i] = 0

            target_size = (object_size + 2 * crop_boundary) / self.target_voxel_size
            target_size = (4 * numpy.round(target_size / 4 + 0.5)).astype(numpy.int16)
            log.info('{} crop {} {} pad {} {}'.format(p, crop_lower, crop_upper, pad_lower, pad_upper))
            cropImgFileName = os.path.join(self.tmpDir, p, 'crop_img_' + p + '_' + label + '.nii.gz')
            crop_and_pad_and_resample(img, (crop_lower, crop_upper), (pad_lower, pad_upper), target_size, cropImgFileName, force=force)
            self.check_and_add_data(p, 'crop_img_' + label, cropImgFileName)

            cropLabelFileName = os.path.join(self.tmpDir, p, 'crop_label_' + p + '_' + label + '.nii.gz')
            crop_and_pad_and_resample(lbl, (crop_lower, crop_upper), (pad_lower, pad_upper), target_size, cropLabelFileName, is_label=True, force=force)
            self.check_and_add_data(p, 'crop_label_' + label, cropLabelFileName)

            cropMaskFileName = os.path.join(self.tmpDir, p, 'crop_mask_' + p + '_' + label + '.nii.gz')
            crop_and_pad_and_resample(mask, (crop_lower, crop_upper), (pad_lower, pad_upper), target_size, cropMaskFileName, is_label=True, force=force)
            self.check_and_add_data(p, 'crop_mask_' + label, cropMaskFileName)

        log.info('preprocess3() finish')

    def load(self, file_name):
        with open(file_name) as f:
            self.data = json.load(f)

    def save(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def check_and_add_data(self, p, tag, file_name):
        result = False
        s = FindData(self.data[p]['Sequences'], { 'Type' : tag })
        if len(s) == 0:
            self.data[p]['Sequences'].append({ 'Type' : tag, 'FileName' : file_name })
            result = True
        return result

    def create_config(self, mode, label, patient_list):
        """
        Training
        """
        f_channels = open('config/{}Channels_ct.cfg'.format(mode), 'w')
        f_labels = open('config/{}GtLabels.cfg'.format(mode), 'w')
        f_masks = open('config/{}RoiMasks.cfg'.format(mode), 'w')
        f_pred =  open('config/{}NamesOfPredictions.cfg'.format(mode), 'w')
        num_valid = 0
        for p in patient_list:
            log.info('  patient ' + p)
            img = None
            mask = None
            lbl = None
            for s in self.data[p]['Sequences']:
                if s['Type'] == 'crop_img_' + label:
                    img = s['FileName']
                if s['Type'] == 'crop_label_' + label:
                    lbl = s['FileName']
                if s['Type'] == 'crop_mask_' + label:
                    mask = s['FileName']
            if not img is None and not mask is None and not lbl is None:
                f_channels.write(img + '\n')
                f_masks.write(mask + '\n')
                f_labels.write(lbl + '\n')
                f_pred.write('{}_{}_predict.nii.gz'.format(os.path.basename(img).replace('.nii.gz', ''), label) + '\n')
                log.info('  OK')
                num_valid += 1
            else:
                log.warning('  skipped')
        f_pred.close()
        f_masks.close()
        f_labels.close()
        f_channels.close()
        log.info('{} valid patients for {}ing'.format(num_valid, mode))

    def train(self, label, validation_fraction=0.0, seed=42, numPatients=0):
        """
        Training
        """
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        trainList, validationList = train_test_split(patientList, test_size=validation_fraction, random_state=seed)
        self.create_config('train', label, trainList)
        self.create_config('validation', label, validationList)

    def inference(self, label, numPatients=0):
        """
        Inference
        """
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        self.create_config('test', label, patientList)
        return

    def plot(self, label, p):

        s = FindData(self.data[p]['Sequences'], { 'Type' : 'crop_img_' + label })
        img_file_name = s[0]['FileName']
        s = FindData(self.data[p]['Sequences'], { 'Type' : 'crop_label_' + label })
        if len(s) == 0:
            log.info('no label {} found for patient {}'.format(label, p))
            return
        label_file_name = s[0]['FileName']
        s = FindData(self.data[p]['Sequences'], { 'Type' : 'crop_mask_' + label })
        mask_file_name = s[0]['FileName']

        img = sitk.ReadImage(img_file_name)
        img = clamp(img, -5.0, 5.0)
        lbl = sitk.ReadImage(label_file_name)
        mask = sitk.ReadImage(mask_file_name)

        stat = statistics(img, lbl)
        log.info(stat)

        lbl_array = sitk.GetArrayViewFromImage(lbl)
        lbl_cog = scipy.ndimage.center_of_mass(lbl_array,labels=[1])
        x = int(round(lbl_cog[2]))
        y = int(round(lbl_cog[1]))
        z = int(round(lbl_cog[0]))
        print(x, y, z, img.GetSize(), lbl.GetSize())

        sitk_show(img[x,:,::-1], dpi=80, clim=(-5,5))
        sitk_show(img[:,y,::-1], dpi=80, clim=(-5,5))
        sitk_show(img[:,:,z], dpi=80, clim=(-5,5))

        img_sc = sitk.Cast(sitk.RescaleIntensity(img), lbl.GetPixelID())

        sitk_show(sitk.LabelOverlay(img_sc[x,:,::-1], lbl[x,:,::-1], opacity=0.1), clim=(-5,5))
        sitk_show(sitk.LabelOverlay(img_sc[:,y,::-1], lbl[:,y,::-1], opacity=0.1), clim=(-5,5))
        sitk_show(sitk.LabelOverlay(img_sc[:,:,z], lbl[:,:,z], opacity=0.1), clim=(-5,5))


if __name__ == '__main__':
    import argparse

    self = os.path.basename(sys.argv[0])
    myName = os.path.splitext(self)[0]
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(myName)
    log.setLevel(logging.INFO)

    if platform.system() == 'Windows':
        rootDir = r'E:\Data\TCIA\CT-ORG'
    else:
        rootDir = '/mnt/e/Data/TCIA/CT-ORG'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', action='store_true', help='debug execution')
    parser.add_argument('-f', '--force', action='store_true', help='enforce recalculation (ignore existing results)')
    parser.add_argument('-i', '--inference', default='', help='run inference')
    parser.add_argument('-n', '--num', type=int, default=0, help='limit number of patients (0 = all)')
    parser.add_argument('--plot', type=str, default='', help='plot patient #')
    parser.add_argument('-p', '--pre', type=int, default=-1, help='execute preprocessing stage (-1: none, 0: all)')
    parser.add_argument('-r', '--root', default=rootDir, help='root directory for data ({})'.format(rootDir))
    parser.add_argument('--seed', type=int, default=42, help='randomization seed')
    parser.add_argument('-t', '--train', default='', help='run training on given label(s)')
    parser.add_argument('--validation_fraction', type=float, default=0.2, help='validation fraction')
    args = parser.parse_args(sys.argv[1:])

    if args.debug:
        log.setLevel(logging.DEBUG)
    mode = 'train'
    label = args.train
    if args.inference:
        mode = 'test'
        label = args.inference
    data = MlData(args.root)
    data_file_name = 'ctorg_{}_data.json'.format(mode)
    if not args.force and os.path.exists(data_file_name):
        data.load(data_file_name)
        log.info('{} patients loaded'.format(len(data.data)))
    else:
        data.readData(mode)
    if args.pre >= 0:
        if args.pre == 0 or args.pre == 1:
            data.preprocess1(numPatients=args.num, force=args.force)
            data.save(data_file_name)
        if args.pre == 0 or args.pre == 2:
            data.preprocess2(numPatients=args.num, force=args.force)
            data.save(data_file_name)
        if args.pre == 0 or args.pre == 3:
            data.preprocess3(label, crop_boundary=15.0, global_align=False, numPatients=args.num, force=args.force)
            data.save(data_file_name)
    elif args.plot:
        data.plot(label, args.plot)
    elif args.train:
        data.train(args.train, validation_fraction=args.validation_fraction, seed=args.seed, numPatients=args.num)
    elif args.inference:
        data.inference(args.inference, numPatients=args.num)
