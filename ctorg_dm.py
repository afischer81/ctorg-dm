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

    def readData(self, subDir):
        self.log.info('readData() start')
        self.data = {}
        for f in GetFileList(os.path.join(self.rootDir, subDir)):
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
            if not force and os.path.exists(maskFileName):
                mask = sitk.ReadImage(maskFileName)
            else:
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
            img = sitk.ReadImage(s[0]['FileName'])
            labelData = sitk.GetArrayFromImage(img)
            for l in range(1, 7):
                labelFileName = os.path.join(self.tmpDir, p, labels[l] + '.nii.gz')
                if force or not os.path.exists(labelFileName):
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
                    self.data[p]['Sequences'].append({ 'Type' : 'Label_' + labels[l], 'FileName' : labelFileName })
        log.info('preprocess2() finish')

    def preprocess3(self, label, crop_boundary, numPatients=0, force=False):
        """
        preprocessing stage 3: identify bounding box to crop images with margin
        """

        log.info('preprocess3() start')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        for p in patientList:
            log.info('  patient ' + p)
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'CT' })
            img_file_name = s[0]['FileName']
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Label_' + label })
            if len(s) == 0:
                log.info('no label {} found for patient {}'.format(label, p))
                continue

            # image
            imgFileName = os.path.join(self.tmpDir, p, 'resimg_' + label + '.nii.gz')
            if not os.path.exists(imgFileName) or force:
                img = sitk.ReadImage(img_file_name)
                img = resample(img)
                sitk.WriteImage(img, imgFileName)
            else:
                img = sitk.ReadImage(imgFileName)

            # label file/mask
            label_file_name = s[0]['FileName']
            labelFileName = os.path.join(self.tmpDir, p, 'resmask_' + label + '.nii.gz')
            if not os.path.exists(labelFileName) or force:
                mask = sitk.ReadImage(label_file_name)
                mask = resample(mask, is_label=True)
                sitk.WriteImage(mask, labelFileName)
            else:
                mask = sitk.ReadImage(labelFileName)

            # body mask
            s = FindData(self.data[p]['Sequences'], { 'Type' : 'Mask' })
            mask_file_name = s[0]['FileName']
            maskFileName = os.path.join(self.tmpDir, 'res' + os.path.basename(mask_file_name))
            if not os.path.exists(maskFileName) or force:
                mask = sitk.ReadImage(mask_file_name)
                mask = resample(mask, is_label=True)
                sitk.WriteImage(mask, maskFileName)

            # bbox = [ xstart, xend, ystart, yend, zstart, zend ]
            bbox = boundingBox(mask)
            img_size = img.GetSize()
            voxel_size = img.GetSpacing()
            log.info('img {} {} {}'.format(imgFileName, img_size, bbox[1]))
            d = [ round(crop_boundary / voxel_size[0]), round(crop_boundary / voxel_size[1]), round(crop_boundary / voxel_size[2]) ]
            crop_lower = ( max(0, bbox[1][0] - d[0]), max(0, bbox[1][2] - d[1]), max(0, bbox[1][4] - d[2]) )
            crop_size = (
                min(img_size[0] - crop_lower[0], bbox[1][1] - crop_lower[0] + d[0]),
                min(img_size[1] - crop_lower[1], bbox[1][3] - crop_lower[1] + d[1]),
                min(img_size[2] - crop_lower[2], bbox[1][5] - crop_lower[2] + d[2])
            )
            #crop_upper = [
            #    img_size[0] - (crop_lower[0] + crop_size[0]),
            #    img_size[1] - (crop_lower[1] + crop_size[1]),
            #    img_size[2] - (crop_lower[2] + crop_size[2])
            #]
            #crop_img = crop(img, crop_lower, crop_upper)
            #cropImgFileName = os.path.join(os.path.dirname(imgFileName), 'crop' + os.path.basename(imgFileName))
            #log.info('cropped img {} spacing {}'.format(crop_img.GetSize(), crop_img.GetSpacing()))
            #sitk.WriteImage(crop_img, cropImgFileName)
            log.info('crop {} {}'.format(crop_lower, crop_size))
            self.data[p]['crop'] = [ crop_lower, crop_size ]
        log.info('preprocess3() finish')

    def preprocess4(self, label, numPatients=0, force=False):
        """
        preprocessing stage 4: crop images to desired label with margin and resample to uniform resolution
        """
        log.info('preprocess4() start')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        # identify the largest object to be extracted
        crop_size = [ 0, 0, 0 ]
        for p in patientList:
            if not 'crop' in self.data[p].keys():
                log.info('no label {} found for patient {}'.format(label, p))
                continue
            crop_size[0] = max(crop_size[0], self.data[p]['crop'][1][0])
            crop_size[1] = max(crop_size[1], self.data[p]['crop'][1][1])
            crop_size[2] = max(crop_size[2], self.data[p]['crop'][1][2])
        # round to multiple of 4 and make square slices
        crop_size = [ 4 * round(crop_size[0] / 4 + 0.5), 4 * round(crop_size[1] / 4 + 0.5), 4 * round(crop_size[2] / 4 + 0.5) ]
        crop_size[0] = max(crop_size[0], crop_size[1])
        crop_size[1] = max(crop_size[0], crop_size[1])
        log.info('crop size {}'.format(crop_size))
        for p in patientList:
            log.info('  patient ' + p)
            imgFileName = os.path.join(self.tmpDir, p, 'img_' + label + '.nii.gz')
            if not os.path.exists(imgFileName):
                continue
            cropImgFileName = os.path.join(os.path.dirname(imgFileName), 'crop' + os.path.basename(imgFileName))
            if not os.path.exists(cropImgFileName) or force:
                img = sitk.ReadImage(imgFileName)
                img_size = img.GetSize()
                p_crop = self.data[p]['crop']
                log.info('{} p_crop {} {}'.format(p, p_crop[0], p_crop[1]))
                crop_lower = list(p_crop[0])
                crop_upper = list(img_size)
                img_pad = [ 0, 0, 0 ]
                for i in range(3):
                    crop_lower[i] += (p_crop[1][i] - crop_size[i]) // 2
                    if crop_lower[i] < 0:
                        img_pad[i] -= crop_lower[i]
                        crop_lower[i] = 0
                    crop_upper[i] -= crop_lower[i] + crop_size[i]
                    if crop_upper[i] < 0:
                        img_pad[i] -= crop_upper[i]
                        crop_upper[i] = 0
                log.info('{} crop {} {} pad {}'.format(p, tuple(crop_lower), tuple(crop_upper), img_pad))
                img = crop(img, crop_lower, crop_upper)
                if img_pad[0] > 0 or img_pad[1] > 0 or img_pad[2] > 0:
                    pad_lower = ( img_pad[0] // 2, img_pad[1] // 2, img_pad[2] // 2 )
                    pad_upper = ( img_pad[0] - pad_lower[0], img_pad[1] - pad_lower[1], img_pad[2] - pad_lower[2] )
                    img = pad(img, pad_lower, pad_upper)
                log.info('cropped/padded img {} spacing {}'.format(img.GetSize(), img.GetSpacing()))
                sitk.WriteImage(img, cropImgFileName)
                self.data[p]['Sequences'].append({ 'Type' : 'CT_crop_' + label, 'FileName' : cropImgFileName })

            maskFileName = os.path.join(self.tmpDir, p, 'mask_' + label + '.nii.gz')
            cropMaskFileName = os.path.join(os.path.dirname(maskFileName), 'crop' + os.path.basename(maskFileName))
            if not os.path.exists(cropMaskFileName) or force:
                mask = sitk.ReadImage(maskFileName)
                mask = crop(mask, crop_lower, crop_upper)
                if img_pad[0] > 0 or img_pad[1] > 0 or img_pad[2] > 0:
                    mask = pad(mask, pad_lower, pad_upper)
                log.info('cropped/padded mask {} spacing {}'.format(mask.GetSize(), mask.GetSpacing()))
                sitk.WriteImage(mask, cropMaskFileName)
                self.data[p]['Sequences'].append({ 'Type' : 'mask_crop_' + label, 'FileName' : cropImgFileName })

        log.info('preprocess4() finish')

    def load(self, file_name):
        with open(file_name) as f:
            self.data = json.load(f)

    def save(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.data, f)

    def create_config(self, mode, label, numPatients=0):
        """
        Training
        """
        f_channels = open('config/{}Channels_ct.cfg'.format(mode), 'w')
        f_labels = open('config/{}GtLabels.cfg'.format(mode), 'w')
        f_masks = open('config/{}RoiMasks.cfg'.format(mode), 'w')
        if mode == 'test':
            f_pred =  open('config/{}NamesOfPredictions.cfg'.format(mode), 'w')
        patientList = sorted(self.data.keys())
        if numPatients > 0:
            patientList = patientList[:numPatients]
        num_valid = 0
        for p in patientList:
            log.info('  patient ' + p)
            img = None
            mask = None
            lbl = None
            for s in self.data[p]['Sequences']:
                if s['Type'] == 'CT':
                    img = s['FileName']
                if s['Type'] == 'Mask':
                    mask = s['FileName']
                if s['Type'] == 'Label_' + label:
                    lbl = s['FileName']
            if not img is None and not mask is None and not lbl is None:
                f_channels.write(img + '\n')
                f_masks.write(mask + '\n')
                f_labels.write(lbl + '\n')
                if mode == 'test':
                    f_pred.write('{}_{}_predict.nii.gz'.format(os.path.basename(img).replace('.nii.gz', ''), label) + '\n')
                log.info('  OK')
                num_valid += 1
            else:
                log.warning('  skipped')
        if mode == 'test':
            f_pred.close()
        f_masks.close()
        f_labels.close()
        f_channels.close()
        log.info('{} valid patients for {}ing'.format(num_valid, mode))

    def train(self, label, numPatients=0):
        """
        Training
        """
        self.create_config('train', label, numPatients)

    def inference(self, label, numPatients=0):
        """
        Inference
        """
        self.create_config('test', label, numPatients)
        return

def Inspect(patientData):
    seq = patientData['Sequences']
    # find x center of tumor
    otseq = FindData(seq, { 'Type' : 'OT' })
    ot = sitk.ReadImage(otseq[0]['FileName'])
    bbox = boundingBox(ot)
    xMin = 1000
    xMax = 0
    for i in list(bbox.keys()):
        if i == 0:
            continue
        xMin = min([ bbox[i][0], xMin ])
        xMax = max([ bbox[i][1], xMax ])
    x = int(round((xMin + xMax) / 2))
    print(patientData['Name'], x)
    flairseq = FindData(seq, { 'Type' : 'Flair' })
    flair = sitk.ReadImage(flairseq[0]['FileName'])
    sitk_show(flair[x,:,::-1])
    print('Flair')
    t1seq = FindData(seq, { 'Type' : 'T1' })
    t1 = sitk.ReadImage(t1seq[0]['FileName'])
    sitk_show(t1[x,:,::-1])
    print('T1')
    t2seq = FindData(seq, { 'Type' : 'T2' })
    t2 = sitk.ReadImage(t2seq[0]['FileName'])
    sitk_show(t2[x,:,::-1])
    print('T2')
    sitk_show(ot[x,:,::-1])
    print('OT')
    maskseq = FindData(seq, { 'Type' : 'Mask' })
    mask = sitk.ReadImage(maskseq[0]['FileName'])
    # Rescale 't1smooth' and cast it to an integer type to match that of 'mask'
    t1Int = sitk.Cast(sitk.RescaleIntensity(t1), mask.GetPixelID())
    sitk_show(sitk.LabelOverlay(t1Int[x,:,::-1], mask[x,:,::-1]))
    print('Mask')
    sitk_show(sitk.LabelOverlay(t1Int[x,:,::-1], ot[x,:,::-1]))
    print('Tumor')

    for img in [ flair, t1, t2 ]:
        for voi in [ mask, ot ]:
            stat = statistics(img, voi)
            for label in list(stat.keys()):
                print(label, stat[label]['Min'], stat[label]['Max'], stat[label]['Mean'])
    #print statistics(t1, mask)[1]
    #patches, labels = loadData(patientData['TrainData'])
    #print len(patches), patches[0].shape, 'patches', len(labels), 'labels'

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
        rootDir = '/mnt/e/Data/BRATS2015'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', action='store_true', help='debug execution')
    parser.add_argument('-f', '--force', action='store_true', help='enforce recalculation (ignore existing results)')
    parser.add_argument('-i', '--inference', default='', help='run inference')
    parser.add_argument('-n', '--num', type=int, default=0, help='limit number of patients (0 = all)')
    parser.add_argument('-r', '--root', default=rootDir, help='root directory for data ({})'.format(rootDir))
    parser.add_argument('-t', '--train', default='', help='run training on given label(s)')
    args = parser.parse_args(sys.argv[1:])

    if args.debug:
        log.setLevel(logging.DEBUG)
    mode = 'train'
    label = args.train
    if args.inference:
        mode = 'test'
        label = args.inference
    data = MlData(rootDir)
    data_file_name = 'ctorg_{}_data.json'.format(mode)
    if not args.force and os.path.exists(data_file_name):
        data.load(data_file_name)
        log.info('{} patients loaded'.format(len(data.data)))
    else:
        data.readData(mode)
        data.preprocess1(numPatients=args.num, force=args.force)
        data.preprocess2(numPatients=args.num, force=args.force)
        data.preprocess3(label, crop_boundary=15.0, numPatients=args.num, force=True)
        #data.preprocess4(label, numPatients=args.num, force=True)
        data.save(data_file_name)
    #if args.train:
    #    data.train(args.train, numPatients=args.num)
    #if args.inference:
    #    data.inference(args.inference, numPatients=args.num)
