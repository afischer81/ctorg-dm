# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:16:35 2016

@author: Alexander
"""

import logging

import numpy
import SimpleITK as sitk

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    import matplotlib.pyplot as plt

    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    if title:
        plt.title(title)
    plt.show()

def biasFieldCorrection(image, mask):
    filter = sitk.N4BiasFieldCorrectionImageFilter()
    result = filter.Execute(image, mask)
    return result

def binaryDilate(image, type=sitk.sitkCross, radius=1):
    log.info('binaryDilate() start')
    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelType(type)
    filter.SetKernelRadius(radius)
    result = filter.Execute(image)
    log.info('binaryDilate() finish')
    return result

def binaryFillHole(image, fullyConnected=False):
    log.info('binaryFillHole() start')
    filter = sitk.BinaryFillholeImageFilter()
    filter.SetFullyConnected(fullyConnected)
    result = filter.Execute(image)
    log.info('binaryFillHole() finish')
    return result

def boundingBox(image):
    log.info('boundingBox() start')
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(image, image)
    result = {}
    for label in filter.GetLabels():
        result[label] = filter.GetBoundingBox(label)
    log.info('boundingBox() finish')
    return result

def crop(image, lower, upper):
    log.info('crop() start')
    filter = sitk.CropImageFilter()
    filter.SetLowerBoundaryCropSize(lower)
    filter.SetUpperBoundaryCropSize(upper)
    result = filter.Execute(image)
    log.info('crop() finish')
    return result

def extract(image, index, size):
    log.info('extract() start')
    filter = sitk.ExtractImageFilter()
    filter.SetDirectionCollapseToStrategy(sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOSUBMATRIX)
    filter.SetIndex(index)
    filter.SetSize(size)
    result = filter.Execute(image)
    log.info('extract() finish')
    return result

def grayscaleClose(image, type=sitk.sitkCross, radius=[1, 1, 1]):
    log.info('grayscaleClose() start')
    filter = sitk.GrayscaleMorphologicalClosingImageFilter()
    filter.SetKernelType(type)
    filter.SetKernelRadius(radius)
    result = filter.Execute(image)
    log.info('grayscaleClose() finish')
    return result

def grayscaleDilate(image, type=sitk.sitkCross, radius=[1, 1, 1]):
    log.info('grayscaleDilate() start')
    filter = sitk.GrayscaleDilateImageFilter()
    filter.SetKernelType(type)
    filter.SetKernelRadius(radius)
    result = filter.Execute(image)
    log.info('grayscaleDilate() finish')
    return result

def grayscaleOpen(image, type=sitk.sitkCross, radius=[1, 1, 1]):
    log.info('grayscaleOpen() start')
    filter = sitk.GrayscaleMorphologicalOpeningImageFilter()
    filter.SetKernelType(type)
    filter.SetKernelRadius(radius)
    result = filter.Execute(image)
    log.info('grayscaleOpen() finish')
    return result

def normalize(image, mask, mean_value=0.0, std_value=1.0, label=1):
    log.info('normalize() start')
    stat = statistics(image, mask)
    norm_offset = -stat[label]['Mean']
    norm_scale = 1.0 / stat[label]['StdDev']
    log.info('  shift={:.3f} scale={:.3f}'.format(norm_offset, norm_scale))
    result = shiftScale(image, shift=norm_offset, scale=norm_scale)
    result_array = sitk.GetArrayFromImage(result)
    result_array[numpy.isnan(result_array)] = 0.0
    result = sitk.GetImageFromArray(result_array)
    result.CopyInformation(image)
    #stat = statistics(result, mask)
    #log.info('  mean={:.3f} sigma={:.3f}'.format(stat[label]['Mean'], stat[label]['StdDev']))
    log.info('normalize() finish')
    return result

def otsuThreshold(image, inside=0, outside=1, mask=1):
    log.info('otsuThreshold() start')
    filter = sitk.OtsuThresholdImageFilter()
    filter.SetInsideValue(inside)
    filter.SetOutsideValue(outside)
    filter.SetMaskValue(mask)
    log.info('  inside = {:d}, outside = {:d}, mask = {:d}'.format(filter.GetInsideValue(), filter.GetOutsideValue(), filter.GetMaskValue()))
    result = filter.Execute(image)
    log.info('  threshold = {:.1f}'.format(filter.GetThreshold()))
    log.info('otsuThreshold() finish')
    return result

def pad(image, lower, upper, value=0):
    log.info('pad() start')
    filter = sitk.ConstantPadImageFilter()
    filter.SetPadLowerBound(lower)
    filter.SetPadUpperBound(upper)
    filter.SetConstant(value)
    result = filter.Execute(image)
    log.info('pad() finish')
    return result

def relabelComponent(image, min_size):
    log.info('relabelComponent() start')
    filter = sitk.RelabelComponentImageFilter()
    filter.SetMinimumObjectSize(min_size)
    #filter.SetSortByObjectSize()
    result = filter.Execute(image)
    log.info('relabelComponent() finish')
    return result

def resample(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    log.info('resample() start')
    original_spacing = itk_image.GetSpacing()
    if out_spacing[0] == original_spacing[0] and out_spacing[1] == original_spacing[1] and out_spacing[2] == original_spacing[2]:
        result = itk_image
    else:
        original_size = itk_image.GetSize()
        out_size = [
            int(numpy.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(numpy.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(numpy.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
        ]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
        result = resample.Execute(itk_image)
    log.info('resample() finish')
    return result

def shiftScale(image, shift=0.0, scale=1.0):
    log.info('shiftScale() start')
    filter = sitk.ShiftScaleImageFilter()
    filter.SetShift(shift)
    filter.SetScale(scale)
    result = filter.Execute(image)
    log.info('shiftScale() finish')
    return result

def smooth(image, numIter=5, timeStep=0.125):
    log.info('smooth() start')
    filter = sitk.CurvatureFlowImageFilter()
    filter.SetNumberOfIterations(5)
    filter.SetTimeStep(0.125)
    result = filter.Execute(image)
    log.info('smooth() finish')
    return result

def statistics(image, mask):
    log.info('statistics() start')
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(image, mask)
    result = {}
    for label in filter.GetLabels():
        r = {}
        r['Min'] = filter.GetMinimum(label)
        r['Max'] = filter.GetMaximum(label)
        r['Median'] = filter.GetMedian(label)
        r['Mean'] = filter.GetMean(label)
        r['StdDev'] = filter.GetSigma(label)
        r['Var'] = filter.GetVariance(label)
        result[label] = r
    log.info('statistics() finish')
    return result

def subtract(image1, image2):
    filter = sitk.SubtractImageFilter()
    result = filter.Execute(image1, image2)
    return result

def threshold(image, lower, upper):
    log.info('threshold() start')
    filter = sitk.ThresholdImageFilter()
    filter.SetLower(lower)
    filter.SetUpper(upper)
    result = filter.Execute(image)
    log.info('threshold() finish')
    return result
