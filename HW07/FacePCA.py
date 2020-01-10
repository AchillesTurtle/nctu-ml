from dataprocess import loadFaces
import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA

def plotSampleFace(imageArray,imageShape=None,sampleShape=(5,5),randomSelect=True,imageIndex=None,scaling=False):
    sampleHeight,sampleWidth=sampleShape
    if imageArray.ndim==2:
        imageHeight,imageWidth=imageShape
        imageCount=imageArray[0]
    elif imageArray.ndim==3:
        imageCount,imageHeight,imageWidth=imageArray.shape
    else:
        print("Illegal image array")
        return
    mergedFaces=np.zeros((sampleHeight*imageHeight,sampleWidth*imageWidth))
    if randomSelect:
        selectedImageIndex=np.random.randint(imageCount,size=sampleShape)
    else:
        if imageIndex is None:
            selectedImageIndex=np.arange(sampleHeight*sampleWidth).reshape(sampleShape)
        else:
            selectedImageIndex=imageIndex

    if scaling is True:
        for imageIndex in selectedImageIndex:
            if imageArray.ndim==2:
                imageArray[imageIndex, :] *= 255.0/imageArray[imageIndex, :].max()
            elif imageArray.ndim==3:
                imageArray[imageIndex, :,:] *= 255.0 / imageArray[imageIndex, :,:].max()
    for sampleRow in range(sampleHeight):
        if imageArray.ndim==2:
            mergedImageRow = mergeImageToRow(imageArray[selectedImageIndex[sampleRow, :],:], (imageHeight, imageWidth))
        elif imageArray.ndim==3:
            mergedImageRow=mergeImageToRow(imageArray[selectedImageIndex[sampleRow,:],:,:],(imageHeight,imageWidth))
        mergedFaces[sampleRow*imageHeight:(sampleRow+1)*imageHeight,:]=mergedImageRow

    plt.imshow(mergedFaces)
    plt.colorbar()
    plt.show()
    return mergedFaces

def mergeImageToRow(imageArray,imageShape=None):
    #print(imageShape)
    #plt.imshow(imageArray[0,:,:])
    #plt.show()
    if imageArray.ndim==2:
        imageHeight,imageWidth=imageShape
        imageCount=imageArray.shape[0]
    elif imageArray.ndim==3:
        imageCount,imageHeight,imageWidth=imageArray.shape
    else:
        print("Illegal image array")
        return
    mergedImage=np.zeros((imageHeight,imageWidth*imageCount))
    for imageRow in range(imageHeight):
        for imageIndex in range(imageCount):
            if imageArray.ndim==2:
                mergedImage[imageRow, imageIndex*imageWidth:(imageIndex + 1) * imageWidth] = imageArray[imageIndex,imageRow * imageWidth:(imageRow + 1) * imageWidth]
            elif imageArray.ndim==3:
                mergedImage[imageRow, imageIndex*imageWidth:(imageIndex + 1) * imageWidth] = imageArray[imageIndex, imageRow, :]
    return mergedImage


if __name__=="__main__":
    #problem 7 eigenfaces
    faceArray=loadFaces()
    #plotSampleFace(faceArray)
    downSampledArray=faceArray[:,::3,::3]
    print(downSampledArray.shape)
    sampleShape=(5,5)
    selectedImageIndex = np.random.randint(400, size=sampleShape)

    facePCA=PCA(downSampledArray)
    facePCA.project(projectDimension=25)
    plotSampleFace(downSampledArray,imageShape=(38,31),imageIndex=selectedImageIndex,randomSelect=False)
    plotSampleFace(facePCA.eigenVector.T,imageShape=(38,31),randomSelect=False,scaling=False)
    print(facePCA.projectEigenVector.T.shape)
    plotSampleFace(facePCA.projectData[selectedImageIndex.flatten(),:]@(facePCA.projectEigenVector.T),imageShape=(38,31),randomSelect=False)