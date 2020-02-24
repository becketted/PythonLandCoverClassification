import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from scipy.io import loadmat
import scipy.stats
import seaborn as sns

# i/o functions
def loadMat(path):
    # load .mat file
    matFile = loadmat(path)
    # isolate ground truth data and transpose (file is rotated)
    groundTruth = np.transpose(matFile['labelled_ground_truth'])
    return groundTruth

def loadBMP(path):
    # load BMP image
    BMP = Image.open(path)
    return BMP

def saveSamples(samples, filename):
    # open the file
    f = open(filename, "w")
    # for each class
    for classNo in range(4):
        # for each sample
        for sampleNo in range(20):
            # write a new line containing the sample coordinates
            f.write(str(samples[classNo][sampleNo]) + "\n")
    # close the file
    f.close()

def loadSamples(filename):
    # create empty list of samples
    samples = [[0] * 20 for x in range(4)]

    # open the file
    f = open(filename, "r")
    # read each line
    lines = f.read().splitlines()
    # for each class
    for classNo in range(4):
        # for each sample
        for sampleNo in range(20):
            # insert the associated line into the correct position
            samples[classNo][sampleNo] = lines[sampleNo + (20 * classNo)]

    return samples


# sampling functions
def getSamples(GT):
    # get the 20 random samples for each class from the ground truth
    # create empty list of samples
    samples = [[0] * 20 for x in range(4)]
    # for each class
    for classNo in range(4):
        # keep track of how many have been collected
        sampleCounter = 0
        # whilst the samples for the class are not complete
        while sampleCounter < 20:
            # generate random coordinates within the ground truth
            randX = random.randint(0, GT.shape[0] - 1)
            randY = random.randint(0, GT.shape[1] - 1)
            # if the value at that pixel matches the desired class
            if GT[randX][randY] == classNo+1:
                # then take that coordinate, store it and increment the sample counter
                samples[classNo][sampleCounter] = (randX, randY)
                sampleCounter += 1
    return samples

def sampleImages(fe, le, r, g, b, nir, samples):
    # used to retrieve the pixel values for each image at the randomly sampled coordinates
    # create an empty sample holder
    trainingSamples = [[0] * 20 for x in range(4)]

    # for each class
    for classNo in range(4):
        # for each  sample
        for sampleNo in range(20):
            # obtain the pixel value for each image
            feVal = fe.getpixel(samples[classNo][sampleNo])
            leVal = le.getpixel(samples[classNo][sampleNo])
            rVal = r.getpixel(samples[classNo][sampleNo])
            gVal = g.getpixel(samples[classNo][sampleNo])
            bVal = b.getpixel(samples[classNo][sampleNo])
            nirVal = nir.getpixel(samples[classNo][sampleNo])
            # insert that into the samples alongside the coordinates for future reference
            trainingSamples[classNo][sampleNo] = [samples[classNo][sampleNo], feVal, leVal, rVal, gVal, bVal, nirVal]

    return trainingSamples

def scanImages(fe, le, r, g, b, nir):
    # used to obtain every pixel value for each image
    # create an empty image sized holder
    pixels = [[0] * fe.height for x in range(fe.width)]

    # iterate through the x axis
    for x in range(fe.width):
        # iterate through the y axis
        for y in range(fe.height):
            # insert each pixel value into the holder
            pixels[x][y] = [fe.getpixel((x,y)), le.getpixel((x, y)), r.getpixel((x, y)), g.getpixel((x, y)), b.getpixel((x, y)), nir.getpixel((x, y))]

    return pixels



# classification functions
def calcGauss(samples):
    # create empty containers
    xyRemoved = [[0] * 20 for x in range(4)]
    meanVectors = [[0] * 6 for x in range(4)]
    covMatrices = [0] * 4
    gaussModels = [0] * 4

    # remove the pixel coordinates
    for classNo in range(4):
        for sampleNo in range(20):
            xyRemoved[classNo][sampleNo] = samples[classNo][sampleNo][1:7]

    # for each class
    for classNo in range(4):
        # calculate the covariance matrix
        covMatrices[classNo] = np.cov(xyRemoved[classNo], rowvar=False)

        # calculate the mean vectors
        meanVectors[classNo] = list(np.mean(xyRemoved[classNo], axis=0))

        # create a gaussian model
        gaussModels[classNo] = scipy.stats.multivariate_normal(meanVectors[classNo], covMatrices[classNo])

    # return the gaussian models for each class
    return gaussModels

def maximumLikelihood(pixels, gaussModels):
    # create empty image
    classes = [[0] * 211 for x in range(356)]

    # for each pixel
    for x in range(356):
        for y in range(211):
            # reset probabilities
            classProbabilities = [0] * 4
            # for each class
            for classNo in range(4):
                # take natural log of the pdf
                classProbabilities[classNo] = gaussModels[classNo].logpdf(pixels[x][y])
            # set predicted class respectively
            classes[x][y] = classProbabilities.index(max(classProbabilities)) + 1

    return classes


# confusion matrix for performance evaluation
def confusionMatrix(truth, predicted):
    # create empty 4*4 matrix
    confusionMatrix = [[0] * 4 for x in range(4)]

    # check each pixel
    for x in range(len(truth)):
        for y in range(len(truth[0])):
            # increment respective section of matrix accordingly
            # if the classification is correct then the diagonal between the class will be incrememnted
            # otherwise, the false classification will be incremented
            confusionMatrix[truth[x][y] - 1][predicted[x][y] - 1] += 1

    return confusionMatrix


# visual aids
def showGroundTruth(GT, GP):
    # displays the registered and predicted ground truths side by side
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # registered ground truth data
    image1 = ax1.imshow(GT, cmap="plasma", origin="lower")
    ax1.set_title('Registered Ground Truth')
    ax1.axis("off")
    fig.colorbar(image1, ax=ax1, ticks=range(5))

    # predicted ground truth data
    image2 = ax2.imshow(GP, cmap="plasma", origin="lower")
    ax2.set_title('Computed Ground Truth')
    ax2.axis("off")
    fig.colorbar(image2, ax=ax2, ticks=range(5))

    # display
    plt.show()

def displayConfusionMatrix(cm):
    # seaborn heatmap
    ax = sns.heatmap(cm, annot=True, annot_kws={"size": 15}, fmt="g", xticklabels=["Building", "Vegetation", "Car", "Ground"], yticklabels=["Building", "Vegetation", "Car", "Ground"])
    # ticks along axis
    plt.yticks(np.arange(4)+0.5, rotation=0, va="center")
    ax.xaxis.set_ticks_position('top')
    # axis labels
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted Ground")
    ax.xaxis.set_label_position('top')
    # display
    plt.show()


def completionStats(matrix):
    # some statistics to display the results of the classification
    categories = ["Building", "Vegetation", "Car", "Ground"]

    # correctly classified pixels overall
    sumCorrect = 0
    for diagonal in range(4):
        sumCorrect += matrix[diagonal][diagonal]

    print(str(sumCorrect) + str(" correctly classified pixels"))
    overallAccuracy = 100 / (75116 / sumCorrect)
    print(str(round(overallAccuracy, 1)) + "% accuracy overall")

    # accuracy for each class
    classTotals = list(np.sum(matrix, axis=0))
    for classNo in range(4):
        classAccuracy = 100 / (classTotals[classNo] / matrix[classNo][classNo])
        print(str(round(classAccuracy, 1)) + "% accuracy for the " + str(categories[classNo]) + " class")



# Main
def main():
    # load in the ground truth
    GT = loadMat('FILEPATH.mat')

    # obtain the 20 random sampling locations for each class
    samples = getSamples(GT)

    # i/o functionality
    #saveSamples(samples, "test")
    #samples = loadSamples("test")

    # load all six images
    FE = loadBMP('FILEPATH.bmp')
    LE = loadBMP('FILEPATH.bmp')
    R = loadBMP('FILEPATH.bmp')
    G = loadBMP('FILEPATH.bmp')
    B = loadBMP('FILEPATH.bmp')
    NIR = loadBMP('FILEPATH.bmp')

    # acquire the pixel values for each image using the sampled coordinates
    samples = sampleImages(FE, LE, R, G, B, NIR, samples)

    # use the previously gathered data to calculate mean vectors, covariance matrices and a gaussian model for each class
    GM = calcGauss(samples)

    # scan every pixel of all images to retrieve pixel data
    pixels = scanImages(FE, LE, R, G, B, NIR)

    # pass those to the maximum likelihood classifier to determine a class
    classes = maximumLikelihood(pixels, GM)

    # display the results
    showGroundTruth(GT, classes)

    # calculate a confusion matrix for performance evaluation
    cm = confusionMatrix(GT,classes)

    # display the confusion matrix
    displayConfusionMatrix(cm)

    # some stats to indicate performance
    completionStats(cm)


# Code to make the main function run properly
if __name__== "__main__":
    main()
