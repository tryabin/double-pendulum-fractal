import datetime
import os


def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()


def create_directory():
    # Create a new folder to store the images.
    currentDateTimeString = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')
    directory = 'double pendulum fractal images - ' + currentDateTimeString
    path = os.path.join(os.getcwd(), directory)
    os.mkdir(path)
    return directory


def save_image_to_file(directory, image):
    currentDateTimeString = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')
    filename = 'double pendulum fractal - ' + currentDateTimeString + '.png'
    image.save(os.path.join(directory, filename))


def create_video_from_images(imagesDirectory, videoName, fps):
    originalWorkingDirectory = os.getcwd()
    os.chdir(imagesDirectory)
    os.popen('ffmpeg -y -framerate ' + str(fps) + ' -i %04d.png -c:v libx264 -r ' + str(fps) + ' -crf 0 -tune fastdecode "' + videoName + '.mp4"')
    os.chdir(originalWorkingDirectory)