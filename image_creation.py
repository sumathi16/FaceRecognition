# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import sys
import argparse
import os

from matplotlib import pyplot

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    input_dir =  os.path.expanduser(args.input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in os.listdir(input_dir):
        dir_name = os.path.join(output_dir,"".join(name.split('.')[:-1]))
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # load the image
        img = load_img(os.path.join(input_dir,name))
        save_img(os.path.join(dir_name,name),img)    
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
            
        for i in range(100):
            datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            image = it.next()[0].astype('uint8')
            save_img(os.path.join(dir_name,"".join(name.split('.')[:-1])+"_z"+str(i)+".jpg"),image)
        for i in range(100):
            datagen = ImageDataGenerator(brightness_range=(0.0,1.0))
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            image = it.next()[0].astype('uint8')
            save_img(os.path.join(dir_name,"".join(name.split('.')[:-1])+"_b"+str(i)+".jpg"),image)      
        for i in range(100): 
            datagen = ImageDataGenerator(rotation_range=30)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            image = it.next()[0].astype('uint8')
            save_img(os.path.join(dir_name,"".join(name.split('.')[:-1])+str(i)+".jpg"),image)
        for i in range(100): 
            datagen = ImageDataGenerator(rotation_range=30,zoom_range=[0.5,1.0],brightness_range=(0.0,1.0))
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            image = it.next()[0].astype('uint8')
            save_img(os.path.join(dir_name,"".join(name.split('.')[:-1])+"all"+str(i)+".jpg"),image)         


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
