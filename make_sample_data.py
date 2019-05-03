import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import argparse
import json
import random
import codecs

from model import resnet50

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--p', type=int, default=7)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--input-size', type=tuple, default=(32, 32))
    parser.add_argument('--load-pretrained', type=bool, default=True)

    args = parser.parse_args()

    return args


def batch_iterator(image_list, batch_size=100, shape=(32, 32)):
    random.shuffle(image_list)
    while len(image_list) != 0:
        batch_keys = image_list[:batch_size]

        images = []
        images_path = []

        for key in batch_keys:
            image = cv2.imread(key)
            image = cv2.resize(image, dsize=shape)

            images.append(image)
            images_path.append(key)

        images = np.array(images)
        images = np.reshape(images, newshape=[-1, 3, 32, 32])
        images_path = np.array(images_path)
        yield images, images_path

        del image_list[:batch_size]


def data_sampling(model, args):
    sampling_dictionary = {}
    model.eval()
    maxk = max((args.p, ))
    with torch.no_grad():
        classes = [
            "beaver", "dolphin", "otter", "seal", "whale",
            "aquarium fish", "flatfish", "ray", "shark", "trout",
            "orchids", "poppies", "roses", "sunflowers", "tulips",
            "bottles", "bowls", "cans", "cups", "plates",
            "apples", "mushrooms", "oranges", "pears", "sweet peppers",
            "clock", "computer keyboard", "lamp", "telephone", "television",
            "bed", "chair", "couch", "table", "wardrobe",
            "bee", "beetle", "butterfly", "caterpillar", "cockroach",
            "bear", "leopard", "lion", "tiger", "wolf",
            "bridge", "castle", "house", "road", "skyscraper",
            "cloud", "forest", "mountain", "plain", "sea",
            "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
            "fox", "porcupine", "possum", "raccoon", "skunk",
            "crab", "lobster", "snail", "spider", "worm",
            "baby", "boy", "girl", "man", "woman",
            "crocodile", "dinosaur", "lizard", "snake", "turtle",
            "hamster", "mouse", "rabbit", "shrew", "squirrel",
            "maple", "oak", "palm", "pine", "willow",
            "bicycle", "bus", "motorcycle", "pickup truck", "train",
            "lawn-mower", "rocket", "streetcar", "tank", "tractor"
        ]
        for each_class in classes:
            print("class name: ", each_class)
            download_path = "C:/Users/myeongjun/github/AutoCrawler/download/"
            if os.path.isdir(download_path + each_class):
                image_path = download_path + each_class + "/*.jpg"
                all_image_path = glob.glob(image_path)
                print("image data count: ", len(all_image_path))
                for batch_image, batch_image_path in batch_iterator(all_image_path, args.batch_size, args.input_size):
                    batch_image = torch.cuda.FloatTensor(batch_image)

                    output = model(batch_image)
                    softmax_output = F.softmax(output, dim=-1)
                    _, top_p = softmax_output.topk(maxk, 1, True, True)
                    # print(top_p.t())

                    # make sampling dictionary
                    for top in top_p.t():
                        for idx, i in enumerate(top):
                            num = i.data.cpu().numpy()
                            value = float(softmax_output[idx][i].data.cpu().numpy())
                            if str(num) in sampling_dictionary:
                                sampling_dictionary[str(num)].append([batch_image_path[idx], value])
                            else:
                                sampling_dictionary[str(num)] = [[batch_image_path[idx], value]]
            else:
                print("Can't find directory")
    print("Saving.. sampling_dict")
    j = json.dumps(sampling_dictionary)
    with open("sampling_dict.json", "w") as f:
        f.write(j)


def select_top_k(k=1000):
    sampled_image_dict = {}
    sampled_image_dict["all"] = []
    with codecs.open("./sampling_dict.json", "r", encoding="utf-8", errors="ignore") as f:
        load_data = json.load(f)

        for key in load_data.keys():
            print("label: ", key)
            all_items = load_data[key]
            all_items.sort(key=lambda x: x[1], reverse=True)
            all_items = np.array(all_items)
            print("each label item count: ", len(all_items))
            for index in range(0, k):
                sampled_image_dict["all"].append([all_items[index][0], int(key)])

    print("Saving.. selected image json")
    j = json.dumps(sampled_image_dict)
    with open("selected_image.json", "w") as f:
        f.write(j)


def main(args):
    if args.load_pretrained:
        model = resnet50().to(device)
        filename = "Best_model_"
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

        data_sampling(model, args)
        select_top_k(args.k)
    else:
        assert args.load_pretrained == True, "You must have the weights of the pretrained model."


if __name__ == "__main__":
    args = get_args()
    main(args)
