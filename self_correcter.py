import numpy as np
import operator
from minibatch import *

class Correcter(object):
    def __init__(self, size_of_data, num_of_classes, queue_size):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.queue_size = queue_size


        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            self.all_predictions[i] = np.zeros(queue_size, dtype=int)

        self.softmax_record = {}

        for i in range(size_of_data):
            self.softmax_record[i] = []

        self.certainty_array = {}

        for i in range(size_of_data):
            self.certainty_array[i] = 1

        self.max_certainty = -np.log(1.0/float(self.queue_size))
        self.clean_key = []
        # Corrected label map
        self.corrected_labels = {}
        for i in range(size_of_data):
            self.corrected_labels[i] = -1

        self.update_counters = np.zeros(size_of_data, dtype=int)


    def async_update_prediction_matrix(self, ids, softmax_matrix,loss_array):
        for i in range(len(ids)):
            id = ids[i].item()
            predicted_label = np.argmax(softmax_matrix[i])
            cur_index = self.update_counters[id] % self.queue_size
            self.all_predictions[id][cur_index] = predicted_label
            self.softmax_record[id] = loss_array[i]

            self.update_counters[id] += 1

    # compute cross-entropy
    def cross_entropy(self,Y, P):
        Y = np.float_(Y)
        P = np.float_(P)
        return -np.sum(Y * np.log(P))

    def separate_clean_and_unclean_keys(self,noise_rate):
        loss_map = {}
        self.clean_key = []
        num_clean_instances = int(np.ceil(float(self.size_of_data) * (1.0 - noise_rate)))
        for i in range(self.size_of_data):
            loss_map[i] = self.softmax_record[i]

        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

        index = 0
        for key in loss_map.keys():
            if index < num_clean_instances:
                # clean_batch.append(key, image_map[key], label_map[key])
                self.clean_key.append(key)
            index += 1
        loss_map.clear()


    def separate_clean_and_unclean_samples(self, ids, images, labels):
        clean_batch = MiniBatch()
        unclean_batch = MiniBatch()

        for i in range(len(ids)):
            if ids[i] in self.clean_key:
                clean_batch.append(ids[i], images[i], labels[i])
            else:
                unclean_batch.append(ids[i], images[i], labels[i])

        return clean_batch, unclean_batch



    def get_corrected_samples(self, ids, images):

        # check correctability for each sample
        accumulator = {}
        for i in range(len(ids)):
            id = ids[i]
            image = images[i]

            predictions = self.all_predictions[id]

            for prediction in predictions:
                if prediction not in accumulator:
                    accumulator[prediction] = 1
                else:
                    accumulator[prediction] = accumulator[prediction] + 1

            p_dict = np.zeros(self.num_of_classes, dtype=float)
            for key, value in accumulator.items():
                p_dict[key] = float(value) / float(self.queue_size)

            # based on entropy
            negative_entropy = 0.0
            for i in range(len(p_dict)):
                if p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[i] * np.log(p_dict[i])
            certainty = - negative_entropy / self.max_certainty
            self.certainty_array[id] = certainty

        # return corrected_batch

    def merge_clean_and_corrected_samples(self, clean_batch):
        final_batch = MiniBatch()

        return final_batch.ids, final_batch.images, final_batch.labels


    def patch_clean_with_corrected_sample_batch(self, ids, images, labels):
        clean_batch, unclean_batch = self.separate_clean_and_unclean_samples(ids, images, labels)
        self.get_corrected_samples(ids, images)
        return clean_batch.ids, clean_batch.images, clean_batch.labels

    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)

    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.queue_size, dtype=int)
