import os
import itertools
import random


class AWEData:

    def __init__(self, dataset_path):
        self.path = dataset_path

    def prepare_ann_files(self):
        # train data
        train_path = self.path + "/train/"
        train_subjects = sorted(filter(lambda f: not f.startswith('.'), os.listdir(train_path)))
        train_paths = []
        train_y = []
        for train_subject in train_subjects:
            subject_path = os.path.join(train_path, train_subject + "/")
            subject_samples = sorted(filter(lambda f: not f.startswith('.'), os.listdir(subject_path)))
            subject_train_paths = ["".join(item) for item in list(itertools.product([subject_path],
                                   list(subject_samples)))]
            train_paths += subject_train_paths
            train_y += [str(int(train_subject))] * len(subject_samples)

        # permute the data randomly and write it to the file
        combined = list(zip(train_paths, train_y))
        random.shuffle(combined)
        train_paths[:], train_y[:] = zip(*combined)
        with open("X_train.txt", "w") as train_x_out:
            train_x_out.write("\r\n".join(train_paths))

        with open("y_train.txt", "w") as train_y_out:
            train_y_out.write("\r\n".join(train_y))

        print("Found", len(train_paths), "training samples!")

        # validation data
        val_path = self.path + "/val/"
        val_subjects = sorted(filter(lambda f: not f.startswith('.'), os.listdir(val_path)))
        val_paths = []
        val_y = []
        for val_subject in val_subjects:
            subject_path = os.path.join(val_path, val_subject + "/")
            subject_samples = sorted(filter(lambda f: not f.startswith('.'), os.listdir(subject_path)))
            subject_val_paths = ["".join(item) for item in
                                 list(itertools.product([subject_path], list(subject_samples)))]
            val_paths += subject_val_paths
            val_y += [str(int(val_subject))] * len(subject_samples)

        # permute the data randomly and write it to the file
        combined = list(zip(val_paths, val_y))
        random.shuffle(combined)
        val_paths[:], val_y[:] = zip(*combined)
        with open("X_test.txt", "w") as test_x_out:
            test_x_out.write("\r\n".join(val_paths))

        with open("y_test.txt", "w") as test_y_out:
            test_y_out.write("\r\n".join(val_y))

        print("Found", len(val_paths), "training samples!")

        # label map
        label_map = list(map(lambda x: str(int(x)), train_subjects))
        label_map = list(zip(label_map, label_map))
        with open("label_map.txt", "w") as label_map_out:
            label_map_out.write('\r\n'.join('%s,%s' % x for x in label_map))


if __name__ == '__main__':
    awe_data = AWEData("AWE_dataset")
    awe_data.prepare_ann_files()
