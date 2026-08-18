import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import dataset_definition as dtdef
import matplotlib.pyplot as plt
import os
import pandas as pd


class RepresentativeDataGenerator():
    def __init__(self, dataset, subject, session, compression, bit):
        self.hdim = dataset.sensors_dim[0]
        self.vdim = dataset.sensors_dim[1]
        self.name = dataset.name
        self.bit = bit
        self.dataset_path = "dataset/train/%s/%s/%s_%s_%s_%sbits.npz"%(self.name, compression, subject, session, compression, bit)
    def yield_representative_data(self):
        data_examples = np.random.randint(0, 256, size=(100, self.hdim, self.vdim, 1))
        data_examples = data_examples.astype('float32')
        for data in data_examples:
            yield [data.reshape(1,self.hdim, self.vdim,1)]  
    def yield_original_data(self):
        indices = np.random.choice(np.arange(36000,48001), size=100, replace=False)
        with np.load(self.dataset_path) as data:
            test_examples = data['data'][indices,:]
        line_dim = dataset.sensors_dim[0]
        column_dim = dataset.sensors_dim[1]
        data_examples = test_examples.astype('float32').reshape(-1,line_dim,column_dim,1)
        for data in data_examples:
            yield [data.reshape(1,self.hdim, self.vdim,1)]  

def convert_model_to_tflite(folder_path, model_name, model_type, dataset, subject, session, compression, bit, tflite_model_name):
    '''
    Convert a keras model to a tflite model

    @param folder_path the path to the folder containing the model
    @param model_name the name of the model to convert
    @param model_type the type of model to convert (normal, tuned, extractor)
    @param tflite_model_name the name of the tflite model to save

    @return the converted tflite model
    '''

    data_generator = RepresentativeDataGenerator(dataset, subject, session, compression, bit)
    print("Converting")
    model_path = '%s/%s.h5'%(folder_path, model_name)
    saved_keras_model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(saved_keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.representative_dataset = data_generator.yield_original_data
    tflite_model = converter.convert()

    with open('offdevice/model/tflite_eval/%s/%s.tflite'%(model_type, tflite_model_name), 'wb') as f:
        f.write(tflite_model)

def evaluate_raw_model(dataset, model_path, dataset_path, model_name):
    BATCH_SIZE = 64

    model_path = '%s/%s.h5'%(model_path, model_name)
    model = tf.keras.models.load_model(model_path)

    with np.load(dataset_path) as data:
        test_examples = data['data']
        test_labels = data['label']
    
    line_dim = dataset.sensors_dim[0]
    column_dim = dataset.sensors_dim[1]
    test_examples = test_examples.astype('float32').reshape(-1,line_dim,column_dim,1)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_examples)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    logits = model.predict(test_dataset)

    prediction = np.argmax(logits, axis=1)
    truth = test_labels

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    result = float(keras_accuracy.result().numpy())
    print("Raw model accuracy: {:.3%}".format(result))
    return result


def evaluate_tflite_model(dataset, model_path, dataset_path, tflite_model_name):
    with np.load(dataset_path) as data:
        test_examples = data['data']
        test_labels = data['label']

    line_dim = dataset.sensors_dim[0]
    column_dim = dataset.sensors_dim[1]
    test_examples = test_examples.astype(np.uint8).reshape(-1,line_dim,column_dim,1)
    data = test_examples[:,:,:,:]
    
    model_path = '%s/%s.tflite'%(model_path, tflite_model_name)
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    prediction = []
    for i in range(len(data)):
        curr_data = np.expand_dims(data[i], axis=0)
        input_tensor= tf.convert_to_tensor(curr_data, np.uint8)
        interpreter.set_tensor(input_details['index'], curr_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        prediction.append(np.argmax(output))

    tflite_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy(prediction, test_labels)
    result = float(tflite_accuracy.result().numpy())
    print("Quant TF Lite accuracy: {:.3%}".format(result))
    return result


if __name__ == '__main__':
    # convert the model

    #dataset = dtdef.CapgmyoDataset()
    dataset = dtdef.EmagerDataset()
    # bits = [1,2,3,4,5,6,7,8]

    # compression_methods = ["minmax", "msb", "smart", "root"]

    # folder_path = "offdevice/model"
    # model_tuning = "normal"
    # model_type = "cnn"
    # for sub in range(12):
    #     for sess in range(2):
    #         for compression_mode in compression_methods:
    #             for bit in bits:
    #                 subject = "0" + str(sub) if sub < 10 else str(sub)
    #                 session = str(sess+1)
    #                 model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_type, subject, session, compression_mode, bit)
    #                 tflite_model_name = model_name
    #                 convert_model_to_tflite(folder_path, model_name, model_tuning, dataset, subject, session, compression_mode, bit, tflite_model_name)

    # folder_path = "offdevice/model/tuned"
    # model_tuning = "tuned"
    # model_type = "cnn"
    # #for sub in range(12):
    # for sub in range(1,11):
    #     subject = "0" + str(sub) if sub < 10 else str(sub)
    #     for sess in range(2):
    #         session = str(sess+1)
    #         for tuning in range(5):
    #             fine_tuning_range = range(tuning*2, tuning*2+2)
    #             for compression_mode in compression_methods:
    #                 for bit in bits:
    #                     #dataset_path = '/dataset/train/%s/'%(compression_mode)
    #                     model_name = "%s_%s_%s_%s_%s_%sbits_tuned_%s_%s"%(dataset.name, model_type, subject, session, compression_mode, bit, fine_tuning_range[0], fine_tuning_range[-1])
    #                     tflite_model_name = model_name
    #                     convert_model_to_tflite(folder_path, model_name, model_tuning, dataset, bit, tflite_model_name)

    subjects = ["00","01","02","03","04","05","06","07","08","09","10","11"]
    sessions = ["1", "2"]
    folder_path = "offdevice/model/"
    tflite_path = "offdevice/model/tflite_eval/normal/"

    difference_list = []
    for subject in subjects:
        for session in sessions:
            model_name = "emager_cnn_%s_%s_root_8bits"%(subject,session)
            dataset_path = "dataset/train/emager/root/%s_%s_root_8bits.npz"%(subject,session)

            raw_result = evaluate_raw_model(dataset, folder_path, dataset_path, model_name)
            tflite_model = evaluate_tflite_model(dataset, tflite_path, dataset_path, model_name)
            difference = raw_result - tflite_model
            print(difference)
            difference_list.append(difference*100)
    mean = np.mean(difference_list)
    std = np.std(difference_list)
    print(difference_list)

    print("The average accuracy difference is %s +- %s"%(mean,std))