import csv
import os

from io import StringIO
import os.path

import numpy as np
from models.LocalModel import VFLPassiveModel
from models.ServerModel import VFLActiveModelWithOneLayer,QuanVFLActiveModelWithOneLayer


#from data_util.data_loader import TwoPartyDataLoader
def parse_header_of_csv(headline):
    # Isolate the headline columns:
    #    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:', '')
        pass

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',')  # ,skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return (X, Y, M, timestamps)


'''
Interpret the feature names to figure out for each feature what is the sensor it was extracted from.
'''


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi, feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'Aud'#AP
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('Tdiscrete'):
            feat_sensor_names[fi] = 'TS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass

    return feat_sensor_names


'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''


def read_user_data(uuid):
    user_data_file = os.path.join('', '%s' % uuid)

    # Read the entire csv file of the user:
    with open(user_data_file, 'r') as fid:
        csv_headline = fid.readline().strip()
        csv_body = fid.read()
        pass

    (feature_names, label_names) = parse_header_of_csv(csv_headline)
    print(len(feature_names))
    feat_sensor_names = get_sensor_names_from_features(feature_names)
    print(feat_sensor_names)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_body, n_features)

    return (X, Y, M, timestamps, feature_names, feat_sensor_names, label_names)


def validate_column_names_are_consistent(old_column_names, new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")

    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci, old_column_names[ci], new_column_names[ci]))
        pass
    return


def read_multiple_users_data(uuids):
    feature_names = None
    feat_sensor_names = None
    label_names = None
    X_parts = []
    Y_parts = []
    M_parts = []
    timestamps_parts = []
    uuid_inds_parts = []
    for (ui, uuid) in enumerate(uuids):
        (X_i, Y_i, M_i, timestamps_i, feature_names_i, feat_sensor_names_i, label_names_i)
        # Make sure the feature names are consistent among all users:
        if feature_names is None:
            feature_names = feature_names_i
            feat_sensor_names = feat_sensor_names_i
            pass
        else:
            validate_column_names_are_consistent(feature_names, feature_names_i)
            pass
        # Make sure the label names are consistent among all users:
        if label_names is None:
            label_names = label_names_i
            pass
        else:
            validate_column_names_are_consistent(label_names, label_names_i)
            pass
        # Accumulate this user's data:
        X_parts.append(X_i)
        Y_parts.append(Y_i)
        M_parts.append(M_i)
        timestamps_parts.append(timestamps_i)
        uuid_inds_parts.append(ui * np.ones(len(timestamps_i)))
        pass

    # Combine all the users' data:
    X = np.concatenate(tuple(X_parts), axis=0)
    Y = np.concatenate(tuple(Y_parts), axis=0)
    M = np.concatenate(tuple(M_parts), axis=0)
    timestamps = np.concatenate(tuple(timestamps_parts), axis=0)
    uuid_inds = np.concatenate(tuple(uuid_inds_parts), axis=0)

    return (X, Y, M, uuid_inds, timestamps, feature_names, feat_sensor_names, label_names)


def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)

        use_feature = np.logical_or(use_feature, is_from_sensor)
        pass
    X = X[:, use_feature]
    #print(use_feature)
    return X


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0)
    std_vec = np.nanstd(X_train, axis=0)
    return (mean_vec, std_vec)


def standardize_features(X, mean_vec, std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1, -1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1))
    X_standard = X_centralized / normalizers
    return X_standard


def load_sensor_data(X_train, Y_train, M_train, feat_sensor_names, label_names, sensors_to_use, target_label,delete_nan):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train, feat_sensor_names, sensors_to_use)
    print(X_train.shape)
    print("== Projected the features to %d features from the sensors: %s" % (
    X_train.shape[1], ', '.join(sensors_to_use)))

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train, mean_vec, std_vec)

    # The single target label:
    label_ind = label_names.index(target_label)
    y = Y_train[:, label_ind]
    missing_label = M_train[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label, :]
    y = y[existing_label]

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    if delete_nan:
        index_good = ~np.isnan(X_train).any(axis=1)
        # a[index_good, :]
        y = y[index_good]
        X_train = X_train[index_good, :]
    else:
        X_train[np.isnan(X_train)] = 0.
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), target_label, sum(y), sum(np.logical_not(y))))

    yy=y.astype(float)
    return X_train, yy


def test_model(X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use'])
    print("== Projected the features to %d features from the sensors: %s" % (
    X_test.shape[1], ', '.join(model['sensors_to_use'])))

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec'])

    # The single target label:
    label_ind = label_names.index(model['target_label'])
    y = Y_test[:, label_ind]
    missing_label = M_test[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label, :]
    y = y[existing_label]
    timestamps = timestamps[existing_label]

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.
   
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), model['target_label'], sum(y), sum(np.logical_not(y))))

    # Preform the prediction:

    y_pred = model['mlp'].predict(X_test)
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y)

    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred, y))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y))

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)

    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.

    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp + fp)

    print("-" * 10)
    print('Accuracy*:         %.2f' % accuracy)
    print('Sensitivity (TPR): %.2f' % sensitivity)
    print('Specificity (TNR): %.2f' % specificity)
    print('Balanced accuracy: %.2f' % balanced_accuracy)
    print('Precision**:       %.2f' % precision)
    print("-" * 10)

    print(
        '* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print(
        '** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')

   
    return


def create_folder(args):
    if not os.path.exists(args.path_predix):
        os.makedirs(args.path_predix)
    print("save to", args.path_predix)

    SavedPaths = []
   
    for i in range(args.num_text_clients):
        _dir = os.path.join(args.path_predix,
                            'sensor_{}_{}'.format(args.text_feature_div[i], args.text_feature_div[i + 1]))
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        SavedPaths.append(_dir)

    _dir = os.path.join(args.path_predix, 'server')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    server_savedpath = _dir

    return SavedPaths, server_savedpath


def create_models(args, quan_server=False):
    if quan_server == False:
        LocalModels = []
        GradientsRes = []
        for i in range(args.num_clients):
            local_model = VFLPassiveModel(emb_dim=args.emb_dim)
            LocalModels.append(local_model)
            GradientsRes.append(None)
        active_model = VFLActiveModelWithOneLayer(emb_dim=args.emb_dim, class_num=args.num_class)
        return LocalModels, GradientsRes, active_model
    else:
        local_model = VFLPassiveModel(emb_dim=args.emb_dim)
        active_model = QuanVFLActiveModelWithOneLayer(emb_dim=args.emb_dim, class_num=args.num_class)
        return local_model, active_model


def load_models(args, quan_server=False):
    if quan_server == False:
        LocalModels = []
        for i in range(args.num_clients):
            local_model = VFLPassiveModel(emb_dim=args.emb_dim)
            local_model.built = True
            local_model.load_weights(os.path.join(args.local_paths[i], args.ckpt_name))
            LocalModels.append(local_model)

        active_model = VFLActiveModelWithOneLayer(emb_dim=args.emb_dim, class_num=args.num_class)
        active_model.built = True
        active_model.load_weights(os.path.join(args.sever_path, args.ckpt_name))
        return LocalModels, active_model
    else:
        local_model = VFLPassiveModel(emb_dim=args.emb_dim)
        local_model.built = True
        local_model.load_weights(os.path.join(args.local_paths[0], args.ckpt_name))
        active_model = QuanVFLActiveModelWithOneLayer(emb_dim=args.emb_dim, class_num=args.num_class)
        active_model.built = True
        active_model.load_weights(os.path.join(args.sever_path, args.ckpt_name))
        return local_model, active_model

def get_local_outputs(num_text_clients, texts, text_feature_div, LocalModels):
    LocalOutputs =[]
    for i in range(num_text_clients):
        text_div = texts[:,text_feature_div[i]:text_feature_div[i+1]]
        LocalOutputs.append(LocalModels[i](text_div))

    return LocalOutputs
