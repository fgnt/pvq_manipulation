from __future__ import annotations

import parselmouth as pm
import warnings
import numpy as np
import pandas as pd
from scipy.signal.windows import hann
from pathlib import Path
from sklearn.impute import SimpleImputer
import creapy
from creapy.feature_extraction.feature_extraction import _cpp, _h1_h2, _jitter, _shimmer, _f0mean, _zcr, _ste
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class Model:
    """The Model for creaky voice classification."""
    def __init__(self):
        self._config = creapy.utils.get_config()["MODEL"]
        self._X_train: pd.DataFrame
        self._y_train: pd.Series
        self._imputer: SimpleImputer
        self._features = self._config["FEATURES"]["for_classification"]
        self._fitted = False
        _clf = self._config["CLASSIFIER"]["clf"]
        self._clf = clfs[_clf](
            **self._config["CLASSIFIER"]["VALUES"][_clf.upper()]["kwargs"])

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Function to fit the model with training data.

        Args:
            X_train (pd.DataFrame): Features of training data.
            y_train (pd.Dataframe): Targets of training data (creak, no-creak).
        """
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.to_numpy()
        if self._config["PREPROCESSING"]["impute_at_fit"] is True:
            self._X_train, self._imputer = creapy.preprocessing.impute(
                X_train=X_train.loc[:, self._features], return_imputer=True)
        else:
            self._X_train = X_train
        self._y_train = pd.Series(y_train, name=self._config["target_label"])
        self._clf.fit(
            self._X_train.loc[:, self._features], self._y_train)
        self._fitted = True

    def predict(self, X_test: pd.DataFrame, predict_proba: bool=None) -> np.ndarray:
        """Predicts the given features.

        Args:
            X_test (pd.DataFrame): Features to be predicted.
            predict_proba (bool, optional): If `True` the likelihood to be creak will be returned, else the predicted target.
            Defaults to None.

        Returns:
            np.ndarray: Predicted targets, or probability of creak.
        """
        self._config = creapy.utils.get_config()["MODEL"]
        if predict_proba is not None:
            assert isinstance(predict_proba, bool)
        else:
            predict_proba = self._config["CLASSIFIER"]["predict_proba"]
        if hasattr(self, "_imputer"):
            X_test = pd.DataFrame(self._imputer.transform(
                X_test.loc[:, self._features]), columns=self._X_train.columns, index=X_test.index)
        if predict_proba is True:
            _target_index = np.argwhere(
                self._clf.classes_ == self._config["CLASSIFIER"]["target_name"]).item()
            y_pred = self._clf.predict_proba(X_test[self._features])[
                :, _target_index].flatten()
            if self._config["POSTPROCESSING"]["MAVG"]["mavg"] is True:
                length, mode = map(
                    self._config["POSTPROCESSING"]["MAVG"]["VALUES"].get, ("length", "mode"))
                y_pred = creapy.postprocessing.moving_average(y_pred, length, mode)
        else:
            y_pred = self._clf.predict(X_test[self._features])

        return y_pred


def read_wav(
        data,
        sr,
        normalize: bool = True,
        start: float = 0.0,
        end: float | int = -1,
        mono=True
) -> tuple[np.ndarray, int]:
    if mono is True and data.ndim > 1:
        data = data.sum(axis=1) / data.shape[1]

    max_ = max(abs(data))
    if end == -1:
        data = data[int(start*sr):]
    else:
        data = data[int(start*sr):int(end*sr)]

    if normalize is True:
        data /= max_

    return data, sr


def _hnr(data: np.ndarray, sound: pm.Sound, sr) -> float:
    try:
        harmonicity = sound.to_harmonicity()
    except pm.PraatError:
        hnr = np.nan
    else:
        # taken from
        # https://parselmouth.readthedocs.io/en/stable/examples/batch_processing.html?highlight=harmonicity#Batch-processing-of-files
        # check if empty
        valid_values = harmonicity.values[harmonicity.values != -200]
        if valid_values.size > 0:
            hnr = valid_values.mean()
        else:
            hnr = np.nan
    return hnr


def blockwise_feature_calculation(data: np.ndarray, sr, feature):
    sounds = [pm.Sound(values=block, sampling_frequency=sr) for block in data]
    function = FEATURE_MAPPING[feature]
    res = [function(block, sound, sr) for block, sound in zip(data, sounds)]
    return np.array(res)


def process_file(data, sample_rate: int = 16_000):
    _config = creapy.utils.get_config()
    user_cfg = _config['USER']
    model_cfg = _config['MODEL']

    start, end = user_cfg['audio_start'], user_cfg['audio_end']
    data, sr = read_wav(data, sample_rate, start=start, end=end)

    w = hann(int(user_cfg["block_size"] * sample_rate))
    creak_data_buff = creapy.preprocessing.buffer(data, sample_rate, window=w)
    data_buffer = creak_data_buff.T

    unvoiced_excl = model_cfg['PREPROCESSING']['UNVOICED_EXCLUSION']
    preprocessing_features = [key for key, val in unvoiced_excl.items() if val is True]

    elimination_chunks = np.stack([
    blockwise_feature_calculation(
            data_buffer, sample_rate, feature
        ) for feature in preprocessing_features
    ], axis=1)

    preproc_values = unvoiced_excl['VALUES']
    preproc_values['ZCR']['threshold'] = user_cfg['zcr_threshold']
    preproc_values['STE']['threshold'] = user_cfg['ste_threshold']

    thresholds = np.array([
        creapy.postprocessing.thresholding(
            series=elimination_chunks[:, i],
            **preproc_values[feature.upper()]
        )
        for i, feature in enumerate(preprocessing_features)
    ])
    included_indices = thresholds.sum(axis=0) == 0

    if not np.any(included_indices):
        warnings.warn("Did not make classification. Adjust ZCR/STE thresholds.")
        y_pred = np.zeros(creak_data_buff.shape[1])
        X_test = pd.DataFrame(elimination_chunks, columns=preprocessing_features)
        return X_test, y_pred, included_indices

    class_features = model_cfg["FEATURES"]["for_classification"]
    X_class = np.stack([
        blockwise_feature_calculation(
            data_buffer[included_indices], sample_rate, feature
        ) for feature in class_features
    ], axis=1)

    _X_test = pd.DataFrame(
        X_class,
        columns=class_features,
        index=np.flatnonzero(included_indices)
    )

    X_all = np.zeros((elimination_chunks.shape[0], elimination_chunks.shape[1] + len(class_features)))
    X_all[:, :elimination_chunks.shape[1]] = elimination_chunks
    X_all[included_indices, elimination_chunks.shape[1]:] = X_class

    X_test = pd.DataFrame(X_all, columns=preprocessing_features + class_features)

    y_pred = np.zeros(creak_data_buff.shape[1])
    model_path = Path("./saved_models/creapy/model_ALL.csv")
    model = load_model(model_path)

    y_pred[included_indices] = model.predict(_X_test)

    return X_test, y_pred, included_indices


def load_model(filepath: str = None) -> Model:
    """Loads a already fitted model from a csv file.

    Args:
        filepath (str, optional): Location of the model csv file. Defaults to None.

    Returns:
        Model: Fitted Model for creak classification.
    """
    filepath = Path(filepath)

    _config = creapy.utils.get_config()
    _X_combined = pd.read_csv(filepath)
    model = Model()
    _target_column = _config["MODEL"]["target_label"]
    _feature_columns = _config["MODEL"]["FEATURES"]["for_classification"]
    _X_train, _y_train = _X_combined[_feature_columns], _X_combined[_target_column]
    model.fit(_X_train, _y_train)
    return model


FEATURE_MAPPING = {
    "cpp": _cpp,
    "hnr": _hnr,
    "h1h2": _h1_h2,
    "jitter": _jitter,
    "shimmer": _shimmer,
    "f0mean": _f0mean,
    "zcr": _zcr,
    "ste": _ste,
}

clfs = {
    "rfc": RandomForestClassifier,
    "mlp": MLPClassifier
}
