import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import NotFittedError
from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler



class Custom_Classifier():

    def __init__(self, initial_classifier: XGBClassifier, lenticular_kwargs: dict, spiral_kwargs: dict,
                 irregular_kwargs: dict, oversample: bool, random_state: int = 109) -> None:
        """Initialize Custom_Classifier class
        Args:
            initial_classifer: Pre-trained classifer that classifies Hubble class.
            <hubble_class>_kwargs: contains classifiers for each Hubble class as well as their parameters:
                {
                    estimator, max_depth, n_estimators, learning_rate
                }
        """
        self.initial_classifier = initial_classifier
        self._is_fit = False  # indicator variable if the sub classifiers are fit

        # Define a lambda function to return the classifier with or without random over sampling
        if oversample: 
            get_pline = lambda pline_kwargs: Pipeline([
                ('ros', RandomOverSampler(random_state=random_state)), 
                ('classifier', pline_kwargs['estimator'](max_depth=pline_kwargs['max_depth'], 
                                                            n_estimators=pline_kwargs['n_estimators'], 
                                                            learning_rate=pline_kwargs['learning_rate'], 
                                                            random_state=random_state))
            ])
        else: 
            get_pline = lambda pline_kwargs: pline_kwargs['estimator'](max_depth=pline_kwargs['max_depth'], 
                                                            n_estimators=pline_kwargs['n_estimators'], 
                                                            learning_rate=pline_kwargs['learning_rate'], 
                                                            random_state=random_state)

        # Get subclassifiers
        self.lenticular_classifier = get_pline(lenticular_kwargs)
        self.spiral_classifier = get_pline(spiral_kwargs)
        self.irregular_classifier = get_pline(irregular_kwargs)


    def fit(self, X_train: np.ndarray, y_train_hubble: np.ndarray, y_train_subclass: np.ndarray):
        """Fit the subclass classifiers.
        Args:
            X_train: Train dset.
            y_train_hubble: Train hubble classes.
            y_train_subclass: Train hubble subclasses.
        Returns:
            Trained instance of self.
        """

        # Create masks for each hubble_class
        lenticular_mask = y_train_subclass <= -1
        spiral_mask = (y_train_subclass > -1) & (y_train_subclass <= 7)
        irregular_mask = y_train_subclass > 7

        # Fit each of the base classifiers
        self.lenticular_classifier.fit(X_train[lenticular_mask], y_train_subclass[lenticular_mask])
        self.spiral_classifier.fit(X_train[spiral_mask], y_train_subclass[spiral_mask])
        self.irregular_classifier.fit(X_train[irregular_mask], y_train_subclass[irregular_mask])
        self._is_fit = True

        return self


    def predict(self, X: np.ndarray):
        """Predict.
        Args:
            X: Inputs.
        Returns:
            Predictions of the subclasses corresponding to each row in the X np array.
        """

        if not self._is_fit:
            raise NotFittedError('You must fit your Custom_Classifier before you call the predict() function!')

        # Predict hubble classes
        hubble_preds = self.initial_classifier.predict(X)

        # Masks for the different hubble classes
        lenticular_mask = hubble_preds == 0
        spiral_mask = hubble_preds == 1
        irregular_mask = hubble_preds == 2

        # Predict subclasses
        preds = np.zeros((hubble_preds.shape[0])) - 1000  # Array of nonsense predictions that we will fill in
        preds[lenticular_mask] = self.lenticular_classifier.predict(X[lenticular_mask])
        preds[spiral_mask] = self.spiral_classifier.predict(X[spiral_mask])
        preds[irregular_mask] = self.irregular_classifier.predict(X[irregular_mask])

        return np.rint(preds).astype(int)
