import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import NotFittedError
from xgboost import XGBClassifier, XGBRegressor


class Custom_Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self, initial_classifier: XGBClassifier, base_sub_classifier: ClassifierMixin, random_state: int = 109) -> None:
        """Initialize Custom_Classifier class
        Args:
            initial_classifer: Pre-trained classifer that classifies Hubble class.
            base_final_classifier: Type of class to initialize our Hubble-subclass classifier as.
        """
        self.initial_classifier = initial_classifier
        self._is_fit = False  # indicator variable if the sub classifiers are fit

        # Sub classifiers
        self.lenticular_classifier = base_sub_classifier(random_state=random_state)
        self.spiral_classifier = base_sub_classifier(random_state=random_state)
        self.irregular_classifier = base_sub_classifier(random_state=random_state)


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
        lenticular_mask = y_train_hubble <= -1
        spiral_mask = (y_train_hubble > -1) & (y_train_hubble <= 7)
        irregular_mask = y_train_hubble > 7

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
        lenticular_mask = hubble_preds <= -1
        spiral_mask = (hubble_preds > -1) & (hubble_preds <= 7)
        irregular_mask = hubble_preds > 7

        # Predict subclasses
        preds = np.zeros(hubble_preds.shape) - 1000  # Array of nonsense predictions that we will fill in
        preds[lenticular_mask] = self.lenticular_classifier.predict(X[lenticular_mask])
        preds[spiral_mask] = self.spiral_classifier.predict(X[spiral_mask])
        preds[irregular_mask] = self.irregular_classifier.predict(X[irregular_mask])

        return preds
