"""
Classifier and Regressor Factory for SIGR Evaluator

Provides multiple classifier options following GenePT evaluation methodology.
Also provides regressors for perturbation prediction task.
"""

import logging
from typing import Union, Dict, Any, Optional

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


logger = logging.getLogger(__name__)


# Type alias for classifiers
ClassifierType = Union[LogisticRegression, RandomForestClassifier]
RegressorType = Union[Ridge, RandomForestRegressor, KNeighborsRegressor, MLPRegressor]


def get_classifier(
    classifier_type: str = 'logistic',
    random_state: int = 42,
    **kwargs
) -> ClassifierType:
    """
    Get a classifier instance.

    Args:
        classifier_type: Type of classifier ('logistic' or 'random_forest')
        random_state: Random seed for reproducibility
        **kwargs: Additional classifier-specific parameters

    Returns:
        Configured classifier instance

    Examples:
        >>> clf = get_classifier('logistic')
        >>> clf = get_classifier('random_forest', n_estimators=200)
    """
    if classifier_type == 'logistic':
        return LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            random_state=random_state,
            multi_class=kwargs.get('multi_class', 'multinomial'),
            solver=kwargs.get('solver', 'lbfgs'),
            class_weight=kwargs.get('class_weight', 'balanced'),
            n_jobs=kwargs.get('n_jobs', -1)
        )
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=random_state,
            class_weight=kwargs.get('class_weight', 'balanced'),
            n_jobs=kwargs.get('n_jobs', -1)
        )
    else:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}. "
            f"Available: 'logistic', 'random_forest'"
        )


def get_default_classifier_params(classifier_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a classifier type.

    Args:
        classifier_type: Type of classifier

    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'logistic': {
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'solver': 'lbfgs',
            'class_weight': 'balanced',
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
        }
    }
    return defaults.get(classifier_type, {})


class ClassifierFactory:
    """
    Factory class for creating and managing classifiers.

    Supports creating multiple classifier instances with consistent configuration.
    """

    AVAILABLE_CLASSIFIERS = ['logistic', 'random_forest']

    def __init__(
        self,
        default_type: str = 'logistic',
        random_state: int = 42,
        **default_kwargs
    ):
        """
        Initialize the factory.

        Args:
            default_type: Default classifier type
            random_state: Random seed
            **default_kwargs: Default parameters for all classifiers
        """
        self.default_type = default_type
        self.random_state = random_state
        self.default_kwargs = default_kwargs

    def create(
        self,
        classifier_type: Optional[str] = None,
        **kwargs
    ) -> ClassifierType:
        """
        Create a classifier instance.

        Args:
            classifier_type: Type of classifier (uses default if None)
            **kwargs: Override default parameters

        Returns:
            Configured classifier instance
        """
        clf_type = classifier_type or self.default_type

        # Merge default kwargs with provided kwargs
        merged_kwargs = {**self.default_kwargs, **kwargs}

        return get_classifier(
            classifier_type=clf_type,
            random_state=self.random_state,
            **merged_kwargs
        )

    def create_all(self, **kwargs) -> Dict[str, ClassifierType]:
        """
        Create instances of all available classifiers.

        Args:
            **kwargs: Parameters passed to all classifiers

        Returns:
            Dictionary mapping classifier name to instance
        """
        return {
            clf_type: self.create(clf_type, **kwargs)
            for clf_type in self.AVAILABLE_CLASSIFIERS
        }


# ============================================================================
# Regressor Factory for Perturbation Task
# ============================================================================

def get_regressor(
    regressor_type: str = 'ridge',
    random_state: int = 42,
    **kwargs
) -> RegressorType:
    """
    Get a regressor instance for perturbation prediction.

    Args:
        regressor_type: Type of regressor ('ridge', 'knn', 'mlp', 'random_forest')
        random_state: Random seed for reproducibility
        **kwargs: Additional regressor-specific parameters

    Returns:
        Configured regressor instance

    Examples:
        >>> reg = get_regressor('ridge', alpha=1.0)
        >>> reg = get_regressor('knn', n_neighbors=10)
    """
    if regressor_type == 'ridge':
        return Ridge(
            alpha=kwargs.get('alpha', 1.0),
            fit_intercept=kwargs.get('fit_intercept', True),
        )
    elif regressor_type == 'knn':
        return KNeighborsRegressor(
            n_neighbors=kwargs.get('n_neighbors', 5),
            weights=kwargs.get('weights', 'distance'),
            algorithm=kwargs.get('algorithm', 'auto'),
            n_jobs=kwargs.get('n_jobs', -1),
        )
    elif regressor_type == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)),
            activation=kwargs.get('activation', 'relu'),
            max_iter=kwargs.get('max_iter', 500),
            early_stopping=kwargs.get('early_stopping', True),
            random_state=random_state,
        )
    elif regressor_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            random_state=random_state,
            n_jobs=kwargs.get('n_jobs', -1),
        )
    else:
        raise ValueError(
            f"Unknown regressor type: {regressor_type}. "
            f"Available: 'ridge', 'knn', 'mlp', 'random_forest'"
        )


def get_default_regressor_params(regressor_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a regressor type.

    Args:
        regressor_type: Type of regressor

    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'ridge': {
            'alpha': 1.0,
            'fit_intercept': True,
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'distance',
        },
        'mlp': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'max_iter': 500,
            'early_stopping': True,
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
        }
    }
    return defaults.get(regressor_type, {})


class RegressorFactory:
    """
    Factory class for creating and managing regressors.

    Used primarily for perturbation prediction task.
    """

    AVAILABLE_REGRESSORS = ['ridge', 'knn', 'mlp', 'random_forest']

    def __init__(
        self,
        default_type: str = 'ridge',
        random_state: int = 42,
        **default_kwargs
    ):
        """
        Initialize the factory.

        Args:
            default_type: Default regressor type
            random_state: Random seed
            **default_kwargs: Default parameters for all regressors
        """
        self.default_type = default_type
        self.random_state = random_state
        self.default_kwargs = default_kwargs

    def create(
        self,
        regressor_type: Optional[str] = None,
        **kwargs
    ) -> RegressorType:
        """
        Create a regressor instance.

        Args:
            regressor_type: Type of regressor (uses default if None)
            **kwargs: Override default parameters

        Returns:
            Configured regressor instance
        """
        reg_type = regressor_type or self.default_type
        merged_kwargs = {**self.default_kwargs, **kwargs}

        return get_regressor(
            regressor_type=reg_type,
            random_state=self.random_state,
            **merged_kwargs
        )

    def create_all(self, **kwargs) -> Dict[str, RegressorType]:
        """
        Create instances of all available regressors.

        Args:
            **kwargs: Parameters passed to all regressors

        Returns:
            Dictionary mapping regressor name to instance
        """
        return {
            reg_type: self.create(reg_type, **kwargs)
            for reg_type in self.AVAILABLE_REGRESSORS
        }
