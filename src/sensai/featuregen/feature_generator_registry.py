import logging
from typing import Callable, Dict, TYPE_CHECKING, Hashable, Union

import pandas as pd

from . import FeatureGenerator, MultiFeatureGenerator
from ..util.string import list_string

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class FeatureGeneratorRegistry:
    """
    Represents a registry for named feature generators which can be instantiated via factories.

    In addition to functions registerFactory and getFeatureGenerator, feature generators can be registered and retrieved via \n
    registry.<name> = <featureGeneratorFactory> \n
    registry.<name> \n

    Example:
        >>> from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns
        >>> import pandas as pd

        >>> df = pd.DataFrame({"foo": [1, 2, 3], "bar": [7, 8, 9]})
        >>> registry = FeatureGeneratorRegistry()
        >>> registry.testFgen = lambda: FeatureGeneratorTakeColumns("foo")
        >>> registry.testFgen().generate(df)
           foo
        0    1
        1    2
        2    3
    """
    def __init__(self, use_singletons=False):
        """
        :param use_singletons: if True, internally maintain feature generator singletons, such that there is at most one
            instance for each name
        """
        # Important: Don't set public members in init. Since we override setattr this would lead to undesired consequences
        self._feature_generator_factories: Dict[Hashable, Callable[[], FeatureGenerator]] = {}
        self._feature_generator_singletons: Dict[Hashable, Callable[[], FeatureGenerator]] = {}
        self._use_singletons = use_singletons

    def __setattr__(self, name: str, value):
        if not name.startswith("_"):
            self.register_factory(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item: str):
        factory = self._feature_generator_factories.get(item)
        if factory is not None:
            return factory
        else:
            raise AttributeError(item)

    @property
    def available_features(self):
        return list(self._feature_generator_factories.keys())

    @staticmethod
    def _name(name: Hashable):
        # for enums, which have .name, use the name only, because it is less problematic to persist
        if hasattr(name, "name"):
            name = name.name
        return name

    def register_factory(self, name: Hashable, factory: Callable[[], FeatureGenerator]):
        """
        Registers a feature generator factory which can subsequently be referenced by models via their name
        :param name: the name (which can, in particular, be a string or an enum item)
        :param factory: the factory
        """
        name = self._name(name)
        if name in self._feature_generator_factories:
            raise ValueError(f"Generator for name '{name}' already registered")
        self._feature_generator_factories[name] = factory

    def get_feature_generator(self, name) -> FeatureGenerator:
        """
        Creates a feature generator from a name, which must have been previously registered.
        The name of the returned feature generator (as returned by getName()) is set to name.

        :param name: the name (which can, in particular, be a string or an enum item)
        :return: a new feature generator instance (or existing instance for the case where useSingletons is enabled)
        """
        name = self._name(name)
        generator = self._feature_generator_singletons.get(name)
        if generator is None:
            factory = self._feature_generator_factories.get(name)
            if factory is None:
                raise ValueError(f"No factory registered for name '{name}': known names: {list_string(self._feature_generator_factories.keys())}. Use registerFeatureGeneratorFactory to register a new feature generator factory.")
            generator = factory()
            generator.set_name(name)
            if self._use_singletons:
                self._feature_generator_singletons[name] = generator
        return generator


class FeatureCollector(object):
    """
    A feature collector which can provide a multi-feature generator from a list of names/generators and registry
    """

    def __init__(self, *feature_generators_or_names: Union[str, FeatureGenerator], registry:
            FeatureGeneratorRegistry = None):
        """
        :param feature_generators_or_names: generator names (known to the registry) or generator instances.
        :param registry: the feature generator registry for the case where names are passed
        """
        self._feature_generators_or_names = feature_generators_or_names
        self._registry = registry
        self._multi_feature_generator = self._create_multi_feature_generator()

    def get_multi_feature_generator(self) -> MultiFeatureGenerator:
        return self._multi_feature_generator

    def get_normalisation_rules(self, include_generated_categorical_rules=True):
        return self.get_multi_feature_generator().get_normalisation_rules(
            include_generated_categorical_rules=include_generated_categorical_rules)

    def get_categorical_feature_name_regex(self) -> str:
        return self.get_multi_feature_generator().get_categorical_feature_name_regex()

    def _create_multi_feature_generator(self):
        feature_generators = []
        for f in self._feature_generators_or_names:
            if isinstance(f, FeatureGenerator):
                feature_generators.append(f)
            else:
                if self._registry is None:
                    raise Exception(f"Received feature name '{f}' instead of instance but no registry to perform the lookup")
                feature_generators.append(self._registry.get_feature_generator(f))
        return MultiFeatureGenerator(*feature_generators)
