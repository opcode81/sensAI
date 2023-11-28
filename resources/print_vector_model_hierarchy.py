from anytree import Node, RenderTree
import inspect

from sensai import VectorRegressionModel, VectorModel, VectorClassificationModel


class ClassHierarchy:
    def __init__(self, cls, skip_intermediate_abc=True, retained_intermediate_classes=()):
        self.retained_intermediate_classes = retained_intermediate_classes
        self.skip_intermediate_abc = skip_intermediate_abc
        self.root = self._scan_subclasses(cls, None, True)

    @staticmethod
    def _isabstract(cls):
        return inspect.isabstract(cls)  # or "Abstract" in cls.__name__

    def _scan_subclasses(self, cls, parent, is_root):
        skip_node = not is_root and self.skip_intermediate_abc \
                    and self._isabstract(cls) \
                    and cls not in self.retained_intermediate_classes

        if not skip_node:
            node = Node(cls.__name__, parent=parent)
        else:
            node = parent

        subclasses = list(cls.__subclasses__())
        subclasses.sort(key=lambda x: x.__name__)
        for subclass in subclasses:
            self._scan_subclasses(subclass, node, False)

        return node

    def print(self):
        for pre, _, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name))


if __name__ == '__main__':
    # import optional packages such that the classes will be included in the hierarchy
    from sensai import nearest_neighbors
    from sensai import xgboost
    from sensai.util import mark_used
    from sensai import torch as sensai_torch
    from sensai import tensorflow as sensai_tf
    from sensai import lightgbm as sensai_lgbm
    from sensai.torch import torch_models
    from sensai import sklearn_quantile

    mark_used(xgboost, nearest_neighbors, sensai_torch, sensai_tf, sensai_lgbm, torch_models, sklearn_quantile)

    h = ClassHierarchy(VectorModel,
        skip_intermediate_abc=True,
        retained_intermediate_classes=(VectorRegressionModel, VectorClassificationModel))
    h.print()

