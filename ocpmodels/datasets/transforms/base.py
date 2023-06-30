from abc import abstractmethod
from typing import Union

from ocpmodels.common import DataDict


class AbstractDataTransform(object):
    @abstractmethod
    def __call__(self, data: DataDict) -> DataDict:
        """
        Call function for an abstract data transformation.

        This abstract method defines the expected input and outputs;
        we retrieve a dictionary of data, and we expect a dictionary
        of data to come back out.

        In some sense, this might not be what you would think
        of canonically as a "transform" in that it's not operating
        in place, but rather as modular components of the pipeline.

        Parameters
        ----------
        data : DataDict
            Item from an abstract dataset

        Returns
        -------
        DataDict
            Transformed item from an abstract dataset
        """
        raise NotImplementedError
