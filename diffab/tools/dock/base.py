import abc
from typing import List


FilePath = str


class DockingEngine(abc.ABC):

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, typ, value, traceback):
        pass

    @abc.abstractmethod
    def set_receptor(self, pdb_path: FilePath):
        pass

    @abc.abstractmethod
    def set_ligand(self, pdb_path: FilePath):
        pass

    @abc.abstractmethod
    def dock(self) -> List[FilePath]:
        pass
