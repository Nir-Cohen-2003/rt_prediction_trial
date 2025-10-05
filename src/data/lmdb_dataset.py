import lmdb
import pickle
import torch
from torch_geometric.data import Dataset, Data, Batch

class LMDBGraphDataset(Dataset):
    """
    Stores PyG graphs in LMDB. Opens LMDB environment lazily per process to avoid
    inheriting C-level handles across multiprocessing forks (common cause of segfaults).
    """
    def __init__(self, lmdb_path, readonly=True):
        super().__init__()
        # keep path/flags for lazy open; do NOT keep env handle across process boundaries
        self.lmdb_path = lmdb_path
        self.readonly = readonly
        self.env = None  # will be opened lazily in the current process
        # read length once (open temporary env and close immediately)
        tmp_env = lmdb.open(lmdb_path, map_size=2**40, subdir=False, readonly=readonly, lock=not readonly)
        with tmp_env.begin() as txn:
            length_bytes = txn.get(b'__len__')
            self.length = pickle.loads(length_bytes) if length_bytes else 0
        tmp_env.close()

    def _ensure_env(self):
        if self.env is None:
            # open env for the current process / worker
            self.env = lmdb.open(self.lmdb_path, map_size=2**40, subdir=False, readonly=self.readonly, lock=not self.readonly)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ensure env is opened in this process (worker or main)
        self._ensure_env()
        with self.env.begin() as txn:
            graph_bytes = txn.get(str(idx).encode())
            if graph_bytes is None:
                raise IndexError(f"Index {idx} out of range")
            graph = pickle.loads(graph_bytes)
            return graph

    def __getstate__(self):
        """
        Remove non-picklable / process-local handles when the dataset is pickled
        (e.g. when DataLoader spawns worker processes). Workers will reopen env.
        """
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None

    @staticmethod
    def from_graphs(graphs, lmdb_path, append=False):
        """
        Save a list of PyG Data objects to LMDB.
        If append=True, adds to existing DB.
        """
        env = lmdb.open(lmdb_path, map_size=2**40, subdir=False, lock=True)
        with env.begin(write=True) as txn:
            # Get current length if appending
            length_bytes = txn.get(b'__len__')
            start_idx = pickle.loads(length_bytes) if (append and length_bytes) else 0
            for i, graph in enumerate(graphs):
                txn.put(str(start_idx + i).encode(), pickle.dumps(graph))
            txn.put(b'__len__', pickle.dumps(start_idx + len(graphs)))
        env.close()
