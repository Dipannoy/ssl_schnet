from __future__ import print_function, division

import csv
import functools
import  json
# import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import os
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils import dense_to_sparse, add_self_loops
import copy

from datasets.augmentation import RotationTransformation, PerturbStructureTransformation, RemoveSitesTransformation


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, val_ratio=0.1, num_workers=1, 
                              pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    train_ratio = 1 - val_ratio
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    indices = list(range(total_size))
    # if kwargs['train_size']:
    #     train_size = kwargs['train_size']
    # else:
    #     train_size = int(train_ratio * total_size)
    # if kwargs['val_size']:
    #     valid_size = kwargs['val_size']
    # else:
    #     valid_size = int(val_ratio * total_size)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:])
    # print('val_sampler')
    # print(val_sampler)
    # if return_test:
    #     test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader  = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn, pin_memory=pin_memory)

    # train_loader = DataLoader(dataset, batch_size=batch_size,
    #                           sampler=train_sampler,
    #                           num_workers=num_workers, drop_last=True,
    #                           collate_fn=collate_fn, pin_memory=pin_memory)
    # val_loader = DataLoader(dataset, batch_size=batch_size,
    #                         sampler=val_sampler,
    #                         num_workers=num_workers, drop_last=True,
    #                         collate_fn=collate_fn, pin_memory=pin_memory)
    # if return_test:
    #     test_loader = DataLoader(dataset, batch_size=batch_size,
    #                              sampler=test_sampler,
    #                              num_workers=num_workers, drop_last=True,
    #                              collate_fn=collate_fn, pin_memory=pin_memory)
    # if return_test:
    #     return train_loader, val_loader, test_loader
    # else:
    #     return train_loader, val_loader
    return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    batch_cif_ids: list
    """
    batch_atom_fea_rot_1, batch_nbr_fea_rot_1, batch_nbr_weight_rot_1, batch_nbr_fea_idx_rot_1 = [], [], [], []
    batch_atom_fea_rot_2, batch_nbr_fea_rot_2, batch_nbr_weight_rot_2, batch_nbr_fea_idx_rot_2 = [], [], [], []
    # batch_atom_fea_rot_1, batch_nbr_fea_rot_1, batch_nbr_fea_idx_rot_1 = [], [], []
    # batch_atom_fea_rot_2, batch_nbr_fea_rot_2, batch_nbr_fea_idx_rot_2 = [], [], []
    crystal_atom_idx = []
    batch_cif_ids = []
    base_idx = 0
    prev_base_idx = 0
    total_atoms = 0
    # print('----------------------dataset length--------------------------')
    # print(len(dataset_list))
    for  i , (( atom_fea_rot_1 , nbr_fea_rot_1, nbr_fea_weight_rot_1, nbr_fea_idx_rot_1 ), ( atom_fea_rot_2 , nbr_fea_rot_2 ,nbr_fea_weight_rot_2, nbr_fea_idx_rot_2 ), cif_id ) \
            in enumerate(dataset_list):
        n_i = atom_fea_rot_1.shape[0]  # number of atoms for this crystal
        total_atoms += n_i

        # print('--------------------number of atoms----------------')
        # print(n_i)
        
        batch_atom_fea_rot_1.append(atom_fea_rot_1)
        batch_atom_fea_rot_2.append(atom_fea_rot_2)
        batch_nbr_fea_rot_1.append(nbr_fea_rot_1)
        batch_nbr_fea_rot_2.append(nbr_fea_rot_2)
        batch_nbr_weight_rot_1.append(nbr_fea_weight_rot_1)
        batch_nbr_weight_rot_2.append(nbr_fea_weight_rot_2)

        if((nbr_fea_idx_rot_1+base_idx)[1].max()>=total_atoms and i==len(dataset_list)-1):
          # print('----------problem found----------')

          nbr_fea_idx_rot_1[0] = nbr_fea_idx_rot_1[0]+base_idx
          nbr_fea_idx_rot_1[1] = nbr_fea_idx_rot_1[1]+(prev_base_idx + n_i)
          batch_nbr_fea_idx_rot_1.append(nbr_fea_idx_rot_1)
          # print(nbr_fea_idx_rot_1)

        else:
          batch_nbr_fea_idx_rot_1.append(nbr_fea_idx_rot_1+base_idx)
          # print(nbr_fea_idx_rot_1+base_idx)

        if((nbr_fea_idx_rot_2+base_idx)[1].max()>=total_atoms and i==len(dataset_list)-1):
          nbr_fea_idx_rot_2[0] = nbr_fea_idx_rot_2[0]+base_idx
          nbr_fea_idx_rot_2[1] = nbr_fea_idx_rot_2[1]+(prev_base_idx + n_i)
          batch_nbr_fea_idx_rot_2.append(nbr_fea_idx_rot_2)
        else:
          batch_nbr_fea_idx_rot_2.append(nbr_fea_idx_rot_2+base_idx)
          

        
        # batch_nbr_fea_idx_rot_2.append(nbr_fea_idx_rot_2+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_cif_ids.append(cif_id)
        prev_base_idx = copy.deepcopy(base_idx)
        base_idx += n_i
    # print('----------------------from collate---------------------------')
    # print(batch_nbr_fea_rot_1)
    # print(len(batch_nbr_fea_idx_rot_1[127][1]))

    edge_index_i = []
    edge_index_j = []
    total_edge_index_1 =[]
    edge_weight_1 = []
    edge_attr_1=[]
    # print(len(batch_nbr_fea_rot_1[0][0][0]))
    # print('---------not working-----------')
    # print(len(batch_nbr_fea_rot_1[0][0][0]))

    # for i in range(0,len(batch_nbr_fea_rot_1)):
    #   for j in range(0,len(batch_nbr_fea_rot_1[0])):
    #     for k in range(0,len(batch_nbr_fea_rot_1[0][0])):
    #       temp=[]
    #       for l in range(0,len(batch_nbr_fea_rot_1[i][j][k])):
    #         temp.append(batch_nbr_fea_rot_1[i][j][k][l])
    #       edge_attr_1.append(temp)

    # for i in range(0,len(batch_nbr_fea_idx_rot_1)):
    # print(batch_nbr_fea_idx_rot_1)
    # print(batch_nbr_fea_idx_rot_1[0][0])

    # print('---------atom length_1-----------')
    # print(len(batch_atom_fea_rot_1))
    # print(len(batch_atom_fea_rot_1[1]))

    # print('---------atom length_2-----------')
    # print(len(batch_atom_fea_rot_2))
    # print(len(batch_atom_fea_rot_2[1]))

    # print('------------------outer collate-------------------------')
    # print(len(batch_nbr_fea_idx_rot_1))
    # print('------------------inner collate-------------------------')

    for i in range(0,len(batch_nbr_fea_idx_rot_1)):
      # print(len(batch_nbr_fea_idx_rot_1[i][0]))
      for j in range(0,len(batch_nbr_fea_idx_rot_1[i][0])):
        edge_index_i.append(batch_nbr_fea_idx_rot_1[i][0][j])
      for j in range(0,len(batch_nbr_fea_idx_rot_1[i][1])):
        edge_index_j.append(batch_nbr_fea_idx_rot_1[i][1][j])
        # edge_weight_1.append(batch_nbr_weight_rot_1[i][j])
    total_edge_index_1.append(edge_index_i)
    total_edge_index_1.append(edge_index_j)

    # edge_attr_1 = torch.Tensor(edge_attr_1)

    edge_index_i = []
    edge_index_j = []
    total_edge_index_2 =[]
    edge_weight_2 = []

    edge_attr_2=[]

    # for i in range(0,len(batch_nbr_fea_rot_2)):
    #   for j in range(0,len(batch_nbr_fea_rot_2[0])):
    #     for k in range(0,len(batch_nbr_fea_rot_2[0][0])):
    #       temp=[]
    #       for l in range(0,len(batch_nbr_fea_rot_2[i][j][k])):
    #         temp.append(batch_nbr_fea_rot_2[i][j][k][l])
    #       edge_attr_2.append(temp)

    for i in range(0,len(batch_nbr_fea_idx_rot_2)):
      for j in range(0,len(batch_nbr_fea_idx_rot_2[i][0])):
        edge_index_i.append(batch_nbr_fea_idx_rot_2[i][0][j])
      for j in range(0,len(batch_nbr_fea_idx_rot_2[i][1])):
        edge_index_j.append(batch_nbr_fea_idx_rot_2[i][1][j])
        # edge_weight_2.append(batch_nbr_weight_rot_2[i][j])

    total_edge_index_2.append(edge_index_i)
    total_edge_index_2.append(edge_index_j)

    # print(edge_attr_2)
    # edge_attr_2 = torch.Tensor(edge_attr_2)

    # print('------------------start from collate-------------------------')
    # print(len(batch_atom_fea_rot_1))
    # print(len(batch_atom_fea_rot_1[0]))
    # print(len(batch_nbr_fea_rot_1))
    # print(len(batch_nbr_fea_rot_1[0]))
    # print(len(batch_nbr_fea_rot_1[0][0]))
    # print(torch.cat(batch_atom_fea_rot_1, dim=0).size())
    # print(torch.cat(batch_nbr_fea_rot_1, dim=0).size())
    # print(len(total_edge_index_1[0]))
    # print(len(total_edge_index_1[1]))


    # print('------------------end from collate-------------------------')





    return (torch.cat(batch_atom_fea_rot_1, dim=0),
            torch.cat(batch_nbr_fea_rot_1, dim=0),
            torch.cat(batch_nbr_weight_rot_1, dim=0),
            torch.Tensor(total_edge_index_1),
            # torch.cat(batch_nbr_fea_idx_rot_1, dim=0),
            crystal_atom_idx), \
    		(torch.cat(batch_atom_fea_rot_2, dim=0),
            torch.cat(batch_nbr_fea_rot_2, dim=0),
            torch.cat(batch_nbr_weight_rot_2, dim=0),
            torch.Tensor(total_edge_index_2),
            # torch.cat(batch_nbr_fea_idx_rot_2, dim=0),
            crystal_atom_idx),\
            batch_cif_ids

    # edge_attr_cat1 = torch.Tensor(torch.cat(batch_nbr_fea_rot_1, dim=0))
    # # edge_attr_cat1 = torch.cat(edge_attr_cat1, dim=0) 
    # edge_attr_cat1 = torch.Tensor(torch.cat(torch.Tensor(edge_attr_cat1), dim=0))

    # print(edge_attr_2[0])

    # return (torch.cat(batch_atom_fea_rot_1, dim=0),
    #         edge_attr_1,
    #         torch.Tensor(edge_weight_1),
    #         torch.LongTensor(total_edge_index_1),
    #         crystal_atom_idx), \
    # 		(torch.cat(batch_atom_fea_rot_2, dim=0),
    #         edge_attr_2,
    #         torch.Tensor(edge_weight_2),
    #         torch.LongTensor(total_edge_index_2),
    #         crystal_atom_idx),\
    #         batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class  AtomCustomJSONInitializer ( AtomInitializer ):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.
    atom_init.json: a JSON file that stores the initialization vector for each
    element.
    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=10, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self . root_dir  =  root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        cif_fns  = []
        for subdir, dirs, files in os.walk(root_dir):
            for  fn  in  files :
                if fn.endswith('.cif'):
                    cif_fns.append(os.path.join(subdir, fn))
        self.cif_data = cif_fns
        random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        random.shuffle(self.cif_data)
        # atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        atom_init_file = os.path.join('datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.rotater = RotationTransformation()
        self.perturber  =  PerturbStructureTransformation ( distance = 0.05 , min_distance = 0.0 )        
        self.masker = RemoveSitesTransformation()

    def __len__(self):
        # return len(self.id_prop_data)
        return len(self.cif_data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        # cif_id, __, target = self.id_prop_data[idx]
        # crys = Structure.from_file(os.path.join(self.root_dir,
        # cif_id + '. cif'))
        try:
            cif_fn = self.cif_data[idx]
            cif_id = cif_fn.split('/')[-1].replace('.cif', '')
            # print(cif_id)
            crys = Structure.from_file(cif_fn)
        except (RuntimeError, TypeError, NameError, ValueError):
            crys =  Structure.from_file('/home/amir/Rishi/Barlow_CGCNN/lanths/0.cif')

        crystal = crys.copy()
        # crystal_per_1 =  self.perturber.apply_transformation(crystal)
        # crystal_per_2 =  self.perturber.apply_transformation(crystal)

        # for  i  in  range ( 3 ):
        #     axis = np.zeros(3)
        #     axis[i] = 1
        #     rot_ang  =  np.random.uniform( - 90.0 , 90.0 )
        #     crystal_rot_1 = self.rotater.apply_transformation(crystal_per_1, axis, rot_ang, angle_in_radians=False)

        # for  i  in  range ( 3 ):
        #     axis = np.zeros(3)
        #     axis[i] = 1
        #     rot_ang  =  np.random.uniform( - 90.0 , 90.0 )
        #     crystal_rot_2 = self.rotater.apply_transformation(crystal_per_2, axis, rot_ang, angle_in_radians=False)

        crystal_rot_1 = crys.copy()
        crystal_rot_2 = crys.copy()

        num_sites = crys.num_sites

        mask_num = int(max([1, math.floor(0.10*num_sites)]))#int(np.floor(0.10*num_sites))
        indices_remove_1 = np.random.choice(num_sites, mask_num, replace=False)
        indices_remove_2 = np.random.choice(num_sites, mask_num, replace=False)

        atom_fea_rot_1 = np.vstack([self.ari.get_atom_fea(crystal_rot_1[i].specie.number)
                              for  i  in  range ( len ( crystal_rot_1 ))])

        # mask 10% atoms
        atom_fea_rot_1[indices_remove_1,:] = 0
        crys_1_length = len(atom_fea_rot_1)
        atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        
        all_nbrs_rot_1 = crystal_rot_1.get_all_neighbors(self.radius, include_index=True)
        # print(all_nbrs_rot_1[0])
        # print('-----------------------------------------------------')
        all_nbrs_rot_1 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_1]
        nbr_fea_idx_rot_1 , nbr_fea_rot_1  = [], []
        for nbr in all_nbrs_rot_1:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*len(nbr)))
                # print('-----edge num------')
                # print(keep_edge_num)
                keep_edge_indices = np.random.choice(len(nbr), keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))

            
            else:
                # mask 10% edges
                keep_edge_num = int(np.ceil(0.90*self.max_num_nbr))
                # print('-----edge num------')
                # print(keep_edge_num)
                keep_edge_indices = np.random.choice(self.max_num_nbr, keep_edge_num, replace=False)
                # print('----------------------------------------')
                # print(keep_edge_indices)
                # print(nbr)
              
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                # print(nbr)
                # print('--------------------------------------')
                nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))
                

        nbr_fea_idx_rot_1, nbr_fea_rot_1 = np.array(nbr_fea_idx_rot_1), np.array(nbr_fea_rot_1)
        # print(nbr_fea_idx_rot_1)
        nbr_fea_weight_rot_1 = copy.deepcopy(nbr_fea_rot_1)
        # nbr_fea_rot_1 = self.gdf.expand(nbr_fea_rot_1)
        atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        # nbr_fea_rot_1 = torch.Tensor(nbr_fea_rot_1)
        nbr_fea_idx_rot_1 = torch.LongTensor(nbr_fea_idx_rot_1)
        nbr_fea_weight_rot_1 = torch.Tensor(nbr_fea_weight_rot_1)
        out1 = dense_to_sparse(nbr_fea_weight_rot_1)
        edge_index_1 = out1[0]
        edge_weight_1 = out1[1]

        edge_index, edge_weight = add_self_loops(
            edge_index_1, edge_weight_1, num_nodes=crys_1_length, fill_value=0
        )
        edge_index_1 = edge_index
        edge_weight_1 = edge_weight

        nbr_fea_rot_1 = self.gdf.expand(edge_weight_1)
        nbr_fea_rot_1 = torch.Tensor(nbr_fea_rot_1)



        # print(edge_index_1.size())
        # print(atom_fea_rot_1.size())
        # atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)

        # print(edge_weight_1.size())


        # target = torch.Tensor([float(target)])

        atom_fea_rot_2 = np.vstack([self.ari.get_atom_fea(crystal_rot_2[i].specie.number)
                              for  i  in  range ( len ( crystal_rot_2 ))])
        
        # mask 25% atoms
        atom_fea_rot_2[indices_remove_2,:] = 0
        crys_2_length = len(atom_fea_rot_2)
        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        all_nbrs_rot_2 = crystal_rot_2.get_all_neighbors(self.radius, include_index=True)
        all_nbrs_rot_2 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_2]
        nbr_fea_idx_rot_2 , nbr_fea_rot_2  = [], []
        for nbr in all_nbrs_rot_2:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))

                # mask 10% edges
                # print('-----edge num------')
                # print(keep_edge_num)
                keep_edge_num = int(np.ceil(0.90*len(nbr)))
                keep_edge_indices = np.random.choice(len(nbr), keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))

            
            else:
                # mask 10% edges
                # print('-----edge num------')
                # print(keep_edge_num)
                keep_edge_num = int(np.ceil(0.90*self.max_num_nbr))
                keep_edge_indices = np.random.choice(self.max_num_nbr, keep_edge_num, replace=False)
                nbr  = [ nbr [ i ] for  i  in  keep_edge_indices ]
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
                                  [0] * (self.max_num_nbr - keep_edge_num))
                nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
                              [self.radius + 1.] * (self.max_num_nbr - keep_edge_num))
        # print('-----nbr fea len------')
        # print(len(nbr_fea_rot_2))
        nbr_fea_idx_rot_2, nbr_fea_rot_2 = np.array(nbr_fea_idx_rot_2), np.array(nbr_fea_rot_2)
        nbr_fea_weight_rot_2 = copy.deepcopy(nbr_fea_rot_2)
        # nbr_fea_rot_2 = self.gdf.expand(nbr_fea_rot_2)
        # print('--------------After Gaussian-------------------------------')
        # # print(nbr_fea_weight_rot_2)
        # print(len(nbr_fea_rot_2[0][1]))

        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        # nbr_fea_rot_2 = torch.Tensor(nbr_fea_rot_2)
        nbr_fea_idx_rot_2 = torch.LongTensor(nbr_fea_idx_rot_2)
        nbr_fea_weight_rot_2 = torch.Tensor(nbr_fea_weight_rot_2)
        out2 = dense_to_sparse(nbr_fea_weight_rot_2)
        edge_index_2 = out2[0]
        edge_weight_2 = out2[1]


        edge_index_2, edge_weight_2 = add_self_loops(
            edge_index_2, edge_weight_2, num_nodes=crys_2_length, fill_value=0
        )
        # edge_index_1 = edge_index
        # edge_weight_1 = edge_weight

        nbr_fea_rot_2 = self.gdf.expand(edge_weight_2)
        nbr_fea_rot_2 = torch.Tensor(nbr_fea_rot_2)

        # print(edge_index_1.size())
        # print(edge_weight_1.size())
        # print(nbr_fea_rot_1.size())


        # print('-----edge weight------')
        # print(nbr_fea_weight_rot_2.size())
        # target = torch.Tensor([float(target)])
        # target = torch.Tensor([float(target)])
        # return (atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), cif_id
        return (atom_fea_rot_1, nbr_fea_rot_1,edge_weight_1, edge_index_1), (atom_fea_rot_2, nbr_fea_rot_2,  edge_weight_2, edge_index_2), cif_id

        # return (atom_fea_rot_1, nbr_fea_rot_1,nbr_fea_weight_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2,  nbr_fea_weight_rot_2, nbr_fea_idx_rot_2), cif_id
