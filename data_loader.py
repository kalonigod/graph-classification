import numpy as np
import os
from scipy.sparse import csc_matrix as sparse_mat


class DataLoader(object):
    def __init__(self, root_dir, dev_ratio=0.1, test_ratio=0.1):
        """
        Root_dir should contain three files: [feature.txt, group.txt, graph.txt]
        :param str root_dir: Root directory for dataset.
        :param float dev_ratio: Dev set ratio.
        :param float test_ratio: Test set ratio.
        """
        with open(os.path.join(root_dir, 'feature.txt')) as fp:
            features = list()
            for line in fp.readlines():
                features.append(list(map(int, line.split())))
        features = np.array(features)

        with open(os.path.join(root_dir, 'group.txt')) as fp:
            labels = list()
            for line in fp.readlines():
                labels.append(int(line.split()[1]))
        assert len(labels) == features.shape[0]
        self.labels = np.array(labels)

        self.n_nodes, self.feature_dim = features.shape

        print('# nodes: {}'.format(self.n_nodes))
        print('Feature Dimension: {}'.format(self.feature_dim))

        with open(os.path.join(root_dir, 'graph.txt')) as fp:
            n1_list, n2_list = list(), list()
            for line in fp.readlines():
                n1, n2 = map(int, line.split())
                n1_list.extend([n1, n2])
                n2_list.extend([n2, n1])
        print('# edges: {}'.format(len(n1_list)//2))
        data = [True] * len(n1_list)
        self.edges = sparse_mat((data, (n1_list, n2_list)), shape=(self.n_nodes, self.n_nodes), dtype=np.bool)

        n_train = int((1-dev_ratio-test_ratio) * self.n_nodes)
        n_dev = int(dev_ratio * self.n_nodes)
        self.split = {
            'train': list(range(n_train)),
            'dev': list(range(n_train, n_train + n_dev)),
            'test': list(range(n_train + n_dev, self.n_nodes))
        }

    def _get_nearby_helper(self, nodes, depth):
        if depth == 0:
            return nodes
        rst = set()
        for node in nodes:
            for adj in self.edges[node].nonzero()[1]:
                rst.add(adj)
        return self._get_nearby_helper(rst, depth-1)

    def get_nearby(self, node: int, depth=1, exclude_self=False) -> list:
        """
        Random walk. Given a node, return the indices of its nearby nodes.
        If depth == 0, then the node itself will be returned.
        :param int node: Index of the central node.
        :param int depth: Maximum steps of random walk.
        :param bool exclude_self: If True, the node itself will not be counted in.
        :return:
        """
        init = set()
        init.add(node)
        rst = self._get_nearby_helper(init, depth)
        if exclude_self:
            rst.remove(node)
        else:
            rst.add(node)
        return sorted(list(rst))


if __name__ == '__main__':
    dl = DataLoader('./data/cora')
    for depth_ in range(3):
        print('depth == {}'.format(depth_))
        print(dl.get_nearby(3, depth_))
