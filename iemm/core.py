"""
Iterative Evidential Mistakeness Minimization

This module implements the core IEMM algorithm for explainable evidential clustering.
The algorithm generates interpretable decision tree explanations for evidential clustering
functions based on Dempster-Shafer theory.

Adapted from https://github.com/ArthurHoa/conflict-edt/tree/master (Conflict EDT, Arthur Hoarau, 02/01/2025).

Author: Victor Souza
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List, Tuple, Callable, Dict, Any

import numpy as np
import pandas as pd

import schemdraw
import schemdraw.flow as flow
# schemdraw.theme('dark')

from . import belief as ibelief

class Loss:
    """
    Loss functions for evidential clustering.
    
    This class provides static methods for computing different types of loss functions
    used in the IEMM algorithm, including mistakeness losses for decision tree construction.
    """
    
    @staticmethod
    def S(A: np.ndarray, B: np.ndarray, lambda_: float) -> float:
        """
        Compute the similarity measure between two focal sets.
        
        Parameters
        ----------
        A : np.ndarray
            First focal set (binary array indicating cluster membership)
        B : np.ndarray
            Second focal set (binary array indicating cluster membership)
        lambda_ : float
            Lambda parameter controlling the similarity measure behavior
            
        Returns
        -------
        float
            Similarity measure between focal sets A and B
            
        Examples
        --------
        >>> A = np.array([1, 0, 1])
        >>> B = np.array([1, 1, 0])
        >>> Loss.S(A, B, 1.0)
        0.333...
        """
        num_intersection = np.sum(np.logical_and(A, B))
        num_union = np.sum(np.logical_or(A, B))

        # print(A)
        # print(B)

        if lambda_ > 0:
            if not num_intersection == np.sum(B):
                # if B is not a subset of A
                num_intersection = 0
        else:
            lambda_ = -lambda_
            if not num_intersection == np.sum(A):
                # if A is not a subset of B
                num_intersection = 0

        if lambda_ == 0:
            return np.floor(num_intersection / num_union)
        elif lambda_ == np.inf:
            return np.ceil(num_intersection / num_union)
        else:
            return (num_intersection / num_union) ** (1/lambda_)

    @staticmethod
    def mistakeness_missed(masses: np.ndarray, metaclusters: List[int], 
                          focal_sets: np.ndarray, lambda_: float) -> float:
        """
        Compute the mistakeness loss for missed assignments.
        
        Parameters
        ----------
        masses : np.ndarray
            Mass functions for each sample, shape (n_samples, n_focal_sets)
        metaclusters : List[int]
            Indices of metaclusters to consider
        focal_sets : np.ndarray
            Focal sets matrix, shape (n_focal_sets, n_clusters)
        lambda_ : float
            Lambda parameter for the loss function
            
        Returns
        -------
        float
            Total mistakeness loss for missed assignments
        """
        mistakeness = 0
        for A_idx in metaclusters:
            A = focal_sets[A_idx]
            mistakeness_A = 0
            for B_idx in range(len(focal_sets)):
                B = focal_sets[B_idx]
                S = Loss.S(A, B, lambda_)
                mistakeness_A += masses[:, B_idx] * S
            mistakeness += mistakeness_A 
        return np.sum(mistakeness)

    @staticmethod
    def mistakeness_assigned(masses: np.ndarray, metaclusters: List[int], 
                           focal_sets: np.ndarray, lambda_: float) -> float:
        """
        Compute the mistakeness loss for assigned samples.
        
        Parameters
        ----------
        masses : np.ndarray
            Mass functions for each sample, shape (n_samples, n_focal_sets)
        metaclusters : List[int]
            Indices of metaclusters to consider
        focal_sets : np.ndarray
            Focal sets matrix, shape (n_focal_sets, n_clusters)
        lambda_ : float
            Lambda parameter for the loss function
            
        Returns
        -------
        float
            Total mistakeness loss for assigned samples
        """
        mistakeness = 0
        for A_idx in metaclusters:
            A = focal_sets[A_idx]
            mistakeness_A = 0
            for B_idx in range(len(focal_sets)):
                B = focal_sets[B_idx]
                S = Loss.S(A, B, lambda_)
                mistakeness_A += masses[:, B_idx] * (1-S)/len(metaclusters)
            mistakeness += mistakeness_A 
        return np.sum(mistakeness)

class IEMM(BaseEstimator, ClassifierMixin):
    """
    Iterative Evidential Mistakeness Minimization (IEMM) classifier.
    
    IEMM is an explainable evidential clustering algorithm that generates interpretable
    decision tree explanations for evidential clustering functions. The algorithm is
    based on Dempster-Shafer theory and accounts for decision-maker preferences.
    
    Parameters
    ----------
    lambda_mistakeness : float, default=np.inf
        Lambda parameter controlling the mistakeness function behavior.
        When lambda_mistakeness > 0, uses mistakeness_missed function.
        When lambda_mistakeness < 0, uses mistakeness_assigned function.
        Default is np.inf (fuzzy mistakeness).
        
    Attributes
    ----------
    is_fitted : bool
        Whether the model has been fitted
    root_node : TreeNode
        Root node of the decision tree
    leafs : List[TreeNode]
        List of leaf nodes in the decision tree
    F : np.ndarray
        Focal sets matrix
    X : np.ndarray
        Training feature matrix
    mass : np.ndarray
        Mass functions for training samples
    pl : np.ndarray
        Plausibility values
    bel : np.ndarray
        Belief values
    number_attributes : int
        Number of features in the dataset
    attributes : np.ndarray
        Array of attribute indices
    metacluster_centroids : np.ndarray
        Centroids of metaclusters
        
    Examples
    --------
    >>> from iemm import IEMM
    >>> import numpy as np
    >>> # Create sample data
    >>> X = np.random.rand(100, 2)
    >>> mass = np.random.rand(100, 4)
    >>> F = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    >>> # Create and fit classifier
    >>> classifier = IEMM(lambda_mistakeness=1.0)
    >>> classifier.fit(X, mass, F)
    >>> predictions = classifier.predict(X)
    
    References
    ----------
    .. [1] Lopes de Souza, V. F., et al. "Explainable Evidential Clustering."
           arXiv preprint arXiv:2507.12192 (2025).
    """
    
    def __init__(self, lambda_mistakeness: float = np.inf) -> None:
        """
        Initialize the IEMM classifier.
        
        Parameters
        ----------
        lambda_mistakeness : float, default=np.inf
            Lambda parameter controlling the mistakeness function behavior
        """
        # state of the model
        self.is_fitted = False
        self.lambda_mistakeness = lambda_mistakeness

        # nodes
        self.root_node = TreeNode()
        self.leafs = []

    def get_metacluster_centroids(self) -> np.ndarray:
        """
        Calculate centroids for each metacluster based on mass-weighted features.
        
        Returns
        -------
        np.ndarray
            Array of metacluster centroids, shape (n_metaclusters, n_features)
        """
        metacluster_centroids = []
        for i in range(len(self.F)):
            if np.sum(self.F[i]) > 0:
                centroid = np.zeros(self.number_attributes)
                for point in range(self.X.shape[0]):
                    centroid += self.X[point] * self.mass[point, i]
                centroid /= np.sum(self.mass[:, i])
                metacluster_centroids.append(centroid)
        return np.array(metacluster_centroids)

    def fit(self, X: np.ndarray, mass: np.ndarray, F: np.ndarray) -> 'IEMM':
        """
        Fit the IEMM classifier to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix, shape (n_samples, n_features)
        mass : np.ndarray
            Mass functions for each sample, shape (n_samples, n_focal_sets)
        F : np.ndarray
            Focal sets matrix, shape (n_focal_sets, n_clusters)
            
        Returns
        -------
        self : IEMM
            Returns the instance itself for method chaining
            
        Examples
        --------
        >>> classifier = IEMM()
        >>> classifier.fit(X_train, mass_train, F)
        """
        # Save train dataset
        self.F = F
        self.X = X
        self.mass = mass
        self.pl = np.matmul(mass, F)
        self.bel = mass[:, np.sum(F, axis=1) == 1]

        # Save metacluster information
        self.number_attributes = self.X.shape[1]
        self.attributes = np.array(range(self.number_attributes))  
        self.metacluster_centroids = self.get_metacluster_centroids()

        # Construction of the tree
        self.root_node = TreeNode(
            contained_clusters=np.max(self.F, axis=0),
            mass=ibelief.DST(self.mass.T, 12).flatten()
        )
        self._build_tree(
            indices = np.array(range(self.X.shape[0])), 
            root_node = self.root_node, 
            metaclusters_idxs = np.array(range(len(self.F)))
        )

        # The model is now fitted
        self._fitted = True

        return self

    def _build_tree(self, indices, root_node, metaclusters_idxs):
        # Node depth
        node_depth = root_node.node_depth + 1

        # Find the best attribute et the treshold for continous values
        attribute, threshold, loss = self._best_gain(indices, metaclusters_idxs)

        # Stopping criteria
        if len(metaclusters_idxs) < 2:
            attribute = None
            if len(metaclusters_idxs) == 1:
                root_node.attributed_metacluster = metaclusters_idxs[0]

        if attribute != None: 
                left_condition = lambda x: np.where(x[:,attribute].astype(float) < threshold)[0]
                right_condition = lambda x: np.where(x[:,attribute].astype(float) >= threshold)[0]
                sides = {
                    1: left_condition,
                    2: right_condition
                }

                for side_i, side in sides.items():
                    metacluster_idxs = list(set(side(self.metacluster_centroids)) & set(metaclusters_idxs))
                    contained_clusters = np.max(self.F[metacluster_idxs], axis=0)
                    node = TreeNode(
                        attribute=attribute,
                        attribute_value=threshold,
                        continuous_attribute=side_i,
                        node_depth=node_depth,
                        contained_clusters=contained_clusters
                        )
                    self._build_tree(
                        indices=indices[side(self.X[indices])],
                        root_node=node,
                        metaclusters_idxs=metacluster_idxs)
                    root_node.leafs.append(node)
        else:
            # Append a mass if the node is a leaf
            root_node.is_leaf = True
            self.leafs.append(root_node)

        root_node.mass = ibelief.DST(self.mass[indices].T, 12).flatten()
        root_node.number_elements = self.mass[indices].shape[0]
        root_node.mistakeness_cut = loss
        root_node.F = self.F

    def _best_gain(self, indices, metacluster_idxs):
        thresholds = []
        losses = []

        # Interate over each attributes
        for attribute in self.attributes:
            # Find best split
            threshold, loss = self._find_treshold(indices, attribute, metacluster_idxs)

            thresholds.append(threshold)
            losses.append(loss)

        # Return the best attribute and the treshold for numerical attributes
        chosen_attribute = np.argmin(losses)
        threshold = thresholds[chosen_attribute]
        
        if threshold is None:
            return None, None, np.inf
        
        return chosen_attribute, threshold, losses[chosen_attribute]
    
    def _find_treshold(self, indices, attribute, metacluster_idxs):
        node_size = indices.shape[0]

        # Find uniques values for the attribute
        values = np.sort(np.unique(self.X[indices , attribute]).astype(float))

        # If there is only one value, return None
        if values.shape[0] < 2:
            return None, 1

        # Find all possible tresholds
        tresholds = []
        for i in range(values.shape[0] - 1):
            tresholds.append((values[i] + values[i + 1]) / 2)

        losses = np.zeros(len(tresholds))

        # For all tresholds, calculate the info gain
        for treshold in tresholds:
            left_condition = lambda x: np.where(x[:,attribute].astype(float) <= treshold)[0]
            right_condition = lambda x: np.where(x[:,attribute].astype(float) > treshold)[0]
            
            left_metacluster_idxs = list(set(left_condition(self.metacluster_centroids)) & set(metacluster_idxs))
            right_metacluster_idxs = list(set(right_condition(self.metacluster_centroids)) & set(metacluster_idxs))

            if len(left_metacluster_idxs) == 0 or len(right_metacluster_idxs) == 0:
                losses[tresholds.index(treshold)] = np.inf
            else:
                left_indices = indices[left_condition(self.X[indices])]
                right_indices = indices[right_condition(self.X[indices])]

                if self.lambda_mistakeness > 0:
                    mistakeness = Loss.mistakeness_missed
                    l_metaclusters = right_metacluster_idxs
                    r_metaclusters = left_metacluster_idxs
                else:
                    mistakeness = Loss.mistakeness_assigned
                    l_metaclusters = left_metacluster_idxs
                    r_metaclusters = right_metacluster_idxs

                # Compute the loss
                loss_left = mistakeness(
                    masses=self.mass[left_indices],
                    metaclusters=l_metaclusters,
                    focal_sets=self.F,
                    lambda_=self.lambda_mistakeness
                )
                loss_right = mistakeness(
                    masses=self.mass[right_indices],
                    metaclusters=r_metaclusters,
                    focal_sets=self.F,
                    lambda_=self.lambda_mistakeness
                )
                loss = (loss_left + loss_right) / node_size
                losses[tresholds.index(treshold)] = loss

        return tresholds[np.argmin(losses)], np.min(losses)

    

    def _predict(self, X, root_node):
        if root_node.is_leaf:
            return root_node.contained_clusters

        for v in root_node.leafs:
            if v.continuous_attribute == 3:
                if X[v.attribute].astype(float) >= v.attribute_value[0] and X[v.attribute].astype(float) < v.attribute_value[1]:
                    return self._predict(X, v)
            else:
                if v.continuous_attribute == 0 and X[v.attribute] == v.attribute_value:
                    return self._predict(X, v)
                elif v.continuous_attribute == 1 and X[v.attribute].astype(float) < v.attribute_value:
                    return self._predict(X, v)
                elif v.continuous_attribute == 2 and X[v.attribute].astype(float) >= v.attribute_value:
                    return self._predict(X, v)
        
        print("Classification Error, Tree not complete.")
        return None

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict metacluster assignments for new samples.
        
        Parameters
        ----------
        X : np.ndarray
            Test feature matrix, shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted metacluster indices, shape (n_samples,)
            
        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet
            
        Examples
        --------
        >>> predictions = classifier.predict(X_test)
        >>> print(predictions.shape)
        (n_test_samples,)
        """
        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        # Predict output bbas for X
        result = []
        for x in range(X.shape[0]):
            metacluster = self._predict(X[x], self.root_node)
            idx_metacluster = np.where(np.all(self.F == metacluster, axis=1))[0][0]
            result.append(idx_metacluster)

        result = np.array(result)

        return result

    def plot_tree(self,
                  feature_names: Optional[List[str]] = None,
                  class_names: Optional[List[str]] = None,
                  cluster_names: Optional[List[str]] = None,
                  focal_colors: Optional[np.ndarray] = None,
                  position: Tuple[float, float] = (0, 0),
                  x_spacing: float = 4,
                  y_spacing: float = 4,
                  box_width: float = 10,
                  box_height: float = 2,
                  box_scale: float = 1,
                  box_reduction: float = 0.9,
                  x_scale: float = 1,
                  x_reduction: float = 0.05,
                  fontsize: int = 16,
                  arrow_label: Optional[str] = None,
                  add_legend: bool = True) -> schemdraw.Drawing:
        """
        Plot the decision tree using schemdraw.
        
        Parameters
        ----------
        feature_names : List[str], optional
            Names of features to use in node labels
        class_names : List[str], optional
            Names of classes for legend
        cluster_names : List[str], optional
            Names of clusters for leaf node labels
        focal_colors : np.ndarray, optional
            Colors for focal sets, shape (n_focal_sets, 3)
        position : Tuple[float, float], default=(0, 0)
            Starting position for the root node
        x_spacing : float, default=4
            Horizontal spacing between nodes
        y_spacing : float, default=4
            Vertical spacing between levels
        box_width : float, default=10
            Width of node boxes
        box_height : float, default=2
            Height of node boxes
        box_scale : float, default=1
            Scaling factor for node boxes
        box_reduction : float, default=0.9
            Scale reduction for child nodes
        x_scale : float, default=1
            Horizontal scale factor
        x_reduction : float, default=0.05
            Scale reduction for child spacing
        fontsize : int, default=16
            Font size for node labels
        arrow_label : str, optional
            Label for arrows
        add_legend : bool, default=True
            Whether to add a legend
            
        Returns
        -------
        schemdraw.Drawing
            The tree diagram
            
        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet
            
        Examples
        --------
        >>> diagram = classifier.plot_tree(feature_names=['x1', 'x2'])
        >>> diagram.save('tree.png')
        """
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return self.root_node.plot_tree(
            feature_names=feature_names,
            class_names=class_names,
            cluster_names=cluster_names,
            focal_colors=focal_colors,
            position=position,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            box_width=box_width,
            box_height=box_height,
            box_scale=box_scale,
            box_reduction=box_reduction,
            x_scale=x_scale,
            x_reduction=x_reduction,
            fontsize=fontsize,
            arrow_label=arrow_label,
            add_legend=add_legend
        )
    

    def get_path(self, feature_names: Optional[List[str]] = None, 
                 focalset_names: Optional[List[str]] = None) -> Union[List, Dict]:
        """
        Get the decision paths from root to leaves.
        
        Parameters
        ----------
        feature_names : List[str], optional
            Names of features to use in path descriptions
        focalset_names : List[str], optional
            Names of focal sets for leaf labels
            
        Returns
        -------
        Union[List, Dict]
            Decision paths as strings or dictionary mapping focal sets to paths
            
        Examples
        --------
        >>> paths = classifier.get_path(feature_names=['x1', 'x2'])
        >>> print(paths)
        """
        # print(feature_names)
        # print(focalset_names)
        feature_labels = (lambda x: feature_names[x]) if feature_names else (lambda x: f"Feat{x}")
        return self.root_node.get_path(feature_labels=feature_labels, focalset_names=focalset_names)

class TreeNode():
    """
    A node in the evidential decision tree.
    
    This class represents a single node in the decision tree constructed by the IEMM algorithm.
    Each node can be either an internal node (with splitting criteria) or a leaf node
    (with final predictions).
    
    Parameters
    ----------
    mass : np.ndarray, optional
        Mass function associated with this node
    attribute : int, optional
        Index of the attribute used for splitting at this node
    attribute_value : Union[float, str], default=0
        Threshold value for continuous attributes or category for categorical attributes
    continuous_attribute : int, default=0
        Type of attribute: 0=categorical, 1=continuous (<), 2=continuous (>=), 3=range
    node_depth : int, default=0
        Depth of this node in the tree (root is 0)
    contained_clusters : np.ndarray, optional
        Binary array indicating which clusters are contained in this node
        
    Attributes
    ----------
    leafs : List[TreeNode]
        Child nodes of this node
    is_leaf : bool
        Whether this node is a leaf node
    number_elements : int
        Number of training samples that reach this node
    impurity_value : float, optional
        Impurity measure at this node
    impurity_type : str, optional
        Type of impurity measure used
    mistakeness_cut : float
        Mistakeness value for the split at this node
    F : np.ndarray
        Focal sets matrix
    attributed_metacluster : int, optional
        Index of the metacluster assigned to this leaf node
        
    Examples
    --------
    >>> node = TreeNode(attribute=0, attribute_value=5.0, continuous_attribute=1)
    >>> node.is_leaf = False
    >>> child = TreeNode()
    >>> node.leafs.append(child)
    """
    
    def __init__(self, 
                 mass: Optional[np.ndarray] = None, 
                 attribute: Optional[int] = None, 
                 attribute_value: Union[float, str] = 0, 
                 continuous_attribute: int = 0,
                 node_depth: int = 0,
                 contained_clusters: Optional[np.ndarray] = None) -> None:
        """
        Initialize a tree node.
        
        Parameters
        ----------
        mass : np.ndarray, optional
            Mass function for this node
        attribute : int, optional
            Index of the splitting attribute
        attribute_value : Union[float, str], default=0
            Value used for splitting (threshold for continuous, category for categorical)
        continuous_attribute : int, default=0
            Type of split: 0=categorical, 1=continuous (<), 2=continuous (>=), 3=range
        node_depth : int, default=0
            Depth of this node in the tree
        contained_clusters : np.ndarray, optional
            Binary array indicating contained clusters
        """

        self.leafs = []

        self.contained_clusters = contained_clusters

        self.mass = mass
        self.is_leaf = False
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.continuous_attribute = continuous_attribute
        self.node_depth = node_depth
        self.number_elements = 0
        self.impurity_value = None
        self.impurity_type = None
    
    def max_depth(self, depth: int = 1) -> int:
        """
        Calculate the maximum depth of the subtree rooted at this node.
        
        Parameters
        ----------
        depth : int, default=1
            Current depth (used for recursion)
            
        Returns
        -------
        int
            Maximum depth of the subtree
        """
        maximum_depth = []
        for i in self.leafs:
            maximum_depth.append(i.max_depth(depth=depth + 1))
        
        if len(self.leafs) == 0:
            return depth

        return np.max(np.array(maximum_depth))

    def plot_tree(
            self,
            feature_names,
            class_names,
            cluster_names,
            focal_colors,
            position,
            x_spacing,
            y_spacing,
            box_width,
            box_height,
            box_scale,
            box_reduction,
            x_scale,
            x_reduction,
            fontsize,
            arrow_label,
            add_legend,
            diagram=None,
            parent=None,):
        """
        Plots a decision tree using Schemdraw while managing overlaps, rescaling labels, and adding legends.

        Parameters:
        - feature_names (list[str], optional): List of feature names corresponding to attributes.
        - class_names (list[str], optional): List of class names for the matrix M.
        - diagram (schemdraw.Drawing, optional): Existing diagram to add nodes to. If None, a new diagram is created.
        - parent (schemdraw.element.Element, optional): Parent node to connect the current node.
        - position (tuple[float, float]): Position (x, y) of the current node in the diagram.
        - x_spacing (float): Horizontal spacing between child nodes.
        - y_spacing (float): Vertical spacing between levels.
        - box_width (float): Width of the node boxes.
        - box_height (float): Height of the node boxes.
        - box_scale (float): Scaling factor for node boxes.
        - box_reduction (float): Scaling reduction for child node boxes.
        - x_scale (float): Scaling factor for horizontal spacing.
        - x_reduction (float): Scaling reduction for child nodes.
        - fontsize (float): Base font size for node labels.
        - arrow_label (str, optional): Label for the arrow connecting to the current node.
        - add_legend (bool): Whether to include a legend for the class names.

        Returns:
        - schemdraw.Drawing: The updated diagram object.
        """
        if diagram is None:
            diagram = schemdraw.Drawing(unit=1)

        # Generate node text
        # metacluster_prediction = cluster_names#"$"+" \\cup ".join([cluster_names[i+1] for i in range(len(self.contained_clusters)) if self.contained_clusters[i] == 1]) + "$"
        # text = f"{self.number_elements} samples\nmean of belief masses: ${np.around(self.mass, decimals=2).tolist()}"#$\nbest label: {metacluster_prediction}"
        text = f"{self.number_elements} samples\n"
        if self.mistakeness_cut < np.inf:
            text += f"mistakeness of cut: {np.around(self.mistakeness_cut, decimals=2)}"
        else:
            # print(cluster_names)
            # print(self.F)
            # print(self.contained_clusters)
            is_idx = (self.F == self.contained_clusters).all(axis=1)
            metacluster_prediction = cluster_names[np.where(is_idx)[0][0]]
            # print(metacluster_prediction)
            text += f"label: {metacluster_prediction}"
            None
        
        #pl = np.matmul(self.mass, self.F)
        #bel = self.mass[:, np.sum(self.F, axis=1) == 1]

        # Generate node color
        if focal_colors is not None:
            color = tuple(np.dot(self.mass.reshape(1, -1), focal_colors).clip(0, 1).flatten().tolist()[:3])
        else:
            color = 'white'
        diagram += (node := flow.Box(
            h=box_height * box_scale,
            w=box_width * box_scale
        ).fill(
            color
        ).at(position).anchor("center").label(
            text,
            fontsize = fontsize * box_scale
        ))

        # Draw arrow from parent
        if parent is not None:
            diagram += flow.Arrow().at(parent.S).to(node.N).label(
                arrow_label,
                fontsize=fontsize * box_scale / 1.5,
                rotate=True,
                loc="bottom",
            )

        # Determine child positions
        num_children = len(self.leafs)
        if num_children > 0:
            total_width = (num_children - 1) * (x_spacing + box_width * box_scale) # Total width of child nodes
            child_positions = [
                (position[0] - total_width / 2 + i * (x_spacing + box_width * box_scale), position[1] - y_spacing)
                for i in range(num_children)
            ]

        # Feature name accessor
        feature_label = (lambda x: feature_names[x]) if feature_names else (lambda x: f"Feat{x}")

        # Plot child nodes
        for idx, child in enumerate(self.leafs):
            # Generate arrow label
            if child.continuous_attribute == 1:
                arrow_label = f"{feature_label(child.attribute)} ≤ {child.attribute_value:.2f}"
            elif child.continuous_attribute == 2:
                arrow_label = f"{feature_label(child.attribute)} > {child.attribute_value:.2f}"
            elif child.continuous_attribute == 3:
                arrow_label = f"{feature_label(child.attribute)} ∈ [{child.attribute_value[0]:.2f}, {child.attribute_value[1]:.2f})"
            else:
                arrow_label = f"{feature_label(child.attribute)} = {child.attribute_value:.2f}"

            # Recursively plot child
            diagram = child.plot_tree(
                feature_names=feature_names,
                class_names=class_names,
                cluster_names=cluster_names,
                focal_colors=focal_colors,
                diagram=diagram,
                parent=node,
                position=child_positions[idx],
                x_spacing=x_spacing * x_reduction,
                y_spacing=y_spacing,
                box_width=box_width,
                box_height=box_height,
                box_scale=box_scale * box_reduction,
                box_reduction=box_reduction,
                x_scale=x_scale * x_reduction,
                x_reduction=x_reduction,
                fontsize=fontsize,
                arrow_label=arrow_label,
                add_legend=False  # Legend added only once
            )

        # Add legend for class names, if applicable
        # if add_legend and class_names:
        #     legend_text = "$M = [$" + ", ".join(
        #         class_names
        #     ) + "$]$"
        #     diagram += flow.Box(
        #         h=box_height * box_scale,
        #         w=box_width * box_scale
        #     ).at((position[0], position[1] + box_height * box_scale)).anchor("center").label(
        #         legend_text,
        #         fontsize=fontsize * box_scale * box_reduction
        #     ).linewidth(0)
                
        return diagram

    def add_leaf(self, node: 'TreeNode') -> None:
        """
        Add a child node to this node.

        Parameters
        ----------
        node : TreeNode
            Child node to add
        """
        self.leafs.append(node)

    def get_path(self, past_path = [], feature_labels=None, focalset_names=None):
        """
        Get the path of the node

        Returns
        -----
        path : list
            List of nodes
        """

        paths = []
        for leaf in self.leafs:
            if leaf.continuous_attribute == 1:
                #path = past_path + [f"${feature_labels(leaf.attribute)} \leq {leaf.attribute_value:.2f}$"]
                #leq case
                path = past_path + [(feature_labels(leaf.attribute), leaf.attribute_value, 1)]
            elif leaf.continuous_attribute == 2:
                #path = past_path + [f"${feature_labels(leaf.attribute)} > {leaf.attribute_value:.2f}$"]
                #greater case
                path = past_path + [(feature_labels(leaf.attribute), leaf.attribute_value, 2)]
            
            new_path = leaf.get_path(path, feature_labels, focalset_names)
            # if there is a dict, in the list new_path
            if any(isinstance(i, dict) for i in new_path):
                paths.extend(new_path)
            else:
                paths.append(new_path)

        if len(paths) == 0:
            # simplify the path, if an attribute has two conditions of the same type, it is not necessary to show both
            past_path_df = pd.DataFrame(past_path, columns=["attribute", "value", "condition"])

            for attribute in past_path_df.attribute.unique():
                for condition in past_path_df[past_path_df.attribute == attribute].condition.unique():
                    if len(past_path_df[(past_path_df.attribute == attribute) & (past_path_df.condition == condition)]) > 1:
                        # if leq case is present, select the smallest value
                        if condition == 1:
                            min_value = past_path_df[(past_path_df.attribute == attribute) & (past_path_df.condition == condition)].value.min()
                            past_path_df = past_path_df.drop(past_path_df[(past_path_df.attribute == attribute) & (past_path_df.condition == condition) & (past_path_df.value != min_value)].index)
                        # if greater case is present, select the greatest value
                        elif condition == 2:
                            max_value = past_path_df[(past_path_df.attribute == attribute) & (past_path_df.condition == condition)].value.max()
                            past_path_df = past_path_df.drop(past_path_df[(past_path_df.attribute == attribute) & (past_path_df.condition == condition) & (past_path_df.value != max_value)].index)

            # format the path as string
            strs_out = []
            for _, row in past_path_df.iterrows():
                if row.condition == 1:
                    strs_out.append(f"({row.attribute} \\leq {row.value:.2f})")
                elif row.condition == 2:
                    strs_out.append(f"({row.attribute} > {row.value:.2f})")

            str_out = " \\wedge ".join(strs_out)
            str_out = f"${str_out}$"

            if focalset_names is not None:
                return {focalset_names[self.attributed_metacluster] : str_out}
            else:
                return {self.attributed_metacluster : str_out}
        
        return paths