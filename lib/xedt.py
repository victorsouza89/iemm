"""
eXplanatory Evidential Decision Tree

Adapted from https://github.com/ArthurHoa/conflict-edt/tree/master (Conflict EDT, Arthur Hoarau, 02/01/2025).

Author: Victor Souza
"""

from typing import List, Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

import numpy as np

import schemdraw
import schemdraw.flow as flow

from lib import ibelief

class Loss:
    @staticmethod
    def define_incoherenceness(bel, pl):
        assert bel.shape == pl.shape
        
        number_observations = bel.shape[0]
        #number_clusters = bel.shape[1]

        incoherenceness = np.zeros(int(number_observations*(number_observations + 1) / 2))
        get_index = lambda i, j: int(j + i * (number_observations - (i + 1) / 2))

        for i in range(number_observations):
            for j in range(number_observations):
                if not i  > j:
                    index = get_index(i, j)
                    incoherenceness[index] = np.sum(bel[i, :] * (1 - pl[j, :]) + bel[j, :] * (1 - pl[i, :]))/2

        def calc_incoherenceness(indices):
            loss = 0
            n = 0
            for i in indices:
                for j in indices:
                    if not i > j:
                        loss += incoherenceness[get_index(i, j)]
                        n += 1
            loss = loss / n
            return loss

        return calc_incoherenceness
    
    @staticmethod
    def mistakeness(masses, present_metacluster_idxs, missing_metacluster_idxs, powered_jaccard_matrix):
        powered_jaccard_matrix_ = powered_jaccard_matrix[missing_metacluster_idxs, :][:, present_metacluster_idxs]
        mistakeness_per_element = np.matmul(masses[:, missing_metacluster_idxs], 1-powered_jaccard_matrix_)
        mistakeness_per_element = np.sum(mistakeness_per_element, axis=1)
        return np.mean(mistakeness_per_element)

class XEDT(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 lambda_mistakeness = np.inf, # default is the fuzzy mistake
                 # the following parameters are stopping criteria
                 min_samples_per_leaf : int = 0,
                 max_depth : int = None,
                 impurity_treshold : float = None):

        # state of the model
        self.is_fitted = False
        self.lambda_mistakeness = lambda_mistakeness

        # nodes
        self.root_node = TreeNode()
        self.leafs = []

        # stopping criteria
        self.min_samples_per_leaf = min_samples_per_leaf
        self.max_depth = max_depth
        self.impurity_treshold = impurity_treshold

    def precompute_incoherenceness(self):
            return Loss.define_incoherenceness(self.bel, self.pl)
    
    def precompute_mistakeness(self):
        number_metaclusters, number_clusters = self.F.shape

        powered_jaccard_matrix = np.ones((number_metaclusters, number_metaclusters))
        if self.lambda_mistakeness == np.inf:
            powered_jaccard_matrix = np.eye(number_metaclusters)
        else:
            for i in range(number_metaclusters):
                for j in range(number_metaclusters):
                    # get indexes that have 1
                    get_focal_idxs = lambda x: set(np.where(self.F[x] == 1)[0])
                    f_i = get_focal_idxs(i)
                    f_j = get_focal_idxs(j)

                    try:
                        if not self.lambda_mistakeness == 0:
                            powered_jaccard_matrix[i, j] = (len(f_i & f_j)/len(f_i | f_j))**self.lambda_mistakeness
                        else:
                            powered_jaccard_matrix[i, j] = np.ceil(len(f_i & f_j)/len(f_i | f_j))
                    except ZeroDivisionError:
                        pass
        
        def mistakeness(indices, present_metacluster_idxs, missing_metacluster_idxs):
            missing_metacluster_idxs = list(np.array(missing_metacluster_idxs) + 1)
            present_metacluster_idxs = list(np.array(present_metacluster_idxs) + 1)

            return Loss.mistakeness(self.mass_train[indices], present_metacluster_idxs, missing_metacluster_idxs, powered_jaccard_matrix)
        
        return mistakeness


    def fit(self, X, mass, F):
        # Save train dataset
        self.X_train = X
        self.mass_train = mass
        self.size_train = self.X_train.shape[0]

        self.number_attributes = self.X_train.shape[1]
        self.attributes = np.array(range(self.number_attributes))
        X_indices = np.array(range(self.size_train))
        
        self.pl = np.matmul(mass, F)
        self.bel = mass[:, np.sum(F, axis=1) == 1]

        self.F = F

        self.metacluster_centroids = []
        for i in range(len(F)):
            if np.sum(F[i]) > 0:
                centroid = np.zeros(self.number_attributes)
                for point in range(self.size_train):
                    centroid += self.X_train[point] * mass[point, i]
                centroid /= np.sum(mass[:, i])
                self.metacluster_centroids.append(centroid)
        self.metacluster_centroids = np.array(self.metacluster_centroids)
        
        # precompute losses
        self.incoherenceness = self.precompute_incoherenceness()
        self.mistakeness = self.precompute_mistakeness()

        # Construction of the tree
        self.root_node = TreeNode()
        self._build_tree(X_indices, self.root_node, metaclusters_idxs = range(len(self.F)-1))

        # The model is now fitted
        self._fitted = True

        return self

    def _build_tree(self, indices, root_node, metaclusters_idxs):
        # Node impurity
        root_node.impurity_value = self.incoherenceness(indices)
        
        # Node depth
        node_depth = root_node.node_depth + 1

        # Find the best attribute et the treshold for continous values
        attribute, threshold, loss = self._best_gain(indices, metaclusters_idxs)

        # Stopping criteria
        if self.max_depth is not None:
            if node_depth >= self.max_depth:
                attribute = None
        if self.impurity_treshold is not None:
            if root_node.impurity_value < self.impurity_treshold:
                attribute = None

        if len(metaclusters_idxs) < 2:
            attribute = None

        if attribute != None: 
                # Left node
                left_condition = lambda x: np.where(x[:,attribute].astype(float) < threshold)[0]
                left_metaclusters_idx = list(set(left_condition(self.metacluster_centroids)) & set(metaclusters_idxs))
                node = TreeNode(attribute=attribute, attribute_value=threshold, continuous_attribute=1, node_depth=node_depth)
                self._build_tree(indices[left_condition(self.X_train[indices])], node, metaclusters_idxs=left_metaclusters_idx)
                root_node.leafs.append(node) 
                
                # Right node
                right_condition = lambda x: np.where(x[:,attribute].astype(float) >= threshold)[0]
                right_metaclusters_idx = list(set(right_condition(self.metacluster_centroids)) & set(metaclusters_idxs))
                node = TreeNode(attribute=attribute, attribute_value=threshold, continuous_attribute=2, node_depth=node_depth)
                self._build_tree(indices[right_condition(self.X_train[indices])], node, metaclusters_idxs=right_metaclusters_idx)
                root_node.leafs.append(node)
        else:
            # Append a mass if the node is a leaf
            root_node.mass = ibelief.DST(self.mass_train[indices].T, 12).flatten()
            root_node.number_leaf = self.mass_train[indices].shape[0]
            self.leafs.append(root_node)
        
        root_node.node_mass = ibelief.DST(self.mass_train[indices].T, 12).flatten()
        root_node.number_elements = self.mass_train[indices].shape[0]
        root_node.impurity_type = "incoherenceness"
        root_node.loss_cut = loss
        root_node.loss_type = "mistakeness"

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
            return None, None
        
        return chosen_attribute, threshold, losses[chosen_attribute]
    
    def _find_treshold(self, indices, attribute, metacluster_idxs):
        node_size = indices.shape[0]

        # Find uniques values for the attribute
        values = np.sort(np.unique(self.X_train[indices , attribute]).astype(float))

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
                left_indices = indices[left_condition(self.X_train[indices])]
                right_indices = indices[right_condition(self.X_train[indices])]
                
                left_size = left_indices.shape[0]
                right_size = right_indices.shape[0]

                # Compute the loss
                loss_left = self.mistakeness(left_indices, left_metacluster_idxs, right_metacluster_idxs)
                loss_right = self.mistakeness(right_indices, right_metacluster_idxs, left_metacluster_idxs)
                loss = (loss_left*left_size + loss_right*right_size) / node_size
                losses[tresholds.index(treshold)] = loss

        return tresholds[np.argmin(losses)], np.min(losses)

    def _predict(self, X, root_node):
        """
        predict bbas on the input.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        result : ndarray
            Array of normalized bba
        """

        if type(root_node.mass) is np.ndarray:
            return root_node.mass

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



    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        # Predict output bbas for X
        result = np.zeros((X.shape[0], self.mass_train.shape[1]))
        for x in range(X.shape[0]):
            result[x] = self._predict(X[x], self.root_node)

        # Max Plausibility
        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        # Return predictions or both predictions and bbas
        if return_bba:
            return predictions, result
        else:
            return predictions
    
    def plot_tree(self,
                  feature_names=None,
                  class_names=None,
                  focal_colors=None,
                  position=(0, 0),
                  x_spacing=4,
                  y_spacing=3,
                  box_width=10,
                  box_height=1.5,
                  box_scale=1,
                  box_reduction=0.75,
                  x_scale=1,
                  x_reduction=0.1,
                  fontsize=12,
                  arrow_label=None,
                  add_legend=True):

        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return self.root_node.plot_tree(
            feature_names=feature_names,
            class_names=class_names,
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

class TreeNode():
    def __init__(self, 
                 mass = None, 
                 attribute = None, 
                 attribute_value = 0, 
                 continuous_attribute = 0, 
                 number_leaf = 0, 
                 node_depth = 0):
        """
        Tree node class used in the Evidential Decision Tree

        Parameters
        -----
        mass : BBA
            Mass of the node
        attribute : int
            indice of the attribute
        attribute_value : float/string
            value of the attribute (string for categorical, float for numerical)
        continuous_attribute : int
            0: categorical, 1: numerical and <, 2: numerical and >=.
        number_leaf : int
            number of elements in the leaf

        Returns
        -----
        The instance of the class.
        """

        self.leafs = []

        self.mass = mass
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.continuous_attribute = continuous_attribute
        self.number_leaf = number_leaf
        self.node_depth = node_depth
        self.number_elements = 0
        self.node_mass = None
        self.impurity_value = None
        self.impurity_type = None
    
    def max_depth(self, depth=1):
        maximum_depth = []
        for i in self.leafs:
            maximum_depth.append(i.max_depth(depth=depth + 1))
        
        if len(self.leafs) == 0:
            return depth

        return np.max(np.array(maximum_depth))

    def mean_samples_leafs(self):
        samples = []

        for i in self.leafs:
            childs = i.mean_samples_leafs()

            if isinstance(childs, int):
                samples.append(childs)
            else:
                for j in childs:
                    samples.append(j)
        
        if len(self.leafs) == 0:
            return self.number_leaf

        return samples

    def plot_tree(
            self,
            feature_names=None,
            class_names=None,
            focal_colors=None,
            diagram=None,
            parent=None,
            position=(0, 0),
            x_spacing=4,
            y_spacing=2,
            box_width=8,
            box_height=1,
            box_scale=1,
            box_reduction=0.75,
            x_scale=1,
            x_reduction=0.1,
            fontsize=12,
            arrow_label=None,
            add_legend=True):
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
        text = f"{self.number_elements} samples\nmean of belief masses: ${np.around(self.node_mass, decimals=2).tolist()}$\nnode impurity ({self.impurity_type}): {np.around(self.impurity_value, decimals=2)}"
        if not self.loss_cut == np.inf:
            text += f"\nloss of cut ({self.loss_type}): {np.around(self.loss_cut, decimals=2)}"
        #text = f"{self.number_elements} samples\n$M = {np.around(self.node_mass, decimals=2).tolist()}$\nnode impurity ({self.impurity_type}): {np.around(self.impurity_value, decimals=2)}\ncut loss ({self.loss_type}): {np.around(self.loss_cut, decimals=2)}"

        # Generate node color
        if focal_colors is not None:
            color = np.dot(self.node_mass.reshape(1, -1), focal_colors).clip(0, 1).flatten().tolist()
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
        if add_legend and class_names:
            legend_text = "$M = [$" + ", ".join(
                class_names
            ) + "$]$"
            diagram += flow.Box(
                h=box_height * box_scale,
                w=box_width * box_scale
            ).at((position[0], position[1] + box_height * box_scale)).anchor("center").label(
                legend_text,
                fontsize=fontsize * box_scale * box_reduction
            ).linewidth(0)
                
        return diagram

    def add_leaf(self, node):
        """
        Add leaf to the node

        Parameters
        -----
        node : TreeNode
            Node
        """

        self.leafs.append(node)