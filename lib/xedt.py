"""
eXplanatory Evidential Decision Tree

Adapted from https://github.com/ArthurHoa/conflict-edt/tree/master (Conflict EDT, Arthur Hoarau, 02/01/2025).

Author: Victor Souza
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

import numpy as np
import pandas as pd

import schemdraw
import schemdraw.flow as flow

from lib import ibelief

class Loss:    
    @staticmethod
    def S(A,B, lambda_, cautious_subset=True):
        num_intersection = np.sum(np.logical_and(A, B))
        num_union = np.sum(np.logical_or(A, B))

        # print(A)
        # print(B)

        if cautious_subset:
            if not num_intersection == np.sum(B):
                # if B is not a subset of A
                num_intersection = 0

        if lambda_ == np.inf:
            return np.floor(num_intersection / num_union)
        elif lambda_ == 0:
            return np.ceil(num_intersection / num_union)
        else:
            return (num_intersection / num_union) ** lambda_

    @staticmethod
    def mistakeness(masses, missed_metaclusters, focal_sets, lambda_):
        mistakeness = 0
        for A_idx in missed_metaclusters:
            A = focal_sets[A_idx]
            mistakeness_A = 0
            for B_idx in range(len(focal_sets)):
                B = focal_sets[B_idx]
                S = Loss.S(A, B, lambda_)
                mistakeness_A += masses[:, B_idx] * S
            mistakeness += mistakeness_A 
        return np.mean(mistakeness)

class XEDT(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 lambda_mistakeness : float = np.inf, # default is the fuzzy mistakeness
                 ):

        # state of the model
        self.is_fitted = False
        self.lambda_mistakeness = lambda_mistakeness

        # nodes
        self.root_node = TreeNode()
        self.leafs = []

    def get_metacluster_centroids(self):
        metacluster_centroids = []
        for i in range(len(self.F)):
            if np.sum(self.F[i]) > 0:
                centroid = np.zeros(self.number_attributes)
                for point in range(self.X.shape[0]):
                    centroid += self.X[point] * self.mass[point, i]
                centroid /= np.sum(self.mass[:, i])
                metacluster_centroids.append(centroid)
        return np.array(metacluster_centroids)

    def fit(self, X, mass, F):
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
                
                left_size = left_indices.shape[0]
                right_size = right_indices.shape[0]

                # Compute the loss
                loss_left = Loss.mistakeness(
                    masses=self.mass[left_indices],
                    missed_metaclusters=right_metacluster_idxs,
                    focal_sets=self.F,
                    lambda_=self.lambda_mistakeness
                )
                loss_right = Loss.mistakeness(
                    masses=self.mass[right_indices],
                    missed_metaclusters=left_metacluster_idxs,
                    focal_sets=self.F,
                    lambda_=self.lambda_mistakeness
                )
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

        if root_node.is_leaf:
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

    def _predict_metacluster(self, X, root_node):
        if root_node.is_leaf:
            return root_node.contained_clusters

        for v in root_node.leafs:
            if v.continuous_attribute == 3:
                if X[v.attribute].astype(float) >= v.attribute_value[0] and X[v.attribute].astype(float) < v.attribute_value[1]:
                    return self._predict_metacluster(X, v)
            else:
                if v.continuous_attribute == 0 and X[v.attribute] == v.attribute_value:
                    return self._predict_metacluster(X, v)
                elif v.continuous_attribute == 1 and X[v.attribute].astype(float) < v.attribute_value:
                    return self._predict_metacluster(X, v)
                elif v.continuous_attribute == 2 and X[v.attribute].astype(float) >= v.attribute_value:
                    return self._predict_metacluster(X, v)
        
        print("Classification Error, Tree not complete.")
        return None



    def predict(self, X):
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
        result = np.zeros((X.shape[0], self.mass.shape[1]))
        for x in range(X.shape[0]):
            result[x] = self._predict(X[x], self.root_node)

        return result
    
    def predict_metacluster(self, X):
        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        # Predict output bbas for X
        result = []
        for x in range(X.shape[0]):
            metacluster = self._predict_metacluster(X[x], self.root_node)
            idx_metacluster = np.where(np.all(self.F == metacluster, axis=1))[0][0]
            result.append(idx_metacluster)

        result = np.array(result)

        return result

    def plot_tree(self,
                  feature_names=None,
                  class_names=None,
                  cluster_names=None,
                  focal_colors=None,
                  position=(0, 0),
                  x_spacing=4,
                  y_spacing=3,
                  box_width=10,
                  box_height=2,
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
    

    def get_path(self, feature_names=None, focalset_names=None):
        # print(feature_names)
        # print(focalset_names)
        feature_labels = (lambda x: feature_names[x]) if feature_names else (lambda x: f"Feat{x}")
        return self.root_node.get_path(feature_labels=feature_labels, focalset_names=focalset_names)

class TreeNode():
    def __init__(self, 
                 mass = None, 
                 attribute = None, 
                 attribute_value = 0, 
                 continuous_attribute = 0,
                 node_depth = 0,
                 contained_clusters = None):
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
    
    def max_depth(self, depth=1):
        maximum_depth = []
        for i in self.leafs:
            maximum_depth.append(i.max_depth(depth=depth + 1))
        
        if len(self.leafs) == 0:
            return depth

        return np.max(np.array(maximum_depth))

    def plot_tree(
            self,
            feature_names=None,
            class_names=None,
            cluster_names=None,
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

        metacluster_prediction = cluster_names#"$"+" \\cup ".join([cluster_names[i+1] for i in range(len(self.contained_clusters)) if self.contained_clusters[i] == 1]) + "$"

        # Generate node text
        text = f"{self.number_elements} samples\nmean of belief masses: ${np.around(self.mass, decimals=2).tolist()}"#$\nbest label: {metacluster_prediction}"
        text += f"\nmistakeness of: {np.around(self.mistakeness_cut, decimals=2)}"
        
        #pl = np.matmul(self.mass, self.F)
        #bel = self.mass[:, np.sum(self.F, axis=1) == 1]

        # Generate node color
        if focal_colors is not None:
            color = np.dot(self.mass.reshape(1, -1), focal_colors).clip(0, 1).flatten().tolist()
        else:
            color = 'white'
        color = 'white'
        # print(f"Node color: {color}")
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
                    strs_out.append(f"({row.attribute} \leq {row.value:.2f})")
                elif row.condition == 2:
                    strs_out.append(f"({row.attribute} > {row.value:.2f})")

            str_out = " \\wedge ".join(strs_out)
            str_out = f"${str_out}$"

            if focalset_names is not None:
                return {focalset_names[self.attributed_metacluster] : str_out}
            else:
                return {self.attributed_metacluster : str_out}
        
        return paths