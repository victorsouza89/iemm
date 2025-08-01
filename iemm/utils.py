"""
Utility functions and classes for the IEMM library.

This module provides utility functions for distance calculations, criteria computation,
and other helper functions used throughout the IEMM algorithm.

Author: Victor Souza
"""

import numpy as np
import math
from typing import Optional, Union, List, Tuple
from . import belief as ibelief

class Utils:
    """
    Utility functions for evidential clustering computations.
    
    This class provides static methods for various utility computations
    used in evidential clustering and decision tree construction.
    """
    
    @staticmethod
    def jaccard_matrix_calculus(number_metaclasses: int) -> np.ndarray:
        """
        Compute the Jaccard similarity matrix for the discernment framework.
        
        This function calculates the Jaccard similarity between all pairs of
        focal elements in a discernment framework with the given number of metaclasses.
        
        Parameters
        ----------
        number_metaclasses : int
            Number of metaclasses in the discernment framework
            
        Returns
        -------
        np.ndarray
            Jaccard similarity matrix, shape (number_metaclasses, number_metaclasses)
            
        Examples
        --------
        >>> jaccard_matrix = Utils.jaccard_matrix_calculus(4)
        >>> print(jaccard_matrix.shape)
        (4, 4)
        
        Notes
        -----
        The function assumes a power set structure where number_metaclasses = 2^n
        for some number of atoms n.
        """
        natoms = round(np.log2(number_metaclasses))
        ind = [{}]*number_metaclasses
        if (np.pow(2, natoms) == number_metaclasses):
            ind[0] = {0} #In fact, the first element should be a None value (for empty set).
            #But in the following calculate, we'll deal with 0/0 which shoud be 1 bet in fact not calculable. So we "cheat" here to make empty = {0}
            ind[1] = {1}
            step = 2
            while (step < number_metaclasses):
                ind[step] = {step}
                step = step+1
                indatom = step
                for step2 in range(1,indatom - 1):
                    ind[step] = (ind[step2] | ind[indatom-1])
                    step = step+1
        out = np.zeros((number_metaclasses,number_metaclasses))

        for i in range(number_metaclasses):
            for j in range(number_metaclasses):
                out[i][j] = float(len(ind[i] & ind[j]))/float(len(ind[i] | ind[j]))
        return out

class Criterion:
    """
    Criterion functions for decision tree splitting.
    
    This class provides methods for computing various criteria used in
    evidential decision tree construction, including conflict measures
    and uncertainty measures.
    """
    
    @staticmethod
    def conflict(m1: np.ndarray, m2: np.ndarray) -> float:
        """
        Compute the conflict measure between two mass functions.
        
        Parameters
        ----------
        m1 : np.ndarray
            First mass function
        m2 : np.ndarray
            Second mass function
            
        Returns
        -------
        float
            Conflict measure between the mass functions
            
        Examples
        --------
        >>> m1 = np.array([0.6, 0.3, 0.1])
        >>> m2 = np.array([0.4, 0.4, 0.2])
        >>> conflict = Criterion.conflict(m1, m2)
        """
        return np.sum(np.abs(m1 - m2))
    
    def _compute_info(self, indices):
        if indices.shape[0] == 0 or indices.shape[0] == 1:
            return 0

        # Choice of the split criterion
        if self.criterion == 'conflict' or self.criterion == 'jousselme' or self.criterion == "euclidian":
            info = self._compute_distance(indices)
        if self.criterion == 'uncertainty':
            info = self._compute_uncertainty(indices)

        return info
    
    # Jousselme distance
    def _compute_distance(self, indices):
        divisor = indices.shape[0]**2 - indices.shape[0]

        mean_distance = np.sum(self.distances[indices][:,indices]) / divisor        

        return mean_distance   

    def _compute_inclusion_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                d_inc = self._compute_inclusion_degree(self.y_trained[i], self.y_trained[j])
                distances[i,j] = (1 - d_inc) * math.sqrt(np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix), self.y_trained[i]-self.y_trained[j])/2.0)

        return distances

    def _compute_jousselme_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                distances[i,j] = math.sqrt(np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix), self.y_trained[i]-self.y_trained[j])/2.0)

        return distances

    def _compute_euclidian_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                distances[i,j] = math.dist(self.y_trained[i], self.y_trained[j])

        return distances

    def _compute_prignistic_prob(self):
        size = self.y_trained.shape[0]

        pign_prob =  np.zeros((size, self.y_trained.shape[1]))
        elemets_size = np.zeros(self.y_trained.shape[1])

        for k in range(size): 
            betp_atoms = ibelief.decisionDST(self.y_trained[k].T, 4, return_prob=True)[0]
            for i in range(1, self.y_trained.shape[1]):
                for j in range(betp_atoms.shape[0]):
                        if ((2**j) & i) == (2**j):
                            pign_prob[k][i] += betp_atoms[j]

        for i in range(1, self.y_trained.shape[1]):
            elemets_size[i] = math.log2(bin(i).count("1"))

        return pign_prob, elemets_size
    
    def _compute_inclusion_degree(self, m1, m2): 
        m1 = m1[:-1]
        m2 = m2[:-1]
        n1 = np.where(m1 > 0)[0]
        n2 = np.where(m2 > 0)[0]

        # If total ignorance, degree is one
        if n1.shape[0] == 0 or n2.shape[0] == 0:
            return 1

        d_inc_l = 0
        d_inc_r = 0
        
        for X1 in n1:
            for X2 in n2:
                if X1 & X2 == X1:
                    d_inc_l += 1
                if X1 & X2 == X2:
                    d_inc_r += 1

        return (1 / (n1.shape[0] * n2.shape[0])) * max(d_inc_r, d_inc_l)
    

    
