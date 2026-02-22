"""
core.py

Author: natelgrw
Last Edited: 01/15/2026

Utility classes for design representation and encoding in optimization algorithms.
Provides ID encoding for design parameter vectors and Design class for 
managing circuit designs with evolutionary properties.
"""

import numpy as np
import sys
from copy import deepcopy, copy


# ===== Design ID Encoder ===== #


class IDEncoder(object):
    """
    Encodes design parameter vectors to unique alphanumeric IDs.

    Uses base conversion to map design parameter indices to unique identifiers.
    For example, parameter indices [1, 2, 3] with bases [10, 3, 8] are converted
    to a compact base-62 string ID: '0k'.
    
    Initialization Parameters:
    --------------------------
    params_vec : dict
        Dictionary mapping parameter names to lists of discrete values.
        The length of each list defines the base for that parameter.
    """

    def __init__(self, params_vec):
        
        # extract base for each parameter from vector lengths
        self._bases = np.array([len(vec) for vec in params_vec.values()])
        self._mulipliers = self._compute_multipliers()
        self._lookup_dict = self._create_lookup()
        self.length_letters = self._compute_length_letter()

    def _compute_length_letter(self):
        """
        Computes the total number of letter positions needed for ID encoding.

        Calculates the maximum possible ID value and determines how many
        base-62 digits are needed to represent it.

        Returns:
        --------
        int
            Number of base-62 digits needed for all possible IDs.
        """
        # compute total search space size
        cur_multiplier = 1
        for base in self._bases:
            cur_multiplier = cur_multiplier * base

        max_number = cur_multiplier
        ret_vec = self._convert_2_base_letters(max_number)
        return len(ret_vec)

    @classmethod
    def _create_lookup(cls):
        """
        Creates lookup table for base-62 character encoding.

        Maps digits 0-61 to characters: 0-9, a-z, A-Z for compact ID representation.

        Returns:
        --------
        dict
            Lookup dictionary mapping integers to base-62 character strings.
        """
        lookup = {}
        n_letters = ord('z') - ord('a') + 1

        # map 0-9
        for i in range(10):
            lookup[i] = str(i)

        # map 10-35 to lowercase letters a-z
        for i in range(n_letters):
            lookup[i + 10] = chr(ord('a') + i)

        # map 36-61 to uppercase letters A-Z
        for i in range(n_letters):
            lookup[i + 10 + n_letters] = chr(ord('A') + i)

        return lookup

    def _compute_multipliers(self):
        """
        Computes multipliers for positional weighted encoding.

        Converts parameter index vectors to unique base-10 integers using
        positional notation similar to number base conversion.
        
        For example, bases [10, 3, 8] generates multipliers [3x8, 8, 0].

        Returns:
        --------
        numpy array
            Multiplier array for weighted position encoding.
        
        Raises:
        -------
        AssertionError
            If search space exceeds machine precision limits.
        """
        cur_multiplier = 1
        ret_list = []
        
        # compute multipliers in reverse order
        for base in self._bases[::-1]:
            cur_multiplier = cur_multiplier * base
            ret_list.insert(0, cur_multiplier)
            assert cur_multiplier < sys.float_info.max, 'search space too large, cannot be represented by this machine'

        # shift multipliers and set last to 0
        ret_list[:-1] = ret_list[1:]
        ret_list[-1] = 0

        return np.array(ret_list)

    def _convert_2_base_10(self, input_vec):
        """
        Converts parameter indices to unique base-10 integer.

        Parameters:
        -----------
        input_vec : list or array
            Vector of parameter indices.
        
        Returns:
        --------
        int
            Unique base-10 representation of the index vector.
        """
        assert len(input_vec) == len(self._bases)
        return np.sum(np.array(input_vec) * self._mulipliers)

    def _convert_2_base_letters(self, input_base_10):
        """
        Converts base-10 integer to base-62 letter representation.

        Parameters:
        -----------
        input_base_10 : int
            Base-10 integer to convert.
        
        Returns:
        --------
        list
            List of base-62 character digits.
        """
        x = input_base_10
        ret_list = []

        # iteratively convert to base-62
        while x:
            key = int(x % len(self._lookup_dict))
            ret_list.insert(0, self._lookup_dict[key])
            x = int(x / len(self._lookup_dict))

        return ret_list

    def _pad(self, input_base_letter):
        """
        Pads letter representation to required length with leading zeros.

        Parameters:
        -----------
        input_base_letter : list
            List of base-62 character digits.
        
        Returns:
        --------
        list
            Padded list with leading zeros to reach required length.
        """
        while len(input_base_letter) < self.length_letters:
            input_base_letter.insert(0, '0')
        return input_base_letter

    def convert_list_2_id(self, input_list):
        """
        Converts a parameter index list to a unique alphanumeric ID.

        Parameters:
        -----------
        input_list : list
            List of parameter indices.
        
        Returns:
        --------
        str
            Unique alphanumeric ID string encoding the design.
        """
        base10 = self._convert_2_base_10(input_list)
        base_letter = self._convert_2_base_letters(base10)
        padded = self._pad(base_letter)

        return ''.join(padded)


# ===== Design Class ===== #


class Design(list):
    """
    Represents a circuit design with parameter values and evolutionary history.

    Extends Python's list class to hold parameter indices. Tracks design cost,
    fitness, specifications, and evolutionary relationships (parents, siblings).
    Each design has a unique ID generated from its parameter vector.
    
    Initialization Parameters:
    --------------------------
    spec_range : dict
        Specification range dictionary with spec keywords as keys.
        Each design has a specs entry for each keyword.
    id_encoder : IDEncoder
        ID encoder instance for generating unique design IDs.
    seq : tuple or list
        Sequence of parameter indices for this design.
    """

    def __init__(self, spec_range, id_encoder, seq=()):

        # initialize parent list class
        list.__init__(self, seq)
        self.cost =     None
        self.fitness =  None
        self.specs = {}

        # encoder determines the ID of the design given the list values
        self.id_encoder = id_encoder

        self.spec_range = spec_range

        for spec_kwrd in spec_range.keys():
            self.specs[spec_kwrd] = None

        self.parent1 = None
        self.parent2 = None
        self.sibling = None

    def set_parents_and_sibling(self, parent1, parent2, sibling):
        """
        Sets the evolutionary lineage for this design.

        Parameters:
        -----------
        parent1 : Design
            First parent design.
        parent2 : Design
            Second parent design.
        sibling : Design
            Sibling design created from same parents.
        
        Returns:
        --------
        None
        """
        self.parent1 = parent1
        self.parent2 = parent2
        self.sibling = sibling

    def is_init_population(self):
        """
        Checks if this design is from the initial population.

        Initial population designs have no parents.

        Returns:
        --------
        bool
            True if design has no parent1, False otherwise.
        """
        if self.parent1 is None:
            return True
        else:
            return False

    def is_mutated(self):
        """
        Checks if this design resulted from mutation (single parent).

        A mutated design has parent1 but not parent2 (unlike crossover offspring).

        Returns:
        --------
        bool
            True if design has parent1 but not parent2, False otherwise.
        """
        if self.parent1 is not None:
            if self.parent2 is None:
                return True
        else:
            return False

    @property
    def id(self):
        """
        Gets the unique alphanumeric ID for this design.

        Returns:
        --------
        str
            Unique design ID encoding the parameter indices.
        """
        return self.id_encoder.convert_list_2_id(list(self))

    @property
    def cost(self):
        """
        Gets the design cost value.

        Returns:
        --------
        float or None
            Current cost value (None if uninitialized).
        """
        return self.__cost

    @property
    def fitness(self):
        """
        Gets the design fitness value.

        Fitness is the negative of cost, so minimizing cost maximizes fitness.

        Returns:
        --------
        float or None
            Current fitness value (None if uninitialized).
        """
        return self.__fitness

    @cost.setter
    def cost(self, x):
        """
        Sets cost and automatically updates fitness (negative of cost).

        Parameters:
        -----------
        x : float or None
            New cost value.
        
        Returns:
        --------
        None
        """
        self.__cost = x
        self.__fitness = -x if x is not None else None

    @fitness.setter
    def fitness(self, x):
        """
        Sets fitness and automatically updates cost (negative of fitness).

        Parameters:
        -----------
        x : float or None
            New fitness value.
        
        Returns:
        --------
        None
        """
        self.__fitness = x
        self.__cost = -x if x is not None else None

    @staticmethod
    def recreate_design(spec_range, old_design, eval_core):
        """
        Creates a new Design object from an existing design.

        Copies all attributes from the old design to preserve evolutionary history
        and specification values in a fresh Design instance.

        Parameters:
        -----------
        spec_range : dict
            Specification range dictionary for the new design.
        old_design : Design
            Source design to copy from.
        eval_core : EvaluationEngine
            Evaluation engine providing ID encoder.
        
        Returns:
        --------
        Design
            New Design object with copied attributes.
        """
        # create new design with same parameter values
        dsn = Design(spec_range, eval_core.id_encoder, old_design)

        # Copy attributes
        dsn.specs.update(**old_design.specs)
        for attr in dsn.__dict__.keys():
            if (attr in old_design.__dict__.keys()) and (attr not in ['specs']):
                dsn.__dict__[attr] = deepcopy(old_design.__dict__[attr])

        return dsn

    @staticmethod
    def genocide(*args):
        """
        Clears evolutionary history for multiple designs.

        Removes parent and sibling relationships to sever evolutionary lineage.
        Used to treat designs as independent (e.g., starting new generation).

        Parameters:
        -----------
        *args : Design
            Variable number of Design objects to clear history for.
        
        Returns:
        --------
        None
        """
        for dsn in args:
            dsn.parent1 = None
            dsn.parent2 = None
            dsn.sibling = None

    def copy(self):
        """
        Creates an independent copy of this design.

        Copies the parameter values and specifications dictionary while
        maintaining the list structure and other attributes.

        Returns:
        --------
        Design
            New Design object with copied data.
        """
        new = copy(self)
        new.specs = deepcopy(self.specs)
        return new

def extract_sizing_map(netlist_path):
    """
    Parses a Spectre netlist to map component parameters to optimization variables.
    
    Returns:
    --------
    dict
        { 
          "M1": { "l": "nA1", "nfin": "nB1" },
          "M2": { "l": "nA2", ... } 
        }
    """
    mapping = {}
    
    with open(netlist_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments
            if line.startswith('*') or line.startswith('//'):
                continue
            
            parts = line.split()
            if not parts:
                continue
                
            name = parts[0]
            
            # Filter out obvious non-components
            if '=' in name: 
                continue

            # Check for simulation directives that might mask as components
            # e.g. "mytran tran ...", "modelParameter info ..."
            if len(parts) > 1 and parts[1] in ['options', 'tran', 'dc', 'ac', 'noise', 'info', 'pz', 'sp', 'pss', 'hb', 'envlp']:
                continue

            # Only care about MOS, R, C
            # Using tuple for startswith is cleaner
            if name.upper().startswith(('M', 'R', 'C')):
                # Filter Testbench artifacts
                if any(tb in name for tb in ['Rin', 'Rfeed', 'Rload', 'Ctran', 'Cload', 'Rsw', 'Rsrc', 'Rshunt', 'R_unity']):
                    continue
                    
                # Scan for parameters
                # Format: MM3 V... nfet l=nA1 nfin=nB1 ...
                comp_map = {}
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        # clean up potential parens or formatting
                        key = key.strip()
                        val = val.strip().replace('(', '').replace(')', '')
                        
                        comp_map[key] = val
                        
                if comp_map:
                    mapping[name] = comp_map
                    
    return mapping
