#!/usr/bin/env python3
"""
Trajectory water removal tool - removes furthest water molecules from XYZ trajectory
while keeping a cluster centered around Xenon atom with unwrapped coordinates.

Usage:
    python remove_waters.py input.xyz output.xyz --n-remove 100 --n-jobs 4
"""

import numpy as np
import argparse
from ase.io import read, write
from ase import Atoms
from ase.geometry import find_mic
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class WaterRemover:
    """Remove water molecules from MD trajectory centered around Xenon."""
    
    def __init__(self, oh_bond_max=1.2, hoh_angle_min=95, hoh_angle_max=115):
        """
        Initialize water remover.
        
        Parameters:
        -----------
        oh_bond_max : float
            Maximum O-H bond length in Angstroms (default: 1.2)
        hoh_angle_min : float
            Minimum H-O-H angle in degrees (default: 95)
        hoh_angle_max : float
            Maximum H-O-H angle in degrees (default: 115)
        """
        self.oh_bond_max = oh_bond_max
        self.hoh_angle_min = np.radians(hoh_angle_min)
        self.hoh_angle_max = np.radians(hoh_angle_max)
    
    def unwrap_molecule(self, center_pos, atom_positions, cell, pbc):
        """
        Unwrap a molecule's atoms to be continuous around a center atom.
        
        Parameters:
        -----------
        center_pos : np.ndarray
            Position of center atom (e.g., oxygen in water)
        atom_positions : np.ndarray
            Positions of atoms to unwrap (e.g., hydrogens)
        cell : ase.Cell
            Simulation cell
        pbc : np.ndarray
            Periodic boundary conditions
            
        Returns:
        --------
        np.ndarray
            Unwrapped positions relative to center
        """
        vectors, distances = find_mic(atom_positions - center_pos, cell, pbc)
        return center_pos + vectors
    
    def unwrap_coordinates_molecular(self, atoms, xe_idx):
        """
        Unwrap periodic boundary conditions with molecular integrity.
        First unwraps all atoms relative to Xenon, then ensures water molecules are intact.
        
        Parameters:
        -----------
        atoms : ase.Atoms
            Atomic configuration with PBC
        xe_idx : int
            Index of Xenon atom (center of cluster)
            
        Returns:
        --------
        np.ndarray
            Unwrapped coordinates with Xenon at the origin and intact molecules
        """
        positions = atoms.get_positions()
        symbols = np.array(atoms.get_chemical_symbols())
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        
        xe_pos = positions[xe_idx]
        
        # First pass: unwrap all atoms relative to Xenon
        vectors, distances = find_mic(positions - xe_pos, cell, pbc)
        unwrapped_positions = vectors.copy()
        
        # Second pass: ensure water molecules are intact
        # Identify water oxygens and their hydrogens
        o_indices = np.where(symbols == 'O')[0]
        h_indices = np.where(symbols == 'H')[0]
        
        if len(o_indices) > 0 and len(h_indices) > 0:
            used_h = set()
            
            for o_idx in o_indices:
                o_pos_unwrapped = unwrapped_positions[o_idx]
                
                # Find hydrogens bonded to this oxygen using ORIGINAL (wrapped) positions
                h_pos = positions[h_indices]
                o_pos_wrapped = positions[o_idx]
                vectors_oh, distances_oh = find_mic(h_pos - o_pos_wrapped, cell, pbc)
                
                bonded_mask = distances_oh <= self.oh_bond_max
                bonded_h_local = np.where(bonded_mask)[0]
                bonded_h = [h_indices[i] for i in bonded_h_local if h_indices[i] not in used_h]
                
                # Unwrap hydrogens to be continuous with their oxygen
                for h_idx in bonded_h[:2]:  # Take first two hydrogens
                    # Use MIC to find proper hydrogen position relative to oxygen
                    h_vec, h_dist = find_mic(positions[h_idx:h_idx+1] - o_pos_wrapped, cell, pbc)
                    unwrapped_positions[h_idx] = o_pos_unwrapped + h_vec[0]
                    used_h.add(h_idx)
        
        return unwrapped_positions
    
    def identify_water_molecules_unwrapped(self, positions, symbols):
        """
        Identify water molecules in unwrapped coordinates using O-H bonds and H-O-H angles.
        
        Parameters:
        -----------
        positions : np.ndarray
            Unwrapped atomic positions
        symbols : np.ndarray
            Chemical symbols
            
        Returns:
        --------
        list of tuples
            Each tuple contains (O_index, H1_index, H2_index) for each water molecule
        """
        # Get oxygen and hydrogen indices
        o_indices = np.where(symbols == 'O')[0]
        h_indices = np.where(symbols == 'H')[0]
        
        if len(o_indices) == 0 or len(h_indices) == 0:
            return []
        
        water_molecules = []
        used_h = set()
        
        # For each oxygen, find its bonded hydrogens
        for o_idx in o_indices:
            o_pos = positions[o_idx]
            
            # Calculate distances to all hydrogens (simple Euclidean in unwrapped space)
            h_pos = positions[h_indices]
            distances = np.linalg.norm(h_pos - o_pos, axis=1)
            
            # Find hydrogens within bonding distance
            bonded_mask = distances <= self.oh_bond_max
            bonded_h_local = np.where(bonded_mask)[0]
            
            # Filter out already used hydrogens
            bonded_h = [h_indices[i] for i in bonded_h_local if h_indices[i] not in used_h]
            
            if len(bonded_h) >= 2:
                # Check H-O-H angle for the first two hydrogens
                h1_idx, h2_idx = bonded_h[0], bonded_h[1]
                
                # Get vectors (simple subtraction in unwrapped space)
                vec_oh1 = positions[h1_idx] - o_pos
                vec_oh2 = positions[h2_idx] - o_pos
                
                # Calculate angle
                cos_angle = np.dot(vec_oh1, vec_oh2) / (np.linalg.norm(vec_oh1) * np.linalg.norm(vec_oh2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                if self.hoh_angle_min <= angle <= self.hoh_angle_max:
                    water_molecules.append((o_idx, h1_idx, h2_idx))
                    used_h.add(h1_idx)
                    used_h.add(h2_idx)
        
        return water_molecules
    
    def get_water_molecule_max_distances(self, positions, xe_idx, water_molecules):
        """
        Calculate the maximum distance from Xenon to any atom in each water molecule.
        This ensures we remove waters based on their furthest extent, not just oxygen position.
        
        Parameters:
        -----------
        positions : np.ndarray
            Unwrapped atomic positions
        xe_idx : int
            Index of Xenon atom
        water_molecules : list of tuples
            List of (O_idx, H1_idx, H2_idx) for each water
            
        Returns:
        --------
        np.ndarray
            Maximum distance from Xenon to any atom in each water molecule
        """
        if len(water_molecules) == 0:
            return np.array([])
        
        xe_pos = positions[xe_idx]
        max_distances = []
        
        for o_idx, h1_idx, h2_idx in water_molecules:
            # Get distances from Xenon to all three atoms in the water molecule
            d_o = np.linalg.norm(positions[o_idx] - xe_pos)
            d_h1 = np.linalg.norm(positions[h1_idx] - xe_pos)
            d_h2 = np.linalg.norm(positions[h2_idx] - xe_pos)
            
            # Take the maximum distance (furthest atom in this water molecule)
            max_dist = max(d_o, d_h1, d_h2)
            max_distances.append(max_dist)
        
        return np.array(max_distances)
    
    def find_waters_to_remove(self, positions, xe_idx, water_molecules, n_remove):
        """
        Find which water molecules to remove based on maximum distance from Xenon.
        
        Parameters:
        -----------
        positions : np.ndarray
            Unwrapped atomic positions
        xe_idx : int
            Index of Xenon atom
        water_molecules : list of tuples
            List of (O_idx, H1_idx, H2_idx) for each water
        n_remove : int
            Number of water molecules to remove
            
        Returns:
        --------
        set
            Set of atom indices to remove
        """
        if n_remove <= 0 or len(water_molecules) == 0:
            return set()
        
        if n_remove >= len(water_molecules):
            # Remove all waters
            indices_to_remove = set()
            for o_idx, h1_idx, h2_idx in water_molecules:
                indices_to_remove.update([o_idx, h1_idx, h2_idx])
            return indices_to_remove
        
        # Get maximum distances for each water molecule
        max_distances = self.get_water_molecule_max_distances(positions, xe_idx, water_molecules)
        
        # Sort waters by maximum distance (furthest first)
        sorted_indices = np.argsort(max_distances)[::-1]
        
        # Select n_remove furthest waters
        waters_to_remove = [water_molecules[i] for i in sorted_indices[:n_remove]]
        
        # Collect all atom indices to remove
        indices_to_remove = set()
        for o_idx, h1_idx, h2_idx in waters_to_remove:
            indices_to_remove.update([o_idx, h1_idx, h2_idx])
        
        return indices_to_remove
    
    def calculate_cluster_diameter(self, positions, xe_idx, remaining_water_molecules):
        """
        Calculate the diameter of the cluster based on the furthest atom in any water molecule.
        
        Parameters:
        -----------
        positions : np.ndarray
            Unwrapped atomic positions
        xe_idx : int
            Index of Xenon atom
        remaining_water_molecules : list of tuples
            List of (O_idx, H1_idx, H2_idx) for remaining water molecules
            
        Returns:
        --------
        float
            Diameter of the cluster based on furthest water atom
        """
        if len(remaining_water_molecules) == 0:
            return 0.0
        
        xe_pos = positions[xe_idx]
        
        # Get maximum distance for each remaining water molecule
        max_distances = []
        for o_idx, h1_idx, h2_idx in remaining_water_molecules:
            d_o = np.linalg.norm(positions[o_idx] - xe_pos)
            d_h1 = np.linalg.norm(positions[h1_idx] - xe_pos)
            d_h2 = np.linalg.norm(positions[h2_idx] - xe_pos)
            max_distances.append(max(d_o, d_h1, d_h2))
        
        # Diameter is twice the maximum distance
        max_distance = np.max(max_distances)
        diameter = 2.0 * max_distance
        
        return diameter
    
    def process_frame(self, atoms, n_remove):
        """
        Process a single frame: unwrap, identify waters, remove furthest n_remove.
        
        Parameters:
        -----------
        atoms : ase.Atoms
            Atomic configuration
        n_remove : int
            Number of water molecules to remove
            
        Returns:
        --------
        tuple
            (ase.Atoms, float): Processed atomic configuration and cluster diameter
        """
        symbols = np.array(atoms.get_chemical_symbols())
        
        # Find Xenon atom
        xe_indices = np.where(symbols == 'Xe')[0]
        if len(xe_indices) == 0:
            raise ValueError("No Xenon atom found in the system")
        xe_idx = xe_indices[0]
        
        # STEP 1: Unwrap coordinates first (with molecular integrity)
        unwrapped_pos = self.unwrap_coordinates_molecular(atoms, xe_idx)
        
        # STEP 2: Identify water molecules in unwrapped space
        water_molecules = self.identify_water_molecules_unwrapped(unwrapped_pos, symbols)
        
        if len(water_molecules) == 0:
            # No waters
            new_atoms = Atoms(
                symbols=symbols,
                positions=unwrapped_pos,
                pbc=False
            )
            return new_atoms, 0.0
        
        # STEP 3: Find waters to remove based on maximum distance of any atom in the molecule
        indices_to_remove = self.find_waters_to_remove(unwrapped_pos, xe_idx, water_molecules, n_remove)
        
        # STEP 4: Create mask for atoms to keep
        n_atoms = len(atoms)
        keep_mask = np.ones(n_atoms, dtype=bool)
        keep_mask[list(indices_to_remove)] = False
        
        # Find new Xenon index after removal
        xe_idx_new = np.sum(keep_mask[:xe_idx])
        
        # Create mapping from old indices to new indices
        old_to_new = np.cumsum(keep_mask) - 1
        
        # Identify remaining water molecules with NEW indices
        remaining_water_molecules = []
        for o_idx, h1_idx, h2_idx in water_molecules:
            if o_idx not in indices_to_remove:
                # This water molecule was kept, convert to new indices
                new_o_idx = old_to_new[o_idx]
                new_h1_idx = old_to_new[h1_idx]
                new_h2_idx = old_to_new[h2_idx]
                remaining_water_molecules.append((new_o_idx, new_h1_idx, new_h2_idx))
        
        # STEP 5: Create new atoms object
        new_atoms = Atoms(
            symbols=symbols[keep_mask],
            positions=unwrapped_pos[keep_mask],
            pbc=False
        )
        
        # STEP 6: Calculate cluster diameter
        diameter = self.calculate_cluster_diameter(
            unwrapped_pos[keep_mask], 
            xe_idx_new, 
            remaining_water_molecules
        )
        
        return new_atoms, diameter
    
    def process_trajectory(self, input_file, output_file, n_remove, n_jobs=1):
        """
        Process entire trajectory in parallel.
        
        Parameters:
        -----------
        input_file : str
            Path to input XYZ trajectory file
        output_file : str
            Path to output XYZ trajectory file
        n_remove : int
            Number of water molecules to remove per frame
        n_jobs : int
            Number of parallel jobs (default: 1)
        """
        print(f"Reading trajectory from {input_file}...")
        trajectory = read(input_file, index=':')
        n_frames = len(trajectory)
        print(f"Found {n_frames} frames")
        
        # Process frames in parallel
        print(f"Processing frames with {n_jobs} parallel jobs...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_frame)(frame, n_remove)
            for frame in tqdm(trajectory, desc="Processing frames")
        )
        
        # Separate frames and diameters
        processed_frames = [result[0] for result in results]
        diameters = np.array([result[1] for result in results])
        
        # Write output
        print(f"Writing output to {output_file}...")
        write(output_file, processed_frames)
        
        # Print statistics
        initial_atoms = len(trajectory[0])
        final_atoms = len(processed_frames[0])
        removed_atoms = initial_atoms - final_atoms
        
        # Filter out zero diameters (frames with no water)
        valid_diameters = diameters[diameters > 0]
        
        if len(valid_diameters) > 0:
            avg_diameter = np.mean(valid_diameters)
            min_diameter = np.min(valid_diameters)
            max_diameter = np.max(valid_diameters)
            std_diameter = np.std(valid_diameters)
        else:
            avg_diameter = min_diameter = max_diameter = std_diameter = 0.0
        
        print("\nProcessing complete!")
        print(f"Initial atoms per frame: {initial_atoms}")
        print(f"Final atoms per frame: {final_atoms}")
        print(f"Removed atoms per frame: {removed_atoms}")
        print(f"Target water molecules to remove: {n_remove}")
        print(f"Actual atoms removed: {removed_atoms} (≈ {removed_atoms//3} water molecules)")
        print(f"Output has unwrapped coordinates centered on Xenon (no PBC)")
        print(f"\nCluster size statistics (based on furthest atom in any water molecule from Xe):")
        print(f"  Average diameter: {avg_diameter:.2f} Å")
        print(f"  Average radius: {avg_diameter/2:.2f} Å")
        print(f"  Min diameter: {min_diameter:.2f} Å")
        print(f"  Max diameter: {max_diameter:.2f} Å")
        print(f"  Std deviation: {std_diameter:.2f} Å")


def main():
    parser = argparse.ArgumentParser(
        description='Remove furthest water molecules from XYZ trajectory centered around Xenon with unwrapped coordinates'
    )
    parser.add_argument('input', type=str, help='Input XYZ trajectory file')
    parser.add_argument('output', type=str, help='Output XYZ trajectory file')
    parser.add_argument('--n-remove', type=int, required=True,
                       help='Number of water molecules to remove')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--oh-bond-max', type=float, default=1.2,
                       help='Maximum O-H bond length in Angstroms (default: 1.2)')
    parser.add_argument('--hoh-angle-min', type=float, default=95,
                       help='Minimum H-O-H angle in degrees (default: 95)')
    parser.add_argument('--hoh-angle-max', type=float, default=115,
                       help='Maximum H-O-H angle in degrees (default: 115)')
    
    args = parser.parse_args()
    
    # Create remover instance
    remover = WaterRemover(
        oh_bond_max=args.oh_bond_max,
        hoh_angle_min=args.hoh_angle_min,
        hoh_angle_max=args.hoh_angle_max
    )
    
    # Process trajectory
    remover.process_trajectory(
        args.input,
        args.output,
        args.n_remove,
        args.n_jobs
    )


if __name__ == '__main__':
    main()
