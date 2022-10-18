import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from diffab.modules.common.layers import AngularEncoding
from diffab.utils.protein.constants import BBHeavyAtom, AA


class PairEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi

        infeat_dim = feat_dim+feat_dim+feat_dim+feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None

        # Pair identities
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)
    
        # Relative sequential positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
            dim = -1, ord = 2,
        )).reshape(N, L, L, -1) # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))    # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all

