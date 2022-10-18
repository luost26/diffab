import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from diffab.modules.common.layers import AngularEncoding
from diffab.utils.protein.constants import BBHeavyAtom, AA


class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.type_embed = nn.Embedding(10, feat_dim, padding_idx=0)    # 1: Heavy, 2: Light, 3: Ag
        infeat_dim = feat_dim + (self.max_aa_types*max_num_atoms*3) + self.dihed_embed.get_out_dim(3) + feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L).
            res_nb:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            fragment_type:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # Amino acid identity features
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Coordinate features
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA], 
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)    # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand  = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types*self.max_num_atoms*3)
        if structure_mask is not None:
            # Avoid data leakage at training time
            crd_feat = crd_feat * structure_mask[:, :, None]

        # Backbone dihedral features
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            # Avoid data leakage at training time
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1), 
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )   # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        # Type feature
        type_feat = self.type_embed(fragment_type) # (N, L, feat)

        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat, type_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat
