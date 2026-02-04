# coding: utf-8
r"""
GenBM3: BM3 + GenRecV2 Generation Pipeline

Two independent paths:
- BM3 path: LightGCN + cosine similarity bootstrap loss (unchanged)
- GenRecV2 path: modal encoding -> conv_ui/conv_ii -> RF generation
                  -> collaborative signal enhancement -> modal signal enhancement
                  -> contrastive learning loss

Inference: RF generated embeddings mixed via mix_embeddings at two levels.
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from models.bm3 import BM3
from models.backbone import BACKBONE
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from utils.utils import build_sim, build_knn_normalized_graph


class GenBM3(BM3):
    # Import methods from BACKBONE (class-level attribute assignment)
    get_ii_adj_intersection_attention = BACKBONE.get_ii_adj_intersection_attention
    sparse_mx_to_torch_sparse_tensor = BACKBONE.sparse_mx_to_torch_sparse_tensor
    conv_ui = BACKBONE.conv_ui  # uses self.n_ui_layers
    InfoNCE = BACKBONE.InfoNCE
    align_vt = BACKBONE.align_vt
    # get_norm_adj_mat is different between BM3 and BACKBONE, import BACKBONE's version separately
    get_gen_norm_adj_mat = BACKBONE.get_norm_adj_mat  # needs self.interaction_matrix
    get_gen_adj_mat = BACKBONE.get_adj_mat  # sets self.R

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ===== GenRecV2 independent infrastructure (not shared with BM3) =====

        # Configuration parameters
        self.n_ui_layers = config['n_ui_layers']
        self.gen_n_layers = config['gen_n_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.gen_cl_loss_weight = config['gen_cl_loss']
        self.vt_loss_weight = config['vt_loss']
        self.bm_temp_gen = config['bm_temp']
        self.ps_loss_weight = config['ps_loss_weight']

        self.embedding_dim = config['embedding_size']
        self.sparse = True

        # ===== KNN graph construction (BACKBONE logic) =====
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        image_adj_file = os.path.join(
            self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse)
        )
        text_adj_file = os.path.join(
            self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse)
        )

        if self.v_feat is not None:
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(
                    image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym'
                )
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.to(self.device)

        if self.t_feat is not None:
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(
                    text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym'
                )
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.to(self.device)

        # Enhancing User-Item Graph (intersection + attention weighting)
        self.ii_adj = self.get_ii_adj_intersection_attention(
            self.image_original_adj, self.text_original_adj, self.mm_image_weight
        )

        # Adjacency matrices:
        # gen_norm_adj_origin: standard user-item (RF starting point)
        # gen_norm_adj: enhanced version with item-item edges (RF target)
        self.gen_norm_adj_origin = self.get_gen_norm_adj_mat().to(self.device)
        self.gen_norm_adj = self.get_gen_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.gen_norm_adj = self.sparse_mx_to_torch_sparse_tensor(
            self.gen_norm_adj
        ).float().to(self.device)

        # ===== Modal space transforms (GenRecV2 style, separate from BM3's image_trs/text_trs) =====
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )

        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        # Collaborative signal enhancement
        self.behavior_adaptive_aware = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        # Modal signal enhancement
        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gen_softmax = nn.Softmax(dim=-1)

        # ===== RF Generator =====
        self.use_rf = config['use_rf'] if 'use_rf' in config else True

        if self.use_rf:
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config['rf_hidden_dim'] if 'rf_hidden_dim' in config else 128,
                n_layers=config['rf_n_layers'] if 'rf_n_layers' in config else 2,
                dropout=config['rf_dropout'] if 'rf_dropout' in config else 0.1,
                learning_rate=config['rf_learning_rate'] if 'rf_learning_rate' in config else 0.0001,
                sampling_steps=config['rf_sampling_steps'] if 'rf_sampling_steps' in config else 10,
                warmup_epochs=config['rf_warmup_epochs'] if 'rf_warmup_epochs' in config else 20,
                train_mix_ratio=config['rf_mix_ratio'] if 'rf_mix_ratio' in config else 0,
                inference_mix_ratio=config['rf_inference_mix_ratio'] if 'rf_inference_mix_ratio' in config else 0.05,
                contrast_temp=config['rf_contrast_temp'] if 'rf_contrast_temp' in config else 0.1,
                contrast_weight=config['rf_loss_weight'] if 'rf_loss_weight' in config else 0.01,
                n_users=self.n_users,
                n_items=self.n_items,
                use_2rf=config['use_2rf'] if 'use_2rf' in config else False,
                rf_2rf_transition_epoch=config['rf_2rf_transition_epoch'] if 'rf_2rf_transition_epoch' in config else None,
                use_gradient_checkpointing=config['use_gradient_checkpointing'] if 'use_gradient_checkpointing' in config else True,
            )
            self._rf_logged_this_epoch = False
            self._current_batch_users = None
            self._current_batch_items = None
            self._training_epoch = -1

        # ===== CausalDenoiser =====
        self.use_denoise = config['use_denoise'] if 'use_denoise' in config else False

        if self.use_denoise:
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=self.embedding_dim,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config['denoise_layers'] if 'denoise_layers' in config else 2,
                clean_rating_threshold=config['clean_rating_threshold'] if 'clean_rating_threshold' in config else 5.0,
                device=self.device,
            )
            self.causal_denoiser.load_treatment_labels(dataset)


    def conv_ii(self, ii_adj, single_modal):
        """conv_ii using gen_n_layers (separate from BACKBONE's n_layers)."""
        for i in range(self.gen_n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def _gen_forward(self):
        """
        GenRecV2 independent path: modal encoding, conv_ui/conv_ii,
        RF generation, collaborative/modal signal enhancement.

        Returns:
            rf_outputs dict or None
        """
        # 1. Multimodal encoding
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.image_space_trans(self.image_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )

        user_embeds = self.user_embedding.weight
        item_embeds = self.item_id_embedding.weight

        # 2. conv_ui: target (enhanced graph) and start (original graph)
        extended_id_embeds = self.conv_ui(
            self.gen_norm_adj, user_embeds, item_embeds
        )
        extended_id_embeds_origin = self.conv_ui(
            self.gen_norm_adj_origin, user_embeds, item_embeds
        )
        extended_id_embeds_target = extended_id_embeds

        # 3. Explicit multimodal features (RF conditions)
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat(
            [explicit_image_user, explicit_image_item], dim=0
        )

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat(
            [explicit_text_user, explicit_text_item], dim=0
        )

        rf_outputs = None
        extended_id_embeds_aug = extended_id_embeds

        if self.use_rf and self.training:
            # ===== Training mode =====
            # 4. CausalDenoiser -> RF target
            ps_loss = 0.0
            if self.use_denoise:
                ego_emb_for_denoise = torch.cat((user_embeds, item_embeds), dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = extended_id_embeds_target.detach()
            else:
                rf_target = extended_id_embeds_target.detach()

            # 5. User prior
            Z_u = explicit_image_embeds[:self.n_users] + explicit_text_embeds[:self.n_users]
            Z_hat = Z_u.mean(dim=0, keepdim=True)
            user_prior = Z_u - Z_hat
            item_prior = torch.zeros(self.n_items, self.embedding_dim).to(Z_u.device)
            full_prior = torch.cat([user_prior, item_prior], dim=0)

            # 6. RF independent training
            rf_start = extended_id_embeds_origin
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=rf_target,
                conditions=[explicit_image_embeds.detach(), explicit_text_embeds.detach()],
                user_prior=full_prior.detach(),
                epoch=self.rf_generator.current_epoch,
                batch_users=self._current_batch_users,
                batch_pos_items=self._current_batch_items,
            )

            if not self._rf_logged_this_epoch:
                log_msg = (
                    f"  [GenBM3 RF Train] epoch={self.rf_generator.current_epoch}, "
                    f"rf_loss={loss_dict['rf_loss']:.6f}, cl_loss={loss_dict['cl_loss']:.6f}"
                )
                if self.use_denoise:
                    log_msg += f", ps_loss={ps_loss if isinstance(ps_loss, float) else ps_loss.item():.6f}"
                print(log_msg)
                self._rf_logged_this_epoch = True

            # 7. RF generation
            rf_embeds = self.rf_generator.generate(
                [explicit_image_embeds, explicit_text_embeds]
            )

            # 8. Collaborative signal enhancement
            origin_weights, gen_weights = torch.split(
                self.gen_softmax(torch.cat([
                    self.behavior_adaptive_aware(extended_id_embeds),
                    self.behavior_adaptive_aware(rf_embeds),
                ], dim=-1)),
                1, dim=-1,
            )
            extended_id_embeds_aug = (
                origin_weights * extended_id_embeds + gen_weights * rf_embeds
            )

            rf_outputs = {"ps_loss": ps_loss}

        elif self.use_rf and not self.training:
            # ===== Inference mode (same as GenRecV2) =====
            with torch.no_grad():
                rf_embeds = self.rf_generator.generate(
                    [explicit_image_embeds, explicit_text_embeds]
                )
                extended_id_embeds = self.rf_generator.mix_embeddings(
                    extended_id_embeds_target,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )

        # 9. Modal signal enhancement
        image_weights, text_weights = torch.split(
            self.gen_softmax(torch.cat([
                self.separate_coarse(explicit_image_embeds),
                self.separate_coarse(explicit_text_embeds),
            ], dim=-1)),
            1, dim=-1,
        )
        integration_embeds = (
            image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
        )

        return {
            "integration_embeds": integration_embeds,
            "extended_id_embeds": extended_id_embeds,
            "extended_id_embeds_aug": extended_id_embeds_aug,
            "explicit_image_embeds": explicit_image_embeds,
            "explicit_text_embeds": explicit_text_embeds,
            "rf_outputs": rf_outputs,
        }

    def forward(self):
        h = self.item_id_embedding.weight

        # ===== BM3 path (unchanged) =====
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_bm3, i_g_bm3 = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        # ===== GenRecV2 independent path =====
        gen_outputs = self._gen_forward()

        if self.training:
            # Training: return BM3 outputs + GenRecV2 outputs separately
            return u_g_bm3, i_g_bm3 + h, gen_outputs
        else:
            # Inference: GenRecV2 path -> gen_all, then mix with BM3
            extended_id_embeds = gen_outputs["extended_id_embeds"]
            integration_embeds = gen_outputs["integration_embeds"]
            gen_all = extended_id_embeds + integration_embeds

            # BM3 as base, GenRecV2 as enhancement
            bm3_embeds = all_embeddings  # (n_users+n_items, dim)
            all_embeds = self.rf_generator.mix_embeddings(
                bm3_embeds, gen_all,
                training=False,
                epoch=self.rf_generator.current_epoch,
            )
            u_g, i_g = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
            return u_g, i_g + h, None

    def calculate_loss(self, interactions):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interactions[0]
        self._current_batch_items = interactions[1]

        users, items = interactions[0], interactions[1]

        # Forward pass
        u_online_ori, i_online_ori, gen_outputs = self.forward()

        # ===== BM3 original loss (fully preserved) =====
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online = self.predictor(u_online_ori)[users, :]
        i_online = self.predictor(i_online_ori)[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online_pred = self.predictor(t_feat_online)
            t_feat_online_pred = t_feat_online_pred[items, :]
            t_feat_target_batch = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(
                t_feat_online_pred, i_target.detach(), dim=-1
            ).mean()
            loss_tv = 1 - cosine_similarity(
                t_feat_online_pred, t_feat_target_batch.detach(), dim=-1
            ).mean()
        if self.v_feat is not None:
            v_feat_online_pred = self.predictor(v_feat_online)
            v_feat_online_pred = v_feat_online_pred[items, :]
            v_feat_target_batch = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(
                v_feat_online_pred, i_target.detach(), dim=-1
            ).mean()
            loss_vt = 1 - cosine_similarity(
                v_feat_online_pred, v_feat_target_batch.detach(), dim=-1
            ).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        bm3_loss = (
            (loss_ui + loss_iu).mean()
            + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori)
            + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt)
        )

        # ===== GenRecV2 loss (independent) =====
        integration_embeds = gen_outputs["integration_embeds"]
        extended_id_embeds_aug = gen_outputs["extended_id_embeds_aug"]
        explicit_image_embeds = gen_outputs["explicit_image_embeds"]
        explicit_text_embeds = gen_outputs["explicit_text_embeds"]
        rf_outputs = gen_outputs["rf_outputs"]

        # Modal alignment loss
        vt_loss = self.vt_loss_weight * self.align_vt(
            explicit_image_embeds, explicit_text_embeds
        )

        # Collaborative signal enhancement contrastive learning
        integration_users, integration_items = torch.split(
            integration_embeds, [self.n_users, self.n_items], dim=0
        )
        aug_users, aug_items = torch.split(
            extended_id_embeds_aug, [self.n_users, self.n_items], dim=0
        )
        cl_loss_aug = self.gen_cl_loss_weight * (
            self.InfoNCE(integration_users[users], aug_users[users], self.bm_temp_gen)
            + self.InfoNCE(integration_items[items], aug_items[items], self.bm_temp_gen)
        )

        # Propensity score loss
        ps_loss_val = 0.0
        if rf_outputs is not None and "ps_loss" in rf_outputs:
            ps_loss_val = self.ps_loss_weight * rf_outputs["ps_loss"]

        total_loss = bm3_loss + vt_loss + cl_loss_aug + ps_loss_val

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online, _ = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui
