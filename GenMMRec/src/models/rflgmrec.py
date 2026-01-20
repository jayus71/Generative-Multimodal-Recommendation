# coding: utf-8
# @email: georgeguo.gzq.cn@gmail.com
r"""
RFLGMRec: RF-Enhanced LGMRec
Integrates Rectified Flow module to enhance collaborative graph embeddings
"""

import torch
import torch.nn.functional as F

from models.lgmrec import LGMRec
from models.rf_modules import RFEmbeddingGenerator


class RFLGMRec(LGMRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # cl_loss_in_main: True=在calculate_loss计算, False=在rf_module的compute_loss_and_step计算
            self.cl_loss_in_main = config["cl_loss_in_main"] if "cl_loss_in_main" in config else True

            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 128,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.0001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 5,
                train_mix_ratio=config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.1,
                inference_mix_ratio=config["rf_inference_mix_ratio"] if "rf_inference_mix_ratio" in config else 0.2,
                contrast_temp=config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2,
                contrast_weight=config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0,
                cl_loss_in_main=self.cl_loss_in_main,
                n_users=self.n_users,
                n_items=self.n_items,
            )
            self._rf_logged_this_epoch = False

    def set_epoch(self, epoch):
        """Set current epoch for RF generator."""
        if self.use_rf:
            self.rf_generator.set_epoch(epoch)
            self._rf_logged_this_epoch = False

    def forward(self):
        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        # CGE: collaborative graph embedding
        cge_embs = self.cge()
        cge_embs_ori = cge_embs.clone()

        # ===== RF Enhancement for full user+item CGE =====
        rf_outputs = None

        if self.use_rf:
            # Get modality features (already user+item)
            full_conditions = []
            if self.v_feat is not None:
                v_feats = self.mge('v')  # Already aggregated on user-item graph
                full_conditions.append(v_feats)

            if self.t_feat is not None:
                t_feats = self.mge('t')  # Already aggregated on user-item graph
                full_conditions.append(t_feats)

            if len(full_conditions) > 0 and self.training:
                # 计算用户先验（用于RF指导）
                # Z_u: 用户特定的多模态兴趣表示
                Z_u = torch.zeros(self.n_users, self.embedding_dim).to(cge_embs_ori.device)
                if self.v_feat is not None:
                    Z_u = Z_u + v_feats[:self.n_users]
                if self.t_feat is not None:
                    Z_u = Z_u + t_feats[:self.n_users]

                # Z_hat_u: 通用用户兴趣表示（所有用户的平均值）
                Z_hat_u = Z_u.mean(dim=0, keepdim=True)

                # 用户先验: 独特的用户兴趣
                user_prior = Z_u - Z_hat_u  # shape: (n_users, embedding_dim)

                # 计算物品先验（用于RF指导）
                # Z_i: 物品特定的多模态特征表示
                Z_i = torch.zeros(self.n_items, self.embedding_dim).to(cge_embs_ori.device)
                if self.v_feat is not None:
                    Z_i = Z_i + v_feats[self.n_users:]
                if self.t_feat is not None:
                    Z_i = Z_i + t_feats[self.n_users:]

                # Z_hat_i: 通用物品特征表示（所有物品的平均值）
                Z_hat_i = Z_i.mean(dim=0, keepdim=True)

                # 物品先验: 独特的物品特征
                item_prior = Z_i - Z_hat_i  # shape: (n_items, embedding_dim)

                # 合并用户和物品先验
                full_prior = torch.cat([user_prior, item_prior], dim=0)

                # RF training with full user+item CGE embeddings
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=cge_embs_ori.detach(),
                    conditions=[c.detach() for c in full_conditions],
                    user_prior=full_prior.detach(),
                    epoch=self.rf_generator.current_epoch,
                )

                if not self._rf_logged_this_epoch:
                    if self.cl_loss_in_main:
                        print(f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                              f"rf_loss={loss_dict['rf_loss']:.6f}")
                    else:
                        print(f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                              f"rf_loss={loss_dict['rf_loss']:.6f}, "
                              f"cl_loss={loss_dict['cl_loss']:.6f}")
                    self._rf_logged_this_epoch = True

                # Generate RF embeddings for full user+item space
                rf_embeds = self.rf_generator.generate(full_conditions)

                # Mix embeddings
                cge_embs = self.rf_generator.mix_embeddings(
                    cge_embs_ori, rf_embeds.detach(), training=True
                )

                # Store rf_outputs for cl_loss in calculate_loss (only when cl_loss_in_main=True)
                if self.cl_loss_in_main:
                    rf_outputs = {
                        "rf_embeds": rf_embeds,
                        "target_embeds": cge_embs_ori,
                    }

            elif len(full_conditions) > 0 and not self.training:
                # Inference mode
                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    cge_embs = self.rf_generator.mix_embeddings(
                        cge_embs_ori, rf_embeds, training=False
                    )

        # Continue with original LGMRec logic
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: modal graph embedding
            v_feats = self.mge('v')
            t_feats = self.mge('t')
            # local embeddings = collaborative-related embedding + modality-related embedding
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
            lge_embs = cge_embs + mge_embs
            # GHE: global hypergraph embedding
            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            # local embeddings + alpha * global embeddings
            all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)
        else:
            all_embs = cge_embs
            uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs = None, None, None, None

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs], rf_outputs

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, hyper_embeddings, rf_outputs = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        batch_hcl_loss = 0.0
        if hyper_embeddings[0] is not None:
            [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
            batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + \
                             self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.cl_weight * batch_hcl_loss + self.reg_weight * batch_reg_loss

        # RF contrastive loss (cl_loss) - only compute here when cl_loss_in_main=True
        if self.use_rf and self.cl_loss_in_main and rf_outputs is not None:
            rf_embeds = rf_outputs["rf_embeds"]
            target_embeds = rf_outputs["target_embeds"]

            rf_users, rf_items = torch.split(rf_embeds, [self.n_users, self.n_items], dim=0)
            target_users, target_items = torch.split(target_embeds, [self.n_users, self.n_items], dim=0)

            rf_cl_loss = self.rf_generator._infonce_loss(rf_items[pos_items], target_items[pos_items], 0.2) + \
                         self.rf_generator._infonce_loss(rf_users[users], target_users[users], 0.2)

            loss = loss + self.rf_generator.contrast_weight * rf_cl_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores
