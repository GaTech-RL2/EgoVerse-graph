# Model configuration dispositions

All 128 source model entries map to the 67 retained command paths. `aliased`
removes a byte-identical copied definition; `merged` uses an explicit architecture
or action-width override. `migrated` identifies a C fragment whose original caller
knobs were removed in C: the retained B experiment/contract supplies its supported
replacement, and source equivalence is not claimed without that caller.

The [CSV](evidence/consolidation/model-dispositions.csv) includes full source SHAs,
resolved hashes, compatibility notes and experiment callers. Full raw model
declarations, resolved components, data/evaluator references, initialization
and training knobs are in the [inventory](evidence/consolidation/config-inventory.json.gz).

| Source | Model command | Disposition | Canonical definition |
| --- | --- | --- | --- |
| A | `abc/yam_bimanual_dp` | kept | `abc/yam_bimanual_dp` |
| A | `abc/yam_bimanual_hpt` | kept | `abc/yam_bimanual_hpt` |
| A | `abc_arc/hpt_bimanual_visual` | kept | `abc_arc/hpt_bimanual_visual` |
| A | `abc_arc/hpt_yam_visual` | kept | `abc_arc/hpt_yam_visual` |
| A | `abc_arc/hpt_yam_visual_arc_stream` | kept | `abc_arc/hpt_yam_visual_arc_stream` |
| A | `bf/bf_planar_v2_arc_graph_tok` | kept | `bf/bf_planar_v2_arc_graph_tok` |
| A | `bf/bf_planar_v2_dp_paper` | kept | `bf/bf_planar_v2_dp_paper` |
| A | `bf/bf_planar_v2_dp_standard` | kept | `bf/bf_planar_v2_dp_standard` |
| A | `bf/bf_planar_v2_flow` | kept | `bf/bf_planar_v2_flow` |
| A | `bf/ct_unite_register_separate_nt8_h384_s42` | kept | `bf/ct_unite_register_separate_nt8_h384_s42` |
| A | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` |
| A | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` |
| A | `bf/us_action_flow_latent_fm_sg_unite_h384` | kept | `bf/us_action_flow_latent_fm_sg_unite_h384` |
| A | `bf/us_action_latent_vfm_nt16_d8_h512_s42` | kept | `bf/us_action_latent_vfm_nt16_d8_h512_s42` |
| A | `bf/us_unite_register_separate_nt4_s42` | kept | `bf/us_unite_register_separate_nt4_s42` |
| A | `bf/us_unite_register_separate_nt8_s42` | kept | `bf/us_unite_register_separate_nt8_s42` |
| A | `bf/us_unite_register_shared_nt16_s42` | kept | `bf/us_unite_register_shared_nt16_s42` |
| A | `bf/us_unite_register_shared_nt4_s42` | kept | `bf/us_unite_register_shared_nt4_s42` |
| A | `bf/us_unite_register_shared_nt8_flow42_s42` | kept | `bf/us_unite_register_shared_nt8_flow42_s42` |
| A | `bf/us_unite_register_shared_nt8_s42` | kept | `bf/us_unite_register_shared_nt8_s42` |
| A | `e1/hpt_flow` | kept | `e1/hpt_flow` |
| A | `e1/hpt_flow_wrists` | kept | `e1/hpt_flow_wrists` |
| A | `pi05/pi0.5_bc_abc_eva_6d` | kept | `pi05/pi0.5_bc_abc_eva_6d` |
| A | `pi05/pi0.5_bc_eva` | kept | `pi05/pi0.5_bc_eva` |
| A | `pi05/pi0.5_bc_mecka_6d` | kept | `pi05/pi0.5_bc_mecka_6d` |
| A | `pi05/pi0.5_bc_mecka_abc_6d` | kept | `pi05/pi0.5_bc_mecka_abc_6d` |
| A | `pi05/pi0.5_cotrain_eva_aria_6d` | kept | `pi05/pi0.5_cotrain_eva_aria_6d` |
| A | `pi05/pi0.5_ft_abc_eva_6d` | aliased | `pi05/pi0.5_bc_abc_eva_6d` |
| A | `pi05/pi0.5_ft_aria_fold_grip` | kept | `pi05/pi0.5_ft_aria_fold_grip` |
| B | `abc/yam_bimanual_dp` | kept | `abc/yam_bimanual_dp` |
| B | `abc/yam_bimanual_hpt` | kept | `abc/yam_bimanual_hpt` |
| B | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled` | kept | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled` |
| B | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` | kept | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` |
| B | `abc_arc/yam_bimanual_dp_arc_D40_M100` | kept | `abc_arc/yam_bimanual_dp_arc_D40_M100` |
| B | `abc_arc/yam_mecka_cotrain_dp_300M_arc_D40_M100` | kept | `abc_arc/yam_mecka_cotrain_dp_300M_arc_D40_M100` |
| B | `abc_arc/yam_mecka_cotrain_dp_300M_baseline` | kept | `abc_arc/yam_mecka_cotrain_dp_300M_baseline` |
| B | `abc_arc/yam_mecka_cotrain_dp_arc_D40_M100` | kept | `abc_arc/yam_mecka_cotrain_dp_arc_D40_M100` |
| B | `abc_arc/yam_mecka_cotrain_dp_baseline` | kept | `abc_arc/yam_mecka_cotrain_dp_baseline` |
| B | `bf/bf_planar_v2_arc_graph_tok` | kept | `bf/bf_planar_v2_arc_graph_tok` |
| B | `bf/bf_planar_v2_dp_paper` | kept | `bf/bf_planar_v2_dp_paper` |
| B | `bf/bf_planar_v2_dp_standard` | kept | `bf/bf_planar_v2_dp_standard` |
| B | `bf/bf_planar_v2_flow` | kept | `bf/bf_planar_v2_flow` |
| B | `bf/ct_unite_register_separate_nt8_h384_s42` | kept | `bf/ct_unite_register_separate_nt8_h384_s42` |
| B | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` |
| B | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` |
| B | `bf/us_action_flow_latent_fm_sg_unite_h384` | kept | `bf/us_action_flow_latent_fm_sg_unite_h384` |
| B | `bf/us_action_latent_vfm_nt16_d8_h512_s42` | kept | `bf/us_action_latent_vfm_nt16_d8_h512_s42` |
| B | `bf/us_unite_register_separate_nt4_s42` | kept | `bf/us_unite_register_separate_nt4_s42` |
| B | `bf/us_unite_register_separate_nt8_s42` | kept | `bf/us_unite_register_separate_nt8_s42` |
| B | `bf/us_unite_register_shared_nt16_s42` | kept | `bf/us_unite_register_shared_nt16_s42` |
| B | `bf/us_unite_register_shared_nt4_s42` | kept | `bf/us_unite_register_shared_nt4_s42` |
| B | `bf/us_unite_register_shared_nt8_flow42_s42` | kept | `bf/us_unite_register_shared_nt8_flow42_s42` |
| B | `bf/us_unite_register_shared_nt8_s42` | kept | `bf/us_unite_register_shared_nt8_s42` |
| B | `e1/hpt_flow` | kept | `e1/hpt_flow` |
| B | `e1/hpt_flow_wrists` | kept | `e1/hpt_flow_wrists` |
| B | `egobridge` | kept | `egobridge` |
| B | `hpt_bc_flow_aria` | aliased | `hpt_bc_flow_human` |
| B | `hpt_bc_flow_eva` | kept | `hpt_bc_flow_eva` |
| B | `hpt_bc_flow_human` | kept | `hpt_bc_flow_human` |
| B | `hpt_bc_flow_mecka` | kept | `hpt_bc_flow_mecka` |
| B | `hpt_bc_flow_scale` | kept | `hpt_bc_flow_scale` |
| B | `hpt_bc_keypoints_base` | kept | `hpt_bc_keypoints_base` |
| B | `hpt_bc_keypoints_wrist` | kept | `hpt_bc_keypoints_wrist` |
| B | `hpt_bc_pickplace_qwen_pertoken` | kept | `hpt_bc_pickplace_qwen_pertoken` |
| B | `hpt_bc_pickplace_qwen_pooled` | kept | `hpt_bc_pickplace_qwen_pooled` |
| B | `hpt_cotrain_enc_dec_base` | kept | `hpt_cotrain_enc_dec_base` |
| B | `hpt_cotrain_flow_seperate_head` | kept | `hpt_cotrain_flow_seperate_head` |
| B | `hpt_cotrain_flow_shared_head` | kept | `hpt_cotrain_flow_shared_head` |
| B | `hpt_cotrain_mecka_flow_shared_head` | kept | `hpt_cotrain_mecka_flow_shared_head` |
| B | `hpt_cotrain_scale_flow_shared_head` | aliased | `hpt_cotrain_mecka_flow_shared_head` |
| B | `pi0.5_base` | kept | `pi0.5_base` |
| B | `pi0.5_bc_aria` | kept | `pi0.5_bc_aria` |
| B | `pi0.5_bc_eva` | kept | `pi0.5_bc_eva` |
| B | `pi0.5_bc_mecka` | kept | `pi0.5_bc_mecka` |
| B | `pi0.5_bc_scale` | aliased | `pi0.5_bc_mecka` |
| B | `pi0.5_cotrain_eva_aria` | kept | `pi0.5_cotrain_eva_aria` |
| B | `pi0.5_cotrain_mecka_scale` | aliased | `pi0.5_bc_mecka` |
| B | `pi05/pi0.5_bc_abc_eva_6d` | kept | `pi05/pi0.5_bc_abc_eva_6d` |
| B | `pi05/pi0.5_bc_eva` | kept | `pi05/pi0.5_bc_eva` |
| B | `pi05/pi0.5_bc_mecka_6d` | kept | `pi05/pi0.5_bc_mecka_6d` |
| B | `pi05/pi0.5_bc_mecka_abc_6d` | kept | `pi05/pi0.5_bc_mecka_abc_6d` |
| B | `pi05/pi0.5_cotrain_eva_aria_6d` | kept | `pi05/pi0.5_cotrain_eva_aria_6d` |
| B | `pi05/pi0.5_ft_abc_eva_6d` | aliased | `pi05/pi0.5_bc_abc_eva_6d` |
| B | `pi05/pi0.5_ft_aria_fold_grip` | kept | `pi05/pi0.5_ft_aria_fold_grip` |
| C | `abc/yam_bimanual_dp` | kept | `abc/yam_bimanual_dp` |
| C | `abc/yam_bimanual_hpt` | kept | `abc/yam_bimanual_hpt` |
| C | `abc_arc/clock_head_abc_cotrain_300M_qwen_pooled_arc_D40_M100` | migrated | `abc_arc/clock_head_abc_cotrain_300M_qwen_pooled_arc_D40_M100` |
| C | `abc_arc/hpt_abc_cotrain_180M_qwen_pooled` | migrated | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled` |
| C | `abc_arc/hpt_abc_cotrain_180M_qwen_pooled_arc_D40_M100` | migrated | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` |
| C | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled` | migrated | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled` |
| C | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` | migrated | `abc_arc/hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` |
| C | `abc_arc/hpt_bimanual_visual` | kept | `abc_arc/hpt_bimanual_visual` |
| C | `abc_arc/hpt_yam_visual` | kept | `abc_arc/hpt_yam_visual` |
| C | `abc_arc/mixture_hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` | migrated | `abc_arc/mixture_hpt_abc_cotrain_300M_qwen_pooled_arc_D40_M100` |
| C | `abc_arc/yam_bimanual_dp_arc_D40_M100` | migrated | `abc_arc/yam_bimanual_dp_arc_D40_M100` |
| C | `abc_arc/yam_mecka_cotrain_dp_300M_arc_D40_M100` | migrated | `abc_arc/yam_mecka_cotrain_dp_300M_arc_D40_M100` |
| C | `abc_arc/yam_mecka_cotrain_dp_300M_baseline` | kept | `abc_arc/yam_mecka_cotrain_dp_300M_baseline` |
| C | `abc_arc/yam_mecka_cotrain_dp_arc_D40_M100` | migrated | `abc_arc/yam_mecka_cotrain_dp_arc_D40_M100` |
| C | `abc_arc/yam_mecka_cotrain_dp_baseline` | kept | `abc_arc/yam_mecka_cotrain_dp_baseline` |
| C | `bf/bf_planar_v2_arc_graph_tok` | kept | `bf/bf_planar_v2_arc_graph_tok` |
| C | `bf/bf_planar_v2_dp_paper` | kept | `bf/bf_planar_v2_dp_paper` |
| C | `bf/bf_planar_v2_dp_standard` | kept | `bf/bf_planar_v2_dp_standard` |
| C | `bf/bf_planar_v2_flow` | kept | `bf/bf_planar_v2_flow` |
| C | `bf/ct_unite_register_separate_nt8_h384_s42` | kept | `bf/ct_unite_register_separate_nt8_h384_s42` |
| C | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_adamw` |
| C | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` | kept | `bf/us_action_flow_bc_latent_fm_sg_200m_muon` |
| C | `bf/us_action_flow_latent_fm_sg_unite_h384` | kept | `bf/us_action_flow_latent_fm_sg_unite_h384` |
| C | `bf/us_action_latent_vfm_nt16_d8_h512_s42` | kept | `bf/us_action_latent_vfm_nt16_d8_h512_s42` |
| C | `bf/us_unite_register_separate_nt4_s42` | kept | `bf/us_unite_register_separate_nt4_s42` |
| C | `bf/us_unite_register_separate_nt8_s42` | kept | `bf/us_unite_register_separate_nt8_s42` |
| C | `bf/us_unite_register_shared_nt16_s42` | kept | `bf/us_unite_register_shared_nt16_s42` |
| C | `bf/us_unite_register_shared_nt4_s42` | kept | `bf/us_unite_register_shared_nt4_s42` |
| C | `bf/us_unite_register_shared_nt8_flow42_s42` | kept | `bf/us_unite_register_shared_nt8_flow42_s42` |
| C | `bf/us_unite_register_shared_nt8_s42` | kept | `bf/us_unite_register_shared_nt8_s42` |
| C | `e1/dp300_wrists_arcdur` | kept | `e1/dp300_wrists_arcdur` |
| C | `e1/dp300pt_wrists_arcdur` | merged | `e1/dp300_wrists_arcdur` |
| C | `e1/hpt300_flow_wrists_arcdur` | merged | `e1/hpt_flow_wrists_ft` |
| C | `e1/hpt_flow` | kept | `e1/hpt_flow` |
| C | `e1/hpt_flow_wrists` | kept | `e1/hpt_flow_wrists` |
| C | `e1/hpt_flow_wrists_ft` | kept | `e1/hpt_flow_wrists_ft` |
| C | `e1/hpt_flow_wrists_ft_arcdur` | merged | `e1/hpt_flow_wrists_ft` |
| C | `pi05/pi0.5_bc_abc_eva_6d` | kept | `pi05/pi0.5_bc_abc_eva_6d` |
| C | `pi05/pi0.5_bc_eva` | kept | `pi05/pi0.5_bc_eva` |
| C | `pi05/pi0.5_bc_mecka_6d` | kept | `pi05/pi0.5_bc_mecka_6d` |
| C | `pi05/pi0.5_bc_mecka_abc_6d` | kept | `pi05/pi0.5_bc_mecka_abc_6d` |
| C | `pi05/pi0.5_cotrain_eva_aria_6d` | kept | `pi05/pi0.5_cotrain_eva_aria_6d` |
| C | `pi05/pi0.5_ft_abc_eva_6d` | aliased | `pi05/pi0.5_bc_abc_eva_6d` |
| C | `pi05/pi0.5_ft_aria_fold_grip` | kept | `pi05/pi0.5_ft_aria_fold_grip` |
