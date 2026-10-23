# Code index

Every script in `code/`, with what it does. Chinese line first, English second.

## Experiments and analysis

| Script | Description |
|---|---|
| `bh_subject.py` | 稳健性验证 2：屏气对照按“被试”聚合，避免 440 段的伪重复。<br>*Robustness check 2: aggregate breath-hold by subject to avoid pseudo-replication over 440 segments.* |
| `bridge_null_criteria.py` | 桥接实验：小波准则库能否充当多径零点论文里的 bin 级判据？<br>*Bridge experiment: can the wavelet criteria library serve as a bin-level criterion for the null paper?* |
| `calibrate_criteria.py` | 无参考准则校准：读 matrix_out/matrix_raw.csv，输出 criteria_calibration.json/csv 与阈值。<br>*Reference-free criterion calibration: reads matrix_out/matrix_raw.csv, writes criteria_calibration.json/csv and thresholds.* |
| `capsule_Q.py` | capsule（100 Hz）平台的覆盖率 Q —— 三点验证的最后一点。<br>*Coverage Q on the capsule platform (100 Hz): the third of three validation points.* |
| `cardiac_null_probe.py` | 逐搏零点检验：在 cardiac 带（0.8–2.5 Hz）内看复平面轨迹与幅度凹陷。<br>*Beat-level null test: complex-plane trajectory and amplitude dips in the cardiac band (0.8-2.5 Hz).* |
| `cardiac_null_rhc.py` | RHC 谐波消除后的逐搏零点复检（复数域陷波，不破坏复平面几何）。<br>*Beat-level null recheck after RHC harmonic cancellation (complex-domain notch, geometry preserved).* |
| `closedloop_criteria.py` | b 闭环验证：只用"通过校准的无参考准则"选基，能否选到金标最优？<br>*Closed-loop check b: selecting a basis only from calibrated reference-free criteria, does it recover the gold optimum?* |
| `criteria_lib.py` | 无参考评价准则库：14 个 bin 级打分函数，供 matrix_eval 与 calibrate_criteria 调用。<br>*Reference-free criteria library: 14 bin-level scoring functions used by matrix_eval and calibrate_criteria.* |
| `decision_presentation.py` | 决策与呈现：硬门槛 → Borda 秩聚合 → Pareto 前沿；产出表 A / B / C 与四张图。<br>*Decision and presentation: hard gate -> Borda rank aggregation -> Pareto front; writes tables A/B/C and four figures.* |
| `diag_capsule_qd.py` | Q_D 在 20 条 capsule 记录上到底有没有触发？——逐 bin 复核 D 的取值。<br>*Does Q_D ever trigger on the 20 capsule records? A per-bin review of D.* |
| `emo_Q.py` | emo（100 Hz）平台补测覆盖率 Q —— 给 100 Hz 那个点加样本。<br>*Coverage Q on the emo platform (100 Hz), adding samples at the 100 Hz point.* |
| `emo_compare.py` | 跨数据集/跨域汇总：合成 · capsule · emo(vital) · emo(cumsum) 四方对照。<br>*Cross-dataset summary: synthetic, capsule, emo(vital) and emo(cumsum) side by side.* |
| `emo_diagnose.py` | emo 侧诚实诊断：把"中位 3.65"拆开，回答三个问题。<br>*Honest diagnosis on the emo side: break the median 3.65 down and answer three questions.* |
| `emo_eval.py` | emo_dataset 跨硬件测试（IWR6843 60 GHz，15 受试者 × 19 trial，PPG 金标）。<br>*emo_dataset cross-hardware test: IWR6843 at 60 GHz, 15 subjects x 19 trials, PPG gold standard.* |
| `emo_extract_volume.py` | 从单个 zip 分卷中提取完整包含的文件（分卷 zip 无法用标准库直接解压）。<br>*Extract complete files from a single split-zip volume; the standard library cannot open split archives.* |
| `emo_live_replacement.py` | 用 emo 数据替代活体采集：D 分布 + 载频裁决所需的表观摆幅。<br>*Use emo data in place of live capture: the D distribution and the apparent swing needed for the carrier-frequency argument.* |
| `emo_null_pilot.py` | emo 数据集零点试点（P01，17 个 trial）。<br>*emo null pilot, P01, 17 trials.* |
| `emo_p02_dissect.py` | 解剖 emo P02 反例：与其它被试对比信号几何、漂移速度、相位质量。<br>*Dissect the emo P02 counter-example: signal geometry, drift speed and phase quality against other subjects.* |
| `exp_A_group.py` | A 组：网格外推 / θ0 推广 / 留出验证 / 稳健性扫描。<br>*Group A: grid extrapolation, theta0 generalization, held-out test and robustness sweep.* |
| `exp_C1_emo_prospective.py` | C1：在独立前端上做前瞻检验（前半段选格，后半段验证）。<br>*C1: prospective first-half to second-half check on an independent front end.* |
| `exp_C2_shape.py` | C2：形状检验，区分几何相消与幅度调制。<br>*C2: shape test separating geometric cancellation from amplitude modulation.* |
| `exp_D_baselines.py` | D 组：把 AUC 从"自证"里救出来，并给出延迟-性能折中。<br>*Group D: rescue AUC from self-validation and quantify the delay-performance trade-off.* |
| `exp_E_synthetic.py` | E 组：四个合成实验，回答"模型是不是唯一解释"这一类质疑。<br>*Group E: four synthetic experiments answering whether the model is the only explanation.* |
| `exp_F_same_bin.py` | F 组：两个人落在同一个距离门里时，深零点会被抑制吗？<br>*Group F: are deep nulls suppressed when two people share one range bin?* |
| `exp_G_angular.py` | G 组：角度维是否真的携带可分信息？<br>*Group G: does the angular dimension actually carry separable information?* |
| `exp_audit_protocol.py` | 口径审计：① 复现 Table II 的事件定义；② 用记录级口径重算；③ 复核距离轴。<br>*Protocol audit: reproduce the Table II event definition, recompute at record level, recheck the range axis.* |
| `exp_center_removal.py` | E1：减圆心（DC offset / null point 修正）对比实验。<br>*E1: comparison of static-centre removal, i.e. DC offset or null-point correction.* |
| `exp_center_removal_t2.py` | 统一口径后的减圆心分析（与 Table II 完全同口径）。<br>*Centre-removal analysis under the unified protocol, identical to Table II.* |
| `exp_envelope_modulation.py` | 包络调制的呼吸带占比：解释"为什么常圆心压不平"。<br>*Respiration-band share of envelope modulation: why a fixed centre cannot flatten the collapse.* |
| `exp_pin_protocol.py` | 反推 Table II 的选 bin 协议。<br>*Reverse-engineer the Table II bin-selection protocol.* |
| `exp_straddle.py` | E2：straddle / scalloping 混杂控制实验。<br>*E2: straddle and scalloping confounder control.* |
| `exp_straddle_t2.py` | straddle 控制实验（V2 协议，与 Table II 同口径）。<br>*Straddle control under the V2 protocol, identical to Table II.* |
| `export_record_Q.py` | 导出 §4.13 的逐记录 Q 明细（376 行，一条记录一行）。<br>*Export the per-record Q detail of section 4.13: 376 rows, one per record.* |
| `fs_level_table.py` | 把"层数边界"写成可查表：采样率 fs → 允许的 L 区间。<br>*Tabulate the level boundary: sampling rate fs to the allowed range of L.* |
| `gen_synthetic_dataset.py` | 合成雷达位移域生命体征数据集生成器。<br>*Generator for a synthetic radar displacement-domain vital-sign dataset.* |
| `lei_D_prospective.py` | D 的前瞻性检验：前 30 s 选格并算 D，后 30 s 数深凹陷。<br>*Prospective test of D: select the bin and compute D on the first 30 s, count deep dips on the last 30 s.* |
| `lei_coverage_20hz.py` | Lei 20 Hz 平台的覆盖率 Q 全量扫描（第三个采样率点）。<br>*Full coverage-Q sweep on the Lei 20 Hz platform, the third sampling-rate point.* |
| `lei_coverage_probe.py` | Lei 20 Hz 平台的覆盖率 Q —— 先做单记录探针。<br>*Coverage Q on the Lei 20 Hz platform, starting with a single-record probe.* |
| `lei_ecg_check.py` | 用同步 ECG/PCG 验证 Lei 数据选中 bin 的生理性。<br>*Check the physiological plausibility of the selected Lei bin against synchronous ECG and PCG.* |
| `lei_ndeep_calibration.py` | n_deep 的校准：给出 R1 里"calibrated threshold"的具体数字。<br>*Calibrate n_deep, giving the numbers behind the calibrated threshold in R1.* |
| `lei_person_bin.py` | 按数据集给出的真实距离约束选 bin，复核分辨率效应。<br>*Select bins under the dataset's true range constraints and recheck the resolution effect.* |
| `lei_pilot.py` | Lei 2024 双人数据试点：同一位置、三种带宽，检验零点几何/相位与 D 的区分度。<br>*Lei 2024 dual-subject pilot: one position, three bandwidths; null geometry, phase and the discriminability of D.* |
| `lei_position1_scan.py` | Lei 数据集：位置 1 的 6 次重复 × 3 带宽扫描。<br>*Lei dataset: position 1, six repeats by three bandwidths.* |
| `lei_scan_all.py` | Lei 数据集全扫描：9 个位置 × 6 次重复 × 3 带宽。<br>*Full Lei sweep: 9 positions by 6 repeats by 3 bandwidths.* |
| `matrix_eval.py` | 候选矩阵批量评估：对每个（方法, 带, 层）组合算指标，写 manifest.csv 与 matrix_config.json。<br>*Batch evaluation of the candidate matrix: metrics for every (method, band, level) combination, written to manifest.csv and matrix_config.json.* |
| `null_alignment_gate.py` | 对齐度门槛：只把“几何零点”纳入相位锁定统计。<br>*Alignment gate: admit only geometric nulls into the phase-locking statistics.* |
| `null_crossdataset.py` | 跨数据集零点分析：capsule（20 例，100 Hz）+ Parralejo 110 人（440 段，10 Hz）。<br>*Cross-dataset null analysis on the capsule set (20 cases, 100 Hz) and Parralejo (110 subjects, 440 segments, 10 Hz).* |
| `null_crossdataset_ext.py` | 补充分析：漂移补偿相位 + ACC 运动控制。<br>*Supplementary analysis: drift-compensated phase with ACC motion control.* |
| `null_probe.py` | 多径零点检验：在 Zhang 等 2024 的 Range-FFT 立方体上直接量复平面轨迹。<br>*Null test: measure the complex trajectory directly on the Range-FFT cube of Zhang et al. 2024.* |
| `null_robustness.py` | 稳健性验证 1：漂移补偿窗口敏感性 + 漂移幅度与固定相位锁定的关系。<br>*Robustness check 1: drift-window sensitivity, and how drift amplitude relates to fixed phase locking.* |
| `parralejo_Q_A.py` | 补测 10 Hz 平台的"约定 A"（心搏取自近似带 aL）。<br>*Convention A on the 10 Hz platform, taking the beat from the approximation band aL.* |
| `parralejo_coverage.py` | 跨采样率复核：DWT 层数的"边界"是频带覆盖规则，不是数字 4。<br>*Cross-rate recheck: the DWT level boundary is a band-coverage rule, not the number 4.* |
| `parralejo_probe.py` | Parralejo 110 人数据集的前端可行性探针（10 Hz、31.2 cm 距离格）。<br>*Front-end feasibility probe on the Parralejo 110-subject dataset at 10 Hz with 31.2 cm range bins.* |
| `parralejo_probe_v2.py` | Parralejo 前端 v2：用距离像选格（而非心搏带功率），并校验 ECG 金标。<br>*Parralejo front end v2: select the bin from the range profile rather than beat-band power, and validate the ECG gold standard.* |
| `real_calibrate.py` | 真实数据上的准则校准 + 闭环：合成结论能否迁移？<br>*Criterion calibration and closed loop on real data: do the synthetic conclusions transfer?* |
| `real_capture.py` | 真实数据回放：从 capsule（RS6240，8 通道，ECG 金标）捕获生产链的位移信号。<br>*Real-data replay: capture the production displacement signal from the capsule (RS6240, 8 channels, ECG gold).* |
| `real_eval.py` | 真实数据矩阵评估：在 capsule 的真实【宽带位移】上跑缩减候选池。<br>*Real-data matrix evaluation on the capsule wideband displacement over a reduced candidate pool.* |
| `real_pilot_boundary.py` | 快速 pilot：真实信号上检验【层数结构性安全边界 L ≤ 4】是否成立。<br>*Quick pilot: does the structural level bound L <= 4 hold on real signals?* |
| `real_rc2_eval.py` | rc2（跨距离格心搏相干）在真实数据上的首次校准。<br>*First calibration of rc2, the cross-range-bin beat coherence, on real data.* |
| `rhc_frontend.py` | 生产链 RHC 前端的薄封装（T3）。<br>*Thin wrapper around the production RHC front end.* |
| `verify_parralejo_range.py` | 校验 parralejo 数据集的量程与门索引，对照 radar_chirpConfig.json。<br>*Check the Parralejo dataset range and bin indexing against radar_chirpConfig.json.* |

## Figure generation

| Script | Description |
|---|---|
| `fig_Lmap.py` | §4.13(e) 图 T8-L：允许的分解层数随采样率的变化。<br>*Figure T8-L: the allowed decomposition level as a function of sampling rate.* |
| `fig_Lmap_3d.py` | 图 T8-L3D v2：两片"覆盖率地形"用网格线绘制，避免实体曲面互相遮挡。<br>*Figure T8-L3D v2: two coverage surfaces drawn as wireframe so they do not occlude each other.* |
| `fig_band_coverage.py` | 图（§4.3 频带覆盖示意）：为什么 L ≤ 4 是安全边界。<br>*Figure for section 4.3: why L <= 4 is the safe boundary.* |
| `fig_bh_control.py` | 屏气对照图（按被试）：110 名受试者，窗内 vs 窗外深凹陷率。<br>*Breath-hold control by subject: 110 subjects, deep-dip rate inside versus outside the window.* |
| `fig_case_detail.py` | 补充材料 S2 图：capsule 逐例明细（为什么必须报分布而不是中位数）。<br>*Supplementary figure S2: capsule case detail, showing why a distribution rather than a median is reported.* |
| `fig_cross_rate.py` | 跨采样率复核图：Q(L) 与"固定 0.7 Hz 归属"在低心率下的失效。<br>*Cross-rate recheck: Q(L) and the failure of a fixed 0.7 Hz assignment at low heart rates.* |
| `fig_failure_map_v6.py` | 失效映射图 v6：摆动轴扩展到实测范围。<br>*Failure map, v6: the swing axis extended to the measured range.* |
| `fig_method_overview.py` | 图（方法总览）：小波分解参数选择 pipeline（§3 方法部分新增总览图）。<br>*Figure: overview of the wavelet parameter-selection pipeline, new in section 3.* |
| `fig_supp_subject_Q.py` | §4.13 补充材料图：逐受试者 Q 分布，以及它与心率的依赖。<br>*Supplementary figure for section 4.13: per-subject Q distribution and its dependence on heart rate.* |
| `fig_three_rates.py` | 图 T8-C：三个采样率上的覆盖率 Q，与表 T8-B 的预测边界对照。<br>*Figure T8-C: coverage Q at three sampling rates against the predicted boundary of Table T8-B.* |
| `make_graphical_abstract.py` | IEEE 图形摘要：三面板流程，660×295 像素。<br>*IEEE graphical abstract: a three-panel workflow at 660x295 px.* |
| `null_crossdataset_figure.py` | 跨数据集零点分析汇总图（三面板）。<br>*Three-panel summary figure for the cross-dataset null analysis.* |
| `null_phase_figure.py` | 《多径零点》双面板代表图：复平面轨迹 + 呼吸相位折叠。<br>*Two-panel representative figure: the complex-plane trajectory and the respiration-phase folding.* |
| `null_phase_supplement.py` | 补充材料小图：除正文代表例外，全部合格记录（深凹陷 >= 5 次）的轨迹 + 呼吸相位折叠。<br>*Supplementary figure: trajectories and phase folding for every qualifying record with at least five deep dips.* |
| `paper_figures_v3.py` | 论文 v3 的补充图：理论几何、失效域、r-D 桥、分条件屏气、对齐度、P02 反例、窗敏感性、载频。<br>*Supplementary figures for paper v3: geometry, failure region, r-D bridge, conditional breath-hold, alignment, the P02 counter-example, window sensitivity and carrier frequency.* |
| `paper_figures_v5.py` | v5 新图：分辨率效应 + 同位置三带宽轨迹（位置 2，第 2 次重复）。<br>*New figures for v5: the resolution effect and three-bandwidth trajectories at one position, position 2 repeat 2.* |
| `real_transfer_figure.py` | 迁移对比图：合成 vs 真实（capsule）四联图。<br>*Transfer figure: synthetic versus real capsule data in four panels.* |

## Tooling

| Script | Description |
|---|---|
| `md2tex.py` | 把 论文_小波基选择_v0.1.md 转成 IEEEtran 双栏 LaTeX（XeLaTeX + xeCJK）。<br>*Convert the wavelet-selection paper from Markdown to two-column IEEEtran LaTeX with XeLaTeX and xeCJK.* |
| `xref_audit.py` | 交叉引用体检：label/ref 完整性、图表编号顺序、cite 与作者名匹配<br>*Cross-reference audit: label and ref completeness, figure and table numbering order, citation-author matching.* |
