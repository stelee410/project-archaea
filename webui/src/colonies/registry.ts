/**
 * 群落注册表（Colony Registry）
 *
 * 一个「群落」= 一个独立的演化任务环境。L1 / L2v2 / 未来的 L3 ... 各自登记一行
 * `ColonyMeta`，前端按这个表决定：
 *
 *   - 进入「图鉴」(ColonyPicker) 后展示什么卡片
 *   - 进入某个群落后，设置 / 观测 / 使用 三页是否需要插入任务专属面板
 *   - 是否隐藏与本任务无关的通用控件（如 L2v2 隐藏 calibration_lambda）
 *
 * 加新群落的成本：
 *   1. archaea/task.py   — 后端注册新 task 字符串
 *   2. archaea/oracle_X.py + population.py 分支 — 后端实现
 *   3. webui/src/colonies/<id>/  — 前端可选的 Dashboard / UseExtras
 *   4. 在下面的 COLONIES 数组里追加一条
 *
 * 不动 SetupPage / ObservePage / UsePage / App 一行代码。
 */
import type React from "react";
import type { SimConfig, SimTask, TelemetryEvent } from "../types";
import { L2Dashboard } from "./l2v2/Dashboard";
import { LogicTester } from "./l2v2/UseExtras";

// ── Slot prop shapes（任务专属面板能拿到的上下文） ───────────────────────────
export interface DashboardProps {
  ev: TelemetryEvent | null;
}

// 任务专属「使用」面板的 props。
// 历史上这里把 target / topK / swarmRadius / durationMs / warmupMs 透传过来，
// 让任务面板与 L1 通用「底层接口」共用同一份 form 值。L2v2 之后把这些查询
// 控件搬进了任务面板自身（LogicTester 内置了选择器），所以 props 现在留空，
// 任务面板自己 usePersistentState 管理。
export type UseExtrasProps = Record<string, never>;

// ── 群落定义 ───────────────────────────────────────────────────────────────
export interface ColonyMeta {
  id: SimTask;
  name: string;            // 中文学名，如「频率跟随者」
  emoji: string;           // 图鉴上的徽章
  oneLiner: string;        // 卡片副标题
  difficulty: 1 | 2 | 3 | 4 | 5;
  specRef: string;         // 来源章节
  intro: string;           // 进入群落后顶部介绍（多行可用 \n）
  // 启动该群落时强行覆盖到 SimConfig 的字段（用于群落专属推荐默认）
  configDefaults?: Partial<SimConfig>;
  // 通用控件可见性
  hideCalibrationLambda?: boolean;  // L1 专属的 r 校准滑块
  hideBudgetChart?: boolean;        // L1 共享预算图
  // 任务专属插槽
  Dashboard?: React.FC<DashboardProps>;     // 观测页顶部
  UseExtras?: React.FC<UseExtrasProps>;     // 使用页顶部
  // 「使用」页：是否把通用 Hz 输入区降级为「高级 / 调试」折叠
  demoteLegacyUseInput?: boolean;
  // 「使用」页：是否完全隐藏 L1 频率工具（底层接口 / 打分 / 鼠标跟随 / Sweep）。
  // 比 demoteLegacyUseInput 更激进——连折叠入口都不显示。
  // 仅当任务专属 UseExtras 已经自带查询目标 / 输入构造器时才打开，否则
  // 用户连 target/topK 都没法调。
  hideLegacyUseTools?: boolean;
  // 卡片状态用：是否「待解锁」(后端 task 没注册时)
  locked?: boolean;
  unlockHint?: string;
}

export const COLONIES: ColonyMeta[] = [
  {
    id: "l1",
    name: "频率跟随者 · Echo",
    emoji: "🎵",
    oneLiner: "学会模仿外界节律 — f_in 进，f_out 跟着抖",
    difficulty: 1,
    specRef: "SPEC §3.1",
    intro:
      "最古老的题目：环境每 500ms 抛出一个频率 f_in（10–100 Hz），群落要让自己唯一的输出神经元发放率 f_out 跟着走。" +
      "整个 220 个突触都围绕这一个目标演化。\n" +
      "Fitness = Pearson r(f_in, f_out)。可选「幅度校准 λ」把斜率压向 1。是 Project Archaea 的基础课。",
  },
  {
    id: "l2v2_ctrl",
    name: "逻辑门控者 · Switcher",
    emoji: "🧠",
    oneLiner: "按指令切换 AND / NOT — 真值表打分",
    difficulty: 3,
    specRef: "SPEC_L2_V2.0",
    intro:
      "在「模仿」之上多了两条输入：B（第二路数据）和 S（指令）。S 的频率告诉群落「现在考你 AND 还是 NOT」，" +
      "群落必须根据 S 切换内部计算路径。引入抑制突触（权重 ∈ [-1.5, 1.5]）让「否决 / NOT」逻辑成为可能。\n" +
      "Oracle 真值表给奖励（v2.2 软着陆）：AND(1,1)=+15、NOT(0,*)=+25（高难溢价）；" +
      "silent 正确 +2.0/+2.5。\n" +
      "ERRATA v2.3 加权采样：1∧1 占 AND 题的 50%、NOT 0 占 NOT 题的 50%，" +
      "总体 P(target=1)=50%。silent 天花板从 75% 降到 50% — 偷懒不再能伪装高分；" +
      "silent 净亏 0.125/win 慢慢淘汰，完美 +9.9/win 暴富。Fitness = mean(acc_AND, acc_NOT)。\n" +
      "想专门按 AND 或 NOT 训练？翻到设置页下方的「② 单门培养皿」选 " +
      "and_only / not_only — 这是 ⚗️ 杂交皿(SPEC_L2_V3.0)的标准前置流程。",
    configDefaults: {
      // 输出 bit 判定靠 f_out > OUT_SPIKING_THRESHOLD_HZ (v2.4: 20Hz, 原 50Hz)；
      // 默认 g=1 经常压不上去，群落跑出的"对答案"会被阈值砍掉。
      // L2v2 推荐先把 g 抬高，再视觉化结果。
      synapse_gain: 2.0,
    },
    hideCalibrationLambda: true,
    hideBudgetChart: true,
    Dashboard: L2Dashboard,
    UseExtras: LogicTester,
    demoteLegacyUseInput: true,
    hideLegacyUseTools: true,
  },
  // ── 占位：未来的群落 ──
  // 例如：
  // { id: "l3", name: "...", locked: true, unlockHint: "完成 L2v2 的 24h 收敛后解锁", ... }
];

export function getColony(id: SimTask): ColonyMeta {
  const c = COLONIES.find((x) => x.id === id);
  if (!c) throw new Error(`unknown colony: ${id}`);
  return c;
}
