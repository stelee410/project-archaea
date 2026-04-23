import { useMemo, useState } from "react";
import clsx from "clsx";
import { api } from "../api";
import { useStore } from "../store";
import { usePersistentState } from "../hooks/usePersistentState";
import { CALIBRATION_LAMBDA_STORAGE_KEY } from "../components/CalibrationLambdaSlider";
import { SYNAPSE_GAIN_STORAGE_KEY } from "../components/SynapseGainSlider";
import type { ColonyMeta } from "../colonies/registry";
import { COLONIES } from "../colonies/registry";
import type { BudgetMode, SimConfig, TaskDifficulty } from "../types";

// Note: SimTask import previously used for the now-deleted `task` <select>;
// removed because task is decided by the chosen Colony.

interface FieldDoc {
  key: keyof SimConfig;
  label: string;
  desc: string;
  detail: string;
  defaultValue: number | string | null;
  hint?: string;
}

const FIELDS: FieldDoc[] = [
  {
    key: "seed",
    label: "随机种子 seed",
    desc: "控制初始权重、Poisson 输入、突变方向的全局 RNG。",
    detail:
      "同一 seed + 同一代码 → 行为可复现。换 seed 是验证「成功不是运气」最简单的方法。",
    defaultValue: 42,
  },
  {
    key: "pop_max",
    label: "种群硬上限 pop_max",
    desc: "槽位总数；繁殖装不下时按最低 Credit 替换。",
    detail:
      "SPEC 默认 1000。WebUI 已优化（Path2D 批量绘制 + 像素方块降级）：1000–2000 流畅，2000–5000 会自动切换为像素方块仍可读，5000+ 时主要瓶颈是 WS 带宽（每帧约 25 KB × pop_max/1000）。建议优先按生态需求选 pop_max，UI 不应是天花板。",
    defaultValue: 1000,
  },
  {
    key: "n_initial",
    label: "初始存活 n_initial",
    desc: "起步时活着的祖先数量；可远小于 pop_max 实现「最小启动」。",
    detail:
      "1 = 单祖先扩张（殖民观感强但前 20 s 命悬一线）；50–100 = 平衡。灭绝判据已自适应：人口曾到过 10+ 才会触发「跌破 10」halt。",
    defaultValue: 100,
  },
  {
    key: "carrying_capacity",
    label: "承载力 K (off-SPEC)",
    desc: "共享预算模式下的目标种群规模；留空或 0 = 关闭。",
    detail:
      "B = K × R_MAX = K × 5 单位/窗。N* ≈ 4 × K × r_mean。例如 K=30 时群体会自发收敛到 ~100 左右，pop_max 退化为安全阀。",
    defaultValue: null,
  },
  {
    key: "budget_mode",
    label: "预算模式 budget_mode",
    desc: "'none' = SPEC §4.4（每人独立奖励，资源无限）；'shared' 启用承载力。",
    detail:
      "'shared' 模拟「资源稀缺」：总需求 D 超过预算 B 时全员等比削减。强者绝对额仍多但所有人都被稀释 → 群体自发停止扩张。",
    defaultValue: "none",
  },
  {
    key: "target_speed_hz",
    label: "仿真速度 target_speed_hz",
    desc: "每实际秒推进多少个 500 ms 仿真窗；0 = 全速。",
    detail:
      "20 Hz ≈ 实时跟踪可视化；100+ Hz 用于快速堆 t_sim；全速时 WebUI 仍按数据帧率重绘。",
    defaultValue: 20,
  },
  {
    key: "calibration_lambda",
    label: "幅度校准惩罚 λ (off-SPEC)",
    desc: "0 = 纯 Pearson r（SPEC §4.1，输出常被压扁）；>0 惩罚 mean(f_out)≠mean(f_in)。",
    detail:
      "fitness = r − λ·|mean(f_out)−mean(f_in)|/std(f_in)。Pearson r 对仿射变换不敏感，所以原 SPEC 下 f_out=0.5·f_in 也能拿满分（r=1）。λ=0.3–0.5 会逼种群把斜率推向 1。可在「观测」页右上角实时调，无需重启。",
    defaultValue: 0,
  },
  {
    key: "synapse_gain",
    label: "输出层突触增益 g (off-SPEC)",
    desc: "1.0 = SPEC 默认；>1 物理放大 I_o，让输出神经元真的发更多 spike，raw f_out 直接抬高。",
    detail:
      "I_o[t] = I_in · g · Σ_j W_ho[j] · h_spike[t,j]。SPEC §1.1 的 N_OUTPUT=1 + T_REFRACTORY=2ms 决定单神经元最大 ~150 Hz；当输入 f_in 已经接近这个上限、或种群学得很「省电」时，需要 g=2~4 把幅度物理拉上去。和 λ 互补：g 抬硬上限，λ 让种群朝它走。可在「观测」页右上角实时调，无需重启。",
    defaultValue: 1,
  },
];

interface SetupPageProps {
  colony: ColonyMeta;
  onLaunched: () => void;
}

const BASE_DEFAULTS: SimConfig = {
  seed: 42,
  pop_max: 200,
  n_initial: 100,
  carrying_capacity: null,
  budget_mode: "none",
  target_speed_hz: 20,
  slime_mold: false,
  grid_size: 16,
  pheromone_decay: 0.05,
  pheromone_diffusion: 0.2,
  pheromone_emit: 0.5,
  pheromone_bonus_k: 0.5,
  hgt_enabled: true,
  hgt_prob: 0.02,
  hgt_blend: 0.3,
  migrate_enabled: true,
  migrate_prob: 0.3,
  calibration_lambda: 0.0,
  synapse_gain: 1.0,
  task: "l1",
  task_difficulty: "balanced",
};

export function SetupPage({ colony, onLaunched }: SetupPageProps) {
  const status = useStore((s) => s.status);
  const setStatus = useStore((s) => s.setStatus);
  const resetHistory = useStore((s) => s.resetHistory);

  // 每个群落用独立的 localStorage key — 切群落时配置互不污染
  const storageKey = `sim-config-${colony.id}`;
  const initialCfg: SimConfig = {
    ...BASE_DEFAULTS,
    ...(colony.configDefaults ?? {}),
    task: colony.id,  // task 由群落定义，不再由用户改
  };
  const [cfg, setCfg] = usePersistentState<SimConfig>(storageKey, initialCfg);

  // 群落切换 / 加载到旧 storage 后，强制把 task 字段对齐到当前 colony
  // （防止 localStorage 残留旧 task 字符串）
  const visibleFields = useMemo(
    () =>
      FIELDS.filter((f) => {
        if (f.key === "calibration_lambda" && colony.hideCalibrationLambda) return false;
        return true;
      }),
    [colony]
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof SimConfig>(k: K, v: SimConfig[K]) {
    setCfg((c) => {
      const next = { ...c, [k]: v } as SimConfig;
      // 切到 shared 模式时，自动给一个合理默认承载力（pop_max/5），方便点开即用。
      if (
        k === "budget_mode" &&
        v === "shared" &&
        (next.carrying_capacity == null || next.carrying_capacity <= 0)
      ) {
        next.carrying_capacity = Math.max(1, Math.round(next.pop_max / 5));
      }
      return next;
    });
  }

  const validationError =
    cfg.budget_mode === "shared" && (!cfg.carrying_capacity || cfg.carrying_capacity <= 0)
      ? "budget_mode=shared 必须填写 carrying_capacity (>0)，否则后端会拒绝启动。"
      : null;

  // 后端任意时刻只能跑一个 sim。如果当前在跑的不是本 colony，
  // 启动 = 强制替换那个群落 → 必须明确告诉用户，避免误把别的实验杀掉。
  const runningTask = status?.running ? status.config?.task : null;
  const runningOther = !!runningTask && runningTask !== colony.id;
  const runningOtherColony = runningOther
    ? COLONIES.find((c) => c.id === runningTask)
    : null;
  const isRestartSameColony = runningTask === colony.id;

  async function launch() {
    if (validationError) {
      setError(validationError);
      return;
    }
    // 二次确认：替换别的群落 / 重启同群落 都让用户点头。
    // 因为这两种操作都会丢失正在跑的 sim 的 in-memory 进展。
    if (runningOther) {
      const ok = window.confirm(
        `当前正在跑的是「${runningOtherColony?.emoji ?? ""} ${runningOtherColony?.name ?? runningTask}」` +
          `（t=${status?.t_sim?.toFixed(0) ?? 0}s，N=${status?.n_living ?? 0}）。\n\n` +
          `启动「${colony.emoji} ${colony.name}」会强制结束它，演化进度将丢失。\n\n确定要替换吗？`
      );
      if (!ok) return;
    } else if (isRestartSameColony) {
      const ok = window.confirm(
        `当前群落已经在跑（t=${status?.t_sim?.toFixed(0) ?? 0}s，N=${status?.n_living ?? 0}）。\n\n` +
          `点「重启」会用新参数从头开始，已经演化的种群会全部消失。\n\n` +
          `如果只是想换 g / λ，可以去「观测」页用滑块在线调，不用重启。\n\n确定要重启吗？`
      );
      if (!ok) return;
    }
    setBusy(true);
    setError(null);
    try {
      resetHistory();
      // task 由当前 colony 决定，覆盖任何 localStorage 残留
      const next = await api.start({ ...cfg, task: colony.id });
      setStatus(next);
      // Sim was just (re)started → discard the user's previous live-tuned
      // λ / g overrides so the sliders reflect the freshly-applied setup
      // values. Without this clearing, the sliders would keep pushing the
      // *old* values right back to the new sim on next mount.
      try {
        window.localStorage.removeItem(`archaea.${CALIBRATION_LAMBDA_STORAGE_KEY}`);
        window.localStorage.removeItem(`archaea.${SYNAPSE_GAIN_STORAGE_KEY}`);
      } catch {
        /* quota / privacy mode — non-fatal */
      }
      onLaunched();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    try {
      const next = await api.stop();
      setStatus(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-[1100px] mx-auto p-6">
      {/* Colony intro banner */}
      <div className="mb-6 rounded-lg border border-slate-700 bg-slate-900/60 p-5">
        <div className="flex items-baseline gap-3 mb-2">
          <span className="text-3xl">{colony.emoji}</span>
          <div>
            <h1 className="text-xl font-semibold text-slate-100">
              {colony.name}
            </h1>
            <div className="text-xs text-slate-500 font-mono">
              task = {colony.id} · {colony.specRef} · 难度{" "}
              <span className="text-amber-300">
                {"★".repeat(colony.difficulty)}
                <span className="text-slate-700">
                  {"☆".repeat(5 - colony.difficulty)}
                </span>
              </span>
            </div>
          </div>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
          {colony.intro}
        </p>
      </div>

      {/* 「已有别的群落在跑」的醒目警告 — 避免误启动覆盖正在做的实验 */}
      {runningOther && runningOtherColony && (
        <div className="mb-6 rounded-lg border-2 border-amber-500/50 bg-amber-950/30 p-4">
          <div className="flex items-start gap-3">
            <span className="text-2xl">⚠️</span>
            <div className="flex-1">
              <div className="text-sm font-semibold text-amber-200 mb-1">
                后端正在跑另一个群落：{runningOtherColony.emoji} {runningOtherColony.name}
              </div>
              <div className="text-xs text-amber-100/80 leading-relaxed">
                t_sim={status?.t_sim?.toFixed(1) ?? 0}s · N={status?.n_living ?? 0}/{status?.pop_max ?? 0}。
                后端任意时刻只能跑一个群落 — 在这里点「替换并启动」会强制结束它，那边的演化进度将丢失。
                如果你只是想看它跑得怎么样，
                <button
                  onClick={() => {
                    // 跳到正在跑那个 colony 的 observe 页
                    window.location.hash = ""; // 不依赖 hash router；通过自定义事件让 App 切视图
                    window.dispatchEvent(
                      new CustomEvent("archaea:goto-colony", {
                        detail: { colonyId: runningOtherColony.id, tab: "observe" },
                      })
                    );
                  }}
                  className="underline text-amber-300 hover:text-amber-200 mx-1"
                >
                  直接去观测它
                </button>
                。
              </div>
            </div>
          </div>
        </div>
      )}
      {isRestartSameColony && (
        <div className="mb-6 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-3 text-xs text-emerald-200/90">
          🟢 这个群落已经在跑（t={status?.t_sim?.toFixed(1) ?? 0}s，N={status?.n_living ?? 0}）。
          下面调任何参数后点「重启」会从头开始 —— 想看现状去
          <span className="font-semibold mx-1 text-emerald-300">「观测」</span>
          页；想在线调 g / λ 也在观测页（不用重启）。
        </div>
      )}

      <h2 className="text-base font-semibold mb-1">仿真参数</h2>
      <p className="text-sm text-slate-400 mb-4">
        修改任何参数后点击 <span className="text-emerald-300">启动 / 重启</span>。
        当前如果已有仿真在跑，会被新参数替换；客户端历史曲线随之清零。
        <span className="text-slate-500">
          {" "}下面的设置仅对该群落生效（独立 localStorage）。
        </span>
      </p>

      {(error || validationError) && (
        <div className="mb-4 px-4 py-2 rounded bg-rose-500/15 text-rose-200 text-sm border border-rose-500/30">
          {error ?? validationError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {visibleFields.map((f) => (
          <FieldRow
            key={f.key as string}
            field={f}
            value={cfg[f.key]}
            onChange={(v) => update(f.key, v as never)}
          />
        ))}
      </div>

      <div className="mt-8 flex items-center gap-3">
        <button
          disabled={busy || !!validationError}
          onClick={launch}
          className={clsx(
            "px-5 py-2 rounded-md font-medium transition-colors",
            runningOther
              ? "bg-amber-500 hover:bg-amber-400 text-slate-950"
              : "bg-emerald-500 hover:bg-emerald-400 text-slate-950",
            "disabled:opacity-40 disabled:cursor-not-allowed"
          )}
          title={
            validationError ??
            (runningOther
              ? `会强制结束「${runningOtherColony?.name}」并启动当前群落`
              : isRestartSameColony
                ? "会从头开始（演化进度归零）"
                : undefined)
          }
        >
          {runningOther
            ? "替换并启动"
            : isRestartSameColony
              ? "重启（用新参数）"
              : "启动"}
        </button>
        <button
          disabled={busy || !status?.running}
          onClick={stop}
          className={clsx(
            "px-5 py-2 rounded-md font-medium",
            "bg-slate-800 hover:bg-slate-700 text-slate-100",
            "disabled:opacity-40 disabled:cursor-not-allowed"
          )}
        >
          停止
        </button>
        <button
          onClick={() => {
            try { window.localStorage.removeItem(`archaea.${storageKey}`); } catch { /* ignore */ }
            window.location.reload();
          }}
          className="px-3 py-2 rounded-md text-xs font-medium bg-slate-800/60 hover:bg-slate-700 text-slate-300"
          title="清空该群落保存的设置并刷新页面"
        >
          重置默认
        </button>
        {status?.running && (
          <span className="text-sm text-slate-400 numeric">
            正在跑：seed={status.config?.seed} · pop_max={status.config?.pop_max}{" "}
            · t_sim={status.t_sim.toFixed(1)}s · N={status.n_living}
          </span>
        )}
      </div>

      {colony.id === "l2v2_ctrl" && (
        <DifficultyPanel
          value={cfg.task_difficulty}
          onChange={(v) => update("task_difficulty", v)}
        />
      )}

      <SlimePanel cfg={cfg} setCfg={setCfg} />

      <div className="mt-10 rounded-lg border border-slate-800 bg-slate-900/50 p-5 text-sm leading-relaxed text-slate-300">
        <h2 className="text-base font-semibold text-slate-100 mb-2">
          一些设计澄清
        </h2>
        <ul className="list-disc list-inside space-y-1.5">
          <li>
            <b>个体之间是否联通？</b>{" "}
            <span className="text-amber-300">没有</span>。SPEC §1
            规定每个 agent 是独立的 10→20→1 SNN，互相只通过共享资源/槽位竞争耦合。
            观测页点击某个 dot 会展开「该 agent 自己」的拓扑图（这才是真实存在的连接）。
          </li>
          <li>
            <b>使用页 Credit 反馈</b>会真正改变种群里被选中 agent 的 Credit。
            Credit ≤ 0 立即饿死；活下来的会进入下一窗 → 真正影响演化。
          </li>
          <li>
            <b>off-SPEC 标记</b>：承载力 / 共享预算 是 SPEC 之外的扩展，
            <code className="px-1 py-0.5 rounded bg-slate-800 mx-1">budget_mode=none</code>
            时行为与 SPEC 完全一致。
          </li>
        </ul>
      </div>
    </div>
  );
}

function FieldRow({
  field,
  value,
  onChange,
}: {
  field: FieldDoc;
  value: SimConfig[keyof SimConfig];
  onChange: (v: SimConfig[keyof SimConfig]) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="flex items-baseline justify-between gap-2">
        <label className="text-sm font-semibold text-slate-100">
          {field.label}
        </label>
        <span className="text-[11px] text-slate-500 font-mono">{field.key}</span>
      </div>
      <p className="text-xs text-slate-400 mt-1">{field.desc}</p>
      <div className="mt-2">
        {field.key === "budget_mode" ? (
          <select
            value={value as string}
            onChange={(e) => onChange(e.target.value as BudgetMode)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono"
          >
            <option value="none">none — SPEC §4.4 (默认)</option>
            <option value="shared">shared — 共享预算 (off-SPEC)</option>
          </select>
        ) : field.key === "carrying_capacity" ? (
          <input
            type="number"
            min={0}
            placeholder="留空 / 0 = 关闭"
            value={(value as number | null) ?? ""}
            onChange={(e) => {
              const v = e.target.value.trim();
              onChange((v === "" ? null : Number(v)) as never);
            }}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono numeric"
          />
        ) : field.key === "calibration_lambda" ? (
          <input
            type="number"
            min={0}
            max={5}
            step={0.05}
            value={(value as number | null) ?? 0}
            onChange={(e) => onChange(Number(e.target.value) as never)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono numeric"
          />
        ) : field.key === "synapse_gain" ? (
          <input
            type="number"
            min={0.1}
            max={20}
            step={0.1}
            value={(value as number | null) ?? 1}
            onChange={(e) => onChange(Number(e.target.value) as never)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono numeric"
          />
        ) : (
          <input
            type="number"
            value={(value as number | null) ?? ""}
            onChange={(e) => onChange(Number(e.target.value) as never)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono numeric"
          />
        )}
      </div>
      <details className="mt-2 text-xs text-slate-400">
        <summary className="cursor-pointer text-slate-500 hover:text-slate-300">
          详细解释
        </summary>
        <p className="mt-1 leading-relaxed">{field.detail}</p>
      </details>
    </div>
  );
}

interface SlimePanelProps {
  cfg: SimConfig;
  setCfg: (updater: (prev: SimConfig) => SimConfig) => void;
}

function SlimePanel({ cfg, setCfg }: SlimePanelProps) {
  function set<K extends keyof SimConfig>(k: K, v: SimConfig[K]) {
    setCfg((c) => ({ ...c, [k]: v }));
  }
  const enabled = cfg.slime_mold;
  return (
    <div className="mt-8 rounded-lg border border-fuchsia-700/40 bg-fuchsia-950/10 p-5">
      <div className="flex items-center gap-3 mb-2">
        <input
          id="slime_mold"
          type="checkbox"
          checked={enabled}
          onChange={(e) => set("slime_mold", e.target.checked)}
          className="h-4 w-4 accent-fuchsia-400"
        />
        <label htmlFor="slime_mold" className="text-base font-semibold text-fuchsia-100">
          🍄 启用赛博黏菌模式 (SPEC v1.1, off-SPEC)
        </label>
      </div>
      <p className="text-xs text-fuchsia-200/70 leading-relaxed">
        在 G×G 网格上给每个 agent 一个空间位置，引入三个相互耦合的"群体"机制：
        <br />
        ① <b>信息素场</b>（协作 / 共识形成）：强者沿轨迹释放信息素，留在浓痕处的 agent 奖励 ×(1+K)。
        <br />
        ② <b>HGT 横向基因转移</b>（社交 / 横向学习）：低 credit agent 概率性吸收邻居权重，模拟古菌真实生物机制。
        <br />
        ③ <b>趋化迁移</b>（社交 / 自组织）：agent 沿信息素梯度移动，自发形成"觅食网络"。
        <br />
        默认关闭 — 关闭时行为与 SPEC v1.0 完全一致。
      </p>

      <div
        className={clsx(
          "mt-4 grid grid-cols-2 lg:grid-cols-3 gap-3 transition-opacity",
          !enabled && "opacity-40 pointer-events-none"
        )}
      >
        <NumField label="网格边长" k="grid_size" cfg={cfg} setCfg={setCfg} step={1} />
        <NumField
          label="信息素挥发率"
          k="pheromone_decay"
          cfg={cfg}
          setCfg={setCfg}
          step={0.01}
          help="每窗 *((1−r))，0.05 ≈ 半衰期 14 窗"
        />
        <NumField
          label="扩散强度"
          k="pheromone_diffusion"
          cfg={cfg}
          setCfg={setCfg}
          step={0.05}
          help="0..1, 4-邻域拉普拉斯，0.2 比较温和"
        />
        <NumField
          label="释放速率"
          k="pheromone_emit"
          cfg={cfg}
          setCfg={setCfg}
          step={0.1}
          help="emit_rate × max(0, r) 单位/窗"
        />
        <NumField
          label="信息素奖励加成 K"
          k="pheromone_bonus_k"
          cfg={cfg}
          setCfg={setCfg}
          step={0.05}
          help="reward × (1 + K × P_local/P_max)"
        />

        <BoolField label="启用 HGT" k="hgt_enabled" cfg={cfg} setCfg={set as never} />
        <NumField
          label="HGT 概率/窗"
          k="hgt_prob"
          cfg={cfg}
          setCfg={setCfg}
          step={0.005}
          help="低 credit agent 每窗触发概率"
        />
        <NumField
          label="HGT 融合系数"
          k="hgt_blend"
          cfg={cfg}
          setCfg={setCfg}
          step={0.05}
          help="η = (1-η)·self + η·donor"
        />

        <BoolField label="启用迁移" k="migrate_enabled" cfg={cfg} setCfg={set as never} />
        <NumField
          label="迁移概率/窗"
          k="migrate_prob"
          cfg={cfg}
          setCfg={setCfg}
          step={0.05}
          help="agent 每窗考虑沿梯度移动一格的概率"
        />
      </div>
    </div>
  );
}

interface NumFieldProps {
  label: string;
  k: keyof SimConfig;
  cfg: SimConfig;
  setCfg: (u: (p: SimConfig) => SimConfig) => void;
  step?: number;
  help?: string;
}

function NumField({ label, k, cfg, setCfg, step = 0.1, help }: NumFieldProps) {
  return (
    <div className="rounded border border-fuchsia-800/30 bg-fuchsia-950/20 p-2">
      <div className="text-xs text-fuchsia-200/90">{label}</div>
      <input
        type="number"
        step={step}
        value={cfg[k] as number}
        onChange={(e) =>
          setCfg((c) => ({ ...c, [k]: Number(e.target.value) }) as SimConfig)
        }
        className="mt-1 w-full bg-slate-950 border border-fuchsia-700/40 rounded px-2 py-1 text-sm font-mono numeric"
      />
      {help && <div className="mt-1 text-[10px] text-fuchsia-200/50">{help}</div>}
    </div>
  );
}

interface BoolFieldProps {
  label: string;
  k: keyof SimConfig;
  cfg: SimConfig;
  setCfg: <K extends keyof SimConfig>(k: K, v: SimConfig[K]) => void;
}

function BoolField({ label, k, cfg, setCfg }: BoolFieldProps) {
  return (
    <div className="rounded border border-fuchsia-800/30 bg-fuchsia-950/20 p-2 flex items-center justify-between">
      <span className="text-xs text-fuchsia-200/90">{label}</span>
      <input
        type="checkbox"
        checked={cfg[k] as boolean}
        onChange={(e) => setCfg(k, e.target.checked as never)}
        className="h-4 w-4 accent-fuchsia-400"
      />
    </div>
  );
}

// ── L2v2 difficulty / environment-shaping panel ────────────────────────────

interface DifficultyOption {
  id: TaskDifficulty;
  label: string;
  pAndOne: number;
  pNotOne: number;
  silentCeiling: number;     // acc_AND a permanently-silent agent could reach
  vibe: string;              // one-liner ecological description
  warning?: string;
}

const DIFFICULTY_OPTIONS: DifficultyOption[] = [
  {
    id: "uniform",
    label: "uniform — SPEC 原版",
    pAndOne: 0.25,
    pNotOne: 0.5,
    silentCeiling: 0.75,
    vibe: "1∧1=1 占 25%。「躺平 (always-silent)」可拿 75% AND 表面正确率 — 适合复现 SPEC v2.0 默认观感。",
  },
  {
    id: "balanced",
    label: "balanced — v2.3 推荐",
    pAndOne: 0.5,
    pNotOne: 0.5,
    silentCeiling: 0.5,
    vibe: "1∧1=1 占 50%。躺平上限被打回 50%，与瞎猜无异 — 仪表盘开始能照见真相。",
  },
  {
    id: "hard",
    label: "hard — 强压力",
    pAndOne: 0.7,
    pNotOne: 0.7,
    silentCeiling: 0.3,
    vibe: "1∧1=1 占 70%。沉默个体每窗都在亏损，演化节奏明显加快 — 但需要已有少量精英才不会崩。",
    warning: "如果 acc_and_11_pop 仍为 0，建议先在 balanced 跑出火苗再切。",
  },
  {
    id: "extreme",
    label: "extreme — 实验室",
    pAndOne: 0.9,
    pNotOne: 0.9,
    silentCeiling: 0.1,
    vibe: "1∧1=1 占 90%。生态剧变 — 几乎只有真懂逻辑的个体能活；常用于「我已经有精英了，能不能稳住」的压力测试。",
    warning: "founder 弱时 ≥80% 概率全员饿死。建议 pop_max ≥ 1000 + 已建立精英子种群。",
  },
];

interface SpecialistOption {
  id: TaskDifficulty;
  label: string;
  vibe: string;
  warning?: string;
}

// SPEC_L2_V3.0 §2.4 — separate sub-group for "single-gate dishes".
// Cleanly separated from the gradient above so the user understands these
// don't sit on the difficulty axis — they switch the *task distribution*.
const SPECIALIST_OPTIONS: SpecialistOption[] = [
  {
    id: "and_only",
    label: "and_only — AND 学家培养皿",
    vibe:
      "100% 时间考 AND（P(AND)=1）。不再问 NOT，群体只能演化"
      + "「两路同高 → 输出高」一种逻辑。用来养一锅纯 AND 学家菌株 → 💾 保存 → 倒进 ⚗️ Mixer。",
    warning:
      "保存出的菌株只会 AND；NOT 通路可能完全没发育（甚至被选择掉）。"
      + "想要它真懂 NOT 必须再单独跑一锅 not_only。",
  },
  {
    id: "not_only",
    label: "not_only — NOT 学家培养皿",
    vibe:
      "100% 时间考 NOT（P(AND)=0）。所有题都是「输入低 → 输出高 / 输入高 → 输出低」，"
      + "群体只能往抑制电路这一条路演化。用来养纯 NOT 学家菌株。",
    warning:
      "NOT 在生物上比 AND 更难涌现（需要抑制突触 + 选择器响应），"
      + "founder 弱时容易卡 ~50% 一段时间。建议 synapse_gain ≥ 2.5、pop_max ≥ 1000、耐心 30 分钟+。",
  },
];

function DifficultyPanel({
  value,
  onChange,
}: {
  value: TaskDifficulty;
  onChange: (v: TaskDifficulty) => void;
}) {
  const current =
    DIFFICULTY_OPTIONS.find((o) => o.id === value) ??
    SPECIALIST_OPTIONS.find((o) => o.id === value) ??
    DIFFICULTY_OPTIONS[1];
  const isSpecialist = SPECIALIST_OPTIONS.some((o) => o.id === value);
  return (
    <div className="mt-8 rounded-lg border border-amber-700/40 bg-amber-950/10 p-5">
      <div className="flex items-baseline gap-3 mb-2">
        <span className="text-base font-semibold text-amber-100">
          🎚️ 出题分布 — 塑造环境
        </span>
        <span className="text-[11px] text-amber-200/60 font-mono">
          task_difficulty · 调采样比例 · 不动奖励规则
        </span>
      </div>
      <p className="text-xs text-amber-200/80 leading-relaxed">
        这里改的是<b>题目分布</b>（环境）而不是奖励表（规则）—
        所以你是「上帝」不是「工程师」。
      </p>

      <div className="mt-4">
        <div className="text-[11px] uppercase tracking-wider text-amber-200/60 mb-1.5">
          ① 混合任务 — 同时考 AND + NOT，调难度梯度
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {DIFFICULTY_OPTIONS.map((opt) => {
            const active = opt.id === value;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onChange(opt.id)}
                className={clsx(
                  "text-left rounded border px-3 py-2 transition-colors",
                  active
                    ? "border-amber-400 bg-amber-500/15 text-amber-100"
                    : "border-amber-800/40 bg-amber-950/20 text-amber-200/80 hover:bg-amber-900/30"
                )}
              >
                <div className="text-xs font-semibold">{opt.label}</div>
                <div className="mt-1 text-[10px] font-mono numeric text-amber-200/70">
                  P(target=1|AND)={opt.pAndOne.toFixed(2)} · 躺平上限{" "}
                  {(opt.silentCeiling * 100).toFixed(0)}%
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-4">
        <div className="text-[11px] uppercase tracking-wider text-cyan-200/70 mb-1.5">
          ② 单门培养皿 — ⚗️ 杂交皿专用，只考一种逻辑
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
          {SPECIALIST_OPTIONS.map((opt) => {
            const active = opt.id === value;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onChange(opt.id)}
                className={clsx(
                  "text-left rounded border px-3 py-2 transition-colors",
                  active
                    ? "border-cyan-400 bg-cyan-500/15 text-cyan-50"
                    : "border-cyan-800/40 bg-cyan-950/20 text-cyan-200/80 hover:bg-cyan-900/30"
                )}
              >
                <div className="text-xs font-semibold">{opt.label}</div>
                <div className="mt-1 text-[10px] font-mono numeric text-cyan-200/70">
                  P(AND) = {opt.id === "and_only" ? "1.00" : "0.00"} · 单一专家
                </div>
              </button>
            );
          })}
        </div>
        <div className="mt-2 text-[11px] text-cyan-200/70 leading-relaxed">
          典型流程：先用 <code className="px-1 rounded bg-cyan-900/40">and_only</code> 跑出
          AND 学家 → 观测页 💾 保存为菌株 →
          换 <code className="px-1 rounded bg-cyan-900/40">not_only</code> + 新 seed 跑出 NOT
          学家 → 💾 保存 → 回图鉴进 ⚗️ <b>Mixer</b> 把两个菌株倒在一起做杂交。
        </div>
      </div>

      <div
        className={clsx(
          "mt-4 rounded border p-3",
          isSpecialist
            ? "border-cyan-800/40 bg-cyan-950/30"
            : "border-amber-800/40 bg-amber-950/30"
        )}
      >
        <div
          className={clsx(
            "text-xs leading-relaxed",
            isSpecialist ? "text-cyan-100" : "text-amber-100"
          )}
        >
          <b>当前选择：</b>
          <span className={isSpecialist ? "text-cyan-200" : "text-amber-200"}>
            {current.label}
          </span>
        </div>
        <div
          className={clsx(
            "mt-1 text-[12px] italic leading-relaxed",
            isSpecialist ? "text-cyan-100/80" : "text-amber-100/80"
          )}
        >
          {current.vibe}
        </div>
        {current.warning && (
          <div
            className={clsx(
              "mt-2 text-[11px] border-l-2 pl-2",
              isSpecialist
                ? "text-cyan-300/90 border-cyan-400/60"
                : "text-amber-300/90 border-amber-400/60"
            )}
          >
            ⚠ {current.warning}
          </div>
        )}
      </div>

      <details className="mt-3 text-xs text-amber-200/70">
        <summary className="cursor-pointer text-amber-200/60 hover:text-amber-200">
          为什么要给「上帝」一个滑块？
        </summary>
        <p className="mt-2 leading-relaxed">
          奖励规则（reward table）一旦改动，原本的演化结果就不再可比 — 那是「换游戏」。
          采样分布只是改了「这局牌发什么」 — 古菌还要靠自己进化的策略来活下来。
          所以这个滑块的哲学是：让用户能塑造生态压力的<b>来源</b>，而不是干预演化的<b>过程</b>。
          配合下方仪表盘的 <code className="px-1 rounded bg-amber-900/40">acc_and_11_pop</code>{" "}
          指标，你能直接看到「沉默」与「真懂」的分界线。
        </p>
      </details>
    </div>
  );
}
