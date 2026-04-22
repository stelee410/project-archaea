import clsx from "clsx";
import { useStore } from "../store";
import type { ColonyMeta } from "../colonies/registry";
import { COLONIES } from "../colonies/registry";

interface Props {
  onPick: (colony: ColonyMeta) => void;
}

/**
 * 「群落图鉴」入口页 — 进入 WebUI 看到的第一屏。
 *
 * 每张卡片对应一个演化任务：L1（频率跟随）、L2v2（逻辑门控）、未来的 L3...
 * 点击卡片 → 进入对应群落的 setup/observe/use 三页。
 *
 * 卡片状态：
 *   🟢 当前 sim 正在跑且 task 匹配 → 显示 t_sim/N_living，按钮「→ 进入培养皿」
 *   ⚠️ 当前 sim 在跑但 task 不同   → 显示「占用中（点击会重启）」
 *   ⚪ 当前 sim 未启动              → 「→ 进入培养皿」
 *   🔒 群落 locked                  → 显示「待解锁」+ unlockHint
 */
export function ColonyPicker({ onPick }: Props) {
  const status = useStore((s) => s.status);
  const runningTask = status?.running ? status.config?.task ?? null : null;

  return (
    <div className="max-w-[1400px] mx-auto p-8">
      <div className="mb-8 text-center">
        <div className="text-3xl font-bold text-slate-100 mb-2">
          🧫 Project Archaea · 古菌群落图鉴
        </div>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto">
          每个群落是一个独立的演化培养皿——同样的代谢机制、同样的繁殖律，
          但被放进不同的环境去解不同的题。点开任意卡片进入它的设置 / 观测 / 使用界面。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {COLONIES.map((c) => (
          <ColonyCard
            key={c.id}
            colony={c}
            running={runningTask === c.id}
            occupiedByOther={runningTask != null && runningTask !== c.id}
            onPick={() => onPick(c)}
            currentT={status?.t_sim ?? 0}
            currentN={status?.n_living ?? 0}
            currentPopMax={status?.pop_max ?? 0}
          />
        ))}
        <FuturePlaceholder />
      </div>

      <div className="mt-10 text-center text-[11px] text-slate-500">
        共 {COLONIES.length} 个已实现群落 · 后端 task 注册见{" "}
        <code className="px-1 bg-slate-800 rounded">archaea/task.py</code>
      </div>
    </div>
  );
}

interface ColonyCardProps {
  colony: ColonyMeta;
  running: boolean;
  occupiedByOther: boolean;
  onPick: () => void;
  currentT: number;
  currentN: number;
  currentPopMax: number;
}

function ColonyCard({
  colony,
  running,
  occupiedByOther,
  onPick,
  currentT,
  currentN,
  currentPopMax,
}: ColonyCardProps) {
  const locked = colony.locked;
  return (
    <button
      disabled={locked}
      onClick={onPick}
      className={clsx(
        "group text-left rounded-xl border-2 p-6 transition-all",
        locked
          ? "border-slate-800 bg-slate-900/30 opacity-50 cursor-not-allowed"
          : running
            ? "border-emerald-500/60 bg-emerald-950/20 hover:bg-emerald-950/40 hover:border-emerald-400"
            : "border-slate-700 bg-slate-900/50 hover:border-sky-500/60 hover:bg-slate-900/80"
      )}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-baseline gap-3">
          <span className="text-4xl">{colony.emoji}</span>
          <div>
            <div className="text-lg font-semibold text-slate-100 leading-tight">
              {colony.name}
            </div>
            <div className="text-[11px] text-slate-500 font-mono mt-0.5">
              {colony.id} · {colony.specRef}
            </div>
          </div>
        </div>
        <DifficultyStars n={colony.difficulty} />
      </div>

      <p className="text-sm text-slate-300 mb-4 leading-relaxed">
        {colony.oneLiner}
      </p>

      {/* status badge */}
      <div className="flex items-center justify-between">
        {locked ? (
          <span className="text-xs text-slate-500">🔒 {colony.unlockHint ?? "待解锁"}</span>
        ) : running ? (
          <span className="text-xs text-emerald-300 font-mono">
            🟢 正在跑 · t={currentT.toFixed(0)}s · N={currentN}/{currentPopMax}
          </span>
        ) : occupiedByOther ? (
          <span className="text-xs text-amber-300">
            ⚠️ 当前 sim 占用中（进入会重启）
          </span>
        ) : (
          <span className="text-xs text-slate-500">⚪ 空闲</span>
        )}
        {!locked && (
          <span className="text-sm font-medium text-sky-300 group-hover:translate-x-1 transition-transform">
            → 进入培养皿
          </span>
        )}
      </div>
    </button>
  );
}

function DifficultyStars({ n }: { n: number }) {
  return (
    <div className="text-xs font-mono text-amber-300 leading-none mt-1" title={`难度 ${n}/5`}>
      {"★".repeat(n)}
      <span className="text-slate-700">{"☆".repeat(5 - n)}</span>
    </div>
  );
}

function FuturePlaceholder() {
  return (
    <div className="rounded-xl border-2 border-dashed border-slate-800 p-6 flex flex-col items-center justify-center text-center text-slate-500 min-h-[180px]">
      <div className="text-3xl mb-2 opacity-40">🌀</div>
      <div className="text-sm font-medium">未来群落</div>
      <div className="text-[11px] mt-1 max-w-[260px] leading-relaxed">
        当 L2v2 跑通后会解锁更高阶任务。
        加新群落只需在{" "}
        <code className="text-slate-400">colonies/registry.ts</code> 追加一行。
      </div>
    </div>
  );
}
