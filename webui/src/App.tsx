import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { api, openTelemetry } from "./api";
import { useStore } from "./store";
import { SetupPage } from "./pages/SetupPage";
import { ObservePage } from "./pages/ObservePage";
import { UsePage } from "./pages/UsePage";
import { ColonyPicker } from "./pages/ColonyPicker";
import { COLONIES, getColony } from "./colonies/registry";
import type { ColonyMeta } from "./colonies/registry";
import type { SimTask } from "./types";

type Tab = "setup" | "observe" | "use";

type View =
  | { kind: "picker" }
  | { kind: "colony"; colonyId: SimTask; tab: Tab };

export default function App() {
  const [view, setView] = useState<View>({ kind: "picker" });
  const status = useStore((s) => s.status);
  const wsState = useStore((s) => s.wsState);
  const setStatus = useStore((s) => s.setStatus);
  const setWs = useStore((s) => s.setWs);
  const ingest = useStore((s) => s.ingest);
  // 「页面刷新后自动恢复到正在跑的群落」只能发生一次。
  // 如果用户主动从 colony 页点「← 图鉴」回来，我们绝不再硬把他踹回去；
  // 因为这意味着他想看图鉴或者准备启动另一个群落。
  const autoEnteredRef = useRef(false);

  // ── WebSocket 自动重连 ────────────────────────────────────────────────
  // 之前挂了就再也连不上（vite HMR / 后端重启 / 网络抖动都会触发 close），
  // 只能刷页面，体验非常糟。这里加指数退避：1s → 2s → 4s → 8s → 上限 15s。
  // 重连成功后会自动再拉一次 status，恢复 N=… 数字。
  const wsBoxRef = useRef<{ ws: WebSocket | null; cancelled: boolean }>({
    ws: null,
    cancelled: false,
  });
  useEffect(() => {
    const box = wsBoxRef.current;
    box.cancelled = false;
    let attempt = 0;
    let retryTimer: number | null = null;

    const refreshStatus = () => {
      api.status().then(setStatus).catch(() => undefined);
    };

    const connect = () => {
      if (box.cancelled) return;
      const ws = openTelemetry({
        onOpen: () => {
          attempt = 0;
          setWs(ws, "open");
          // 连上后立刻同步一次 sim status，避免显示陈旧的 t_sim/N
          refreshStatus();
        },
        onClose: () => {
          if (box.cancelled) return;
          setWs(null, "closed");
          const delay = Math.min(15000, 1000 * Math.pow(2, attempt));
          attempt += 1;
          retryTimer = window.setTimeout(connect, delay);
        },
        onError: () => {
          // onerror 后浏览器一定会再触发 onclose，所以重连逻辑统一放 close 里。
          setWs(ws, "error");
        },
        onMessage: (m) => ingest(m),
      });
      box.ws = ws;
      setWs(ws, "connecting");
    };

    refreshStatus();
    connect();

    return () => {
      box.cancelled = true;
      if (retryTimer != null) window.clearTimeout(retryTimer);
      box.ws?.close();
    };
  }, [setStatus, setWs, ingest]);

  // ── 刷新后自动恢复到正在发展的群落 ─────────────────────────────────────
  // 行为：mount 后第一次拿到 status 时，如果发现有任务正在跑，
  // 自动从图鉴跳进对应 colony 的 observe 页（沿用群落卡片上的「→ 进入培养皿」逻辑）。
  // 这样用户刷新浏览器、或者重新打开 tab 时，能直接看到生态延续，
  // 而不是误以为「需要重新启动」并因此触发冲突。
  useEffect(() => {
    if (autoEnteredRef.current) return;
    if (view.kind !== "picker") return;
    if (!status?.running) return;
    const taskId = status.config?.task as SimTask | undefined;
    if (!taskId) return;
    const found = COLONIES.find((c) => c.id === taskId);
    if (!found) return;
    autoEnteredRef.current = true;
    setView({ kind: "colony", colonyId: found.id, tab: "observe" });
  }, [status, view.kind]);

  // ── 跨组件跳转桥：SetupPage 上的「直接去观测它」链接通过 window event 调用 ──
  // 用 CustomEvent 的好处：避免把 setView 通过 props 一路下钻到所有子组件。
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ colonyId: SimTask; tab?: Tab }>).detail;
      if (!detail?.colonyId) return;
      const found = COLONIES.find((c) => c.id === detail.colonyId);
      if (!found) return;
      setView({ kind: "colony", colonyId: found.id, tab: detail.tab ?? "observe" });
    };
    window.addEventListener("archaea:goto-colony", handler as EventListener);
    return () => window.removeEventListener("archaea:goto-colony", handler as EventListener);
  }, []);

  function pickColony(c: ColonyMeta) {
    // 进入群落时的默认 tab：如果该群落正在跑就直接去观测，否则去设置
    const runningId = status?.running ? status.config?.task : null;
    const defaultTab: Tab = runningId === c.id ? "observe" : "setup";
    setView({ kind: "colony", colonyId: c.id, tab: defaultTab });
  }

  function backToPicker() {
    setView({ kind: "picker" });
  }

  function setTab(tab: Tab) {
    setView((v) => (v.kind === "colony" ? { ...v, tab } : v));
  }

  const colony =
    view.kind === "colony" ? getColony(view.colonyId) : null;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-10">
        <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-4 flex-wrap">
          {view.kind === "colony" && (
            <button
              onClick={backToPicker}
              className="px-2 py-1 rounded text-xs bg-slate-800 hover:bg-slate-700 text-slate-300"
              title="返回群落图鉴"
            >
              ← 图鉴
            </button>
          )}
          <button
            onClick={backToPicker}
            className="font-semibold text-lg tracking-wide text-left hover:opacity-80"
            title="回到主页"
          >
            🧫 Project Archaea
            {colony && (
              <span className="ml-3 text-sm text-slate-300 font-normal">
                · {colony.emoji} {colony.name}
              </span>
            )}
            {!colony && (
              <span className="ml-2 text-xs text-slate-400 font-normal">· 群落图鉴</span>
            )}
          </button>
          {view.kind === "colony" && (
            <nav className="flex gap-1">
              {(
                [
                  ["setup", "设置 / 启动"],
                  ["observe", "观测"],
                  ["use", "使用"],
                ] as [Tab, string][]
              ).map(([k, label]) => (
                <button
                  key={k}
                  onClick={() => setTab(k)}
                  className={clsx(
                    "px-3 py-1.5 rounded-md text-sm transition-colors",
                    view.tab === k
                      ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/40"
                      : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60"
                  )}
                >
                  {label}
                </button>
              ))}
            </nav>
          )}
          <div className="ml-auto flex items-center gap-4 text-xs">
            <Indicator
              label="WS"
              ok={wsState === "open"}
              warn={wsState === "connecting"}
              text={wsStateLabel(wsState)}
            />
            <Indicator
              label="SIM"
              ok={!!status?.running}
              text={
                status
                  ? status.running
                    ? `${labelForTask(status.config?.task)} · t=${status.t_sim.toFixed(1)}s · N=${status.n_living}/${status.pop_max}`
                    : "未启动"
                  : "?"
              }
            />
          </div>
        </div>
      </header>
      <main className="flex-1 min-h-0">
        {view.kind === "picker" && <ColonyPicker onPick={pickColony} />}
        {view.kind === "colony" && colony && view.tab === "setup" && (
          <SetupPage
            colony={colony}
            onLaunched={() => setTab("observe")}
          />
        )}
        {view.kind === "colony" && colony && view.tab === "observe" && (
          <ObservePage colony={colony} />
        )}
        {view.kind === "colony" && colony && view.tab === "use" && (
          <UsePage colony={colony} />
        )}
      </main>
    </div>
  );
}

function labelForTask(task: SimTask | undefined): string {
  if (!task) return "?";
  const c = COLONIES.find((x) => x.id === task);
  return c ? `${c.emoji} ${c.id}` : task;
}

function wsStateLabel(s: string): string {
  switch (s) {
    case "open":
      return "在线";
    case "connecting":
      return "连接中";
    case "closed":
      return "已断开";
    case "error":
      return "错误";
    default:
      return "空闲";
  }
}

function Indicator({
  label,
  ok,
  warn,
  text,
}: {
  label: string;
  ok?: boolean;
  warn?: boolean;
  text: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={clsx(
          "inline-block w-2 h-2 rounded-full",
          ok
            ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
            : warn
              ? "bg-amber-400"
              : "bg-rose-400"
        )}
      />
      <span className="text-slate-400">{label}</span>
      <span className="numeric text-slate-200">{text}</span>
    </div>
  );
}
