import { useEffect, useState } from "react";
import clsx from "clsx";
import { api, openTelemetry } from "./api";
import { useStore } from "./store";
import { SetupPage } from "./pages/SetupPage";
import { ObservePage } from "./pages/ObservePage";
import { UsePage } from "./pages/UsePage";

type Tab = "setup" | "observe" | "use";

export default function App() {
  const [tab, setTab] = useState<Tab>("setup");
  const status = useStore((s) => s.status);
  const wsState = useStore((s) => s.wsState);
  const setStatus = useStore((s) => s.setStatus);
  const setWs = useStore((s) => s.setWs);
  const ingest = useStore((s) => s.ingest);

  useEffect(() => {
    api.status().then(setStatus).catch(() => undefined);
    const ws = openTelemetry({
      onOpen: () => setWs(ws, "open"),
      onClose: () => setWs(null, "closed"),
      onError: () => setWs(ws, "error"),
      onMessage: (m) => ingest(m),
    });
    setWs(ws, "connecting");
    return () => ws.close();
  }, [setStatus, setWs, ingest]);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-10">
        <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-6">
          <div className="font-semibold text-lg tracking-wide">
            Project Archaea
            <span className="ml-2 text-xs text-slate-400 font-normal">L1 · WebUI</span>
          </div>
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
                  tab === k
                    ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/40"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60"
                )}
              >
                {label}
              </button>
            ))}
          </nav>
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
                    ? `运行中 t=${status.t_sim.toFixed(1)}s N=${status.n_living}/${status.pop_max}`
                    : "未启动"
                  : "?"
              }
            />
          </div>
        </div>
      </header>
      <main className="flex-1 min-h-0">
        {tab === "setup" && <SetupPage onLaunched={() => setTab("observe")} />}
        {tab === "observe" && <ObservePage />}
        {tab === "use" && <UsePage />}
      </main>
    </div>
  );
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
