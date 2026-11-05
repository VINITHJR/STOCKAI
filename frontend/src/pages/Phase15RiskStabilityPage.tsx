import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, BarChart3, CheckCircle2, RefreshCw, ShieldCheck, TrendingDown, Waves } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getOverview, getReport, type Overview, type Row } from "../services/phase15Api";

const names: Record<string,string> = {
  extra_trees_top5:"Extra Trees · Top 5", random_forest_top5:"Random Forest · Top 5",
  equal_weight_20_integer_redistributed:"Equal Weight · 20",
};
const n = (r: Row | undefined, k: string): number | null => typeof r?.[k] === "number" ? r[k] as number : null;
const f = (v: number | null | undefined, d=2) => v == null ? "—" : v.toLocaleString("en-IN",{maximumFractionDigits:d});
const money = (v: number | null | undefined) => v == null ? "—" : `₹${v.toLocaleString("en-IN",{maximumFractionDigits:0})}`;
const formatSessionDate = (value: string) => {
 const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value));
 if(!match)return String(value);
 const date = new Date(Number(match[1]),Number(match[2])-1,Number(match[3]));
 return date.toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"});
};

function Metric({label,value,note,Icon}:{label:string;value:string;note:string;Icon:typeof Activity}) {
 return <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5"><div className="flex justify-between text-xs uppercase text-slate-400">{label}<Icon size={17}/></div><b className="mt-3 block text-2xl text-white">{value}</b><p className="mt-1 text-xs text-slate-500">{note}</p></section>;
}
function Panel({title,note,children}:{title:string;note?:string;children:React.ReactNode}) {
 return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5"><h2 className="font-semibold text-slate-100">{title}</h2>{note&&<p className="mb-4 mt-1 text-xs text-slate-500">{note}</p>}{children}</section>;
}
type InstrumentDisplay = { instrument_key: string; symbol: string; name: string };
const Empty = () => <p className="py-8 text-center text-xs text-slate-500">No rows for this selection.</p>;

export default function Phase15RiskStabilityPage() {
 const [overview,setOverview]=useState<Overview|null>(null);
 const [data,setData]=useState<Record<string,Row[]>>({});
 const [portfolio,setPortfolio]=useState("extra_trees_top5");
 const [slippage,setSlippage]=useState(20);
 const [dateRange,setDateRange]=useState({from:"",to:""});

 const [loading,setLoading]=useState(true);
 const [error,setError]=useState("");
 const [instrumentMap,setInstrumentMap]=useState<Record<string,InstrumentDisplay>>({});
 const load=useCallback(async()=>{
  setLoading(true);setError("");
  try {
   const keys=["daily_risk_series","drawdown_episodes","market_regime_exposure","fold_calendar_risk","stock_risk_concentration","reliability_exposure","source_metric_reconciliation"];
   const [o,...reports]=await Promise.all([getOverview(),...keys.map(getReport)]);
   setOverview(o);setData(Object.fromEntries(keys.map((k,i)=>[k,reports[i].rows])));
   try {
    const mapResponse = await fetch(`${import.meta.env.BASE_URL}data/phase15/instrument_display_map.json`, { cache: "no-cache" });
    if (!mapResponse.ok) throw new Error(`Instrument map request failed (${mapResponse.status})`);
    const mapData = await mapResponse.json() as Record<string,InstrumentDisplay>;
    setInstrumentMap(mapData);
   } catch (mapError) {
    console.warn("Phase 15 instrument display map unavailable; showing instrument keys only.", mapError);
    setInstrumentMap({});
   }
  } catch(e) {setError(e instanceof Error?e.message:"Could not load Phase 15 reports.");}
  finally {setLoading(false);}
 },[]);
 useEffect(()=>{void load();},[load]);
 const metric=overview?.portfolio_risk_metrics.find(r=>r.portfolio===portfolio&&r.slippage_bps_per_side===slippage);
 const rows=(key:string)=>(data[key]??[]).filter(r=>r.portfolio===portfolio&&r.slippage_bps_per_side===slippage);
 const daily=useMemo(()=>rows("daily_risk_series").sort((a,b)=>String(a.session_date).localeCompare(String(b.session_date))),[data,portfolio,slippage]);
 const firstSession = daily.length ? String(daily[0].session_date ?? "") : "";
 const lastSession = daily.length ? String(daily[daily.length-1].session_date ?? "") : "";
 const chartFrom=dateRange.from||firstSession;
 const chartTo=dateRange.to||lastSession;
 useEffect(()=>{
  if(!firstSession||!lastSession)return;
  setDateRange(current=>{
   const validFrom=current.from>=firstSession&&current.from<=lastSession?current.from:firstSession;
   const validTo=current.to>=firstSession&&current.to<=lastSession?current.to:lastSession;
   if(validFrom<=validTo)return {from:validFrom,to:validTo};
   return {from:firstSession,to:lastSession};
  });
 },[firstSession,lastSession]);
 const chartDaily=useMemo(()=>daily.filter(row=>{
  const session=String(row.session_date??"").slice(0,10);
  return session>=chartFrom&&session<=chartTo;
 }),[daily,chartFrom,chartTo]);

 const episodes=rows("drawdown_episodes"), regimes=rows("market_regime_exposure");
 const folds=rows("fold_calendar_risk").sort((a,b)=>Number(a.fold_number)-Number(b.fold_number));
 const concentration=rows("stock_risk_concentration").sort((a,b)=>(n(b,"share_of_all_negative_trade_loss_pct")??0)-(n(a,"share_of_all_negative_trade_loss_pct")??0)).slice(0,10);
 const reliability=rows("reliability_exposure"), reconciliation=rows("source_metric_reconciliation");
 const passed=reconciliation.length>0&&reconciliation.every(r=>r.passed===true);

 if(loading&&!overview)return <main className="min-h-screen bg-slate-950 p-10 text-slate-300">Loading saved Phase 15 reports…</main>;
 if(!overview)return <main className="min-h-screen bg-slate-950 p-6 text-white"><div className="mx-auto max-w-3xl rounded-xl border border-red-900 p-6"><h1>Phase 15 dashboard unavailable</h1><p className="mt-3 text-sm text-slate-300">{error}</p><button onClick={()=>void load()} className="mt-4 rounded-lg border px-4 py-2">Retry</button></div></main>;

 return <main className="min-h-screen bg-slate-950 px-4 py-7 text-slate-100 sm:px-6"><div className="mx-auto max-w-7xl space-y-6">
  <header className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end"><div><div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-cyan-400"><ShieldCheck size={15}/> Phase 15 · Validation Research</div><h1 className="text-2xl font-semibold sm:text-3xl">Risk, Drawdown & Portfolio Stability</h1><p className="mt-2 max-w-3xl text-sm text-slate-400">Read-only dashboard from saved Part 4.3 reports. No backtest, model fitting, or live market request is performed.</p></div><button onClick={()=>void load()} disabled={loading} className="inline-flex items-center gap-2 self-start rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm"><RefreshCw size={15} className={loading?"animate-spin":""}/>Refresh reports</button></header>
  {error&&<p className="rounded-xl border border-amber-900 bg-amber-950/20 p-3 text-sm text-amber-200">{error}</p>}
  <section className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/50 p-4 sm:flex-row">
   <label className="flex-1 text-xs text-slate-400">Portfolio<select value={portfolio} onChange={e=>setPortfolio(e.target.value)} className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white">{overview.portfolios.map(p=><option key={p} value={p}>{names[p]??p}</option>)}</select></label>
   <label className="flex-1 text-xs text-slate-400">Slippage (bps per side)<select value={slippage} onChange={e=>setSlippage(Number(e.target.value))} className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white">{overview.slippage_scenarios_bps_per_side.map(v=><option key={v} value={v}>{v} bps / side</option>)}</select></label>
   <div className="self-end pb-2 text-xs text-slate-500">Run <span className="font-mono text-slate-300">{overview.run_id}</span></div>
  </section>
  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
   <Metric label="Net total return" value={`${f(n(metric,"net_total_return_pct"))}%`} note="After modeled costs" Icon={TrendingDown}/>
   <Metric label="Maximum drawdown" value={`${f(n(metric,"max_drawdown_pct"))}%`} note="Peak-to-trough decline" Icon={Activity}/>
   <Metric label="Annualized volatility" value={`${f(n(metric,"annualized_volatility_pct"))}%`} note="Simulated daily returns" Icon={Waves}/>
   <Metric label="Sharpe · zero risk-free" value={f(n(metric,"sharpe_rf_zero"))} note="Descriptive statistic" Icon={BarChart3}/>
  </div>
  <div className="flex flex-col gap-1 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3 text-xs sm:flex-row sm:items-center sm:justify-between">
   <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
    <div className="flex flex-wrap items-end gap-3">
     <div className="pb-2 text-xs font-semibold text-slate-300">Chart date range</div>
     <label className="text-xs text-slate-400">From<input type="date" value={chartFrom} min={firstSession} max={lastSession} disabled={!daily.length} onChange={e=>{const value=e.currentTarget.value;if(!value||value<firstSession||value>lastSession)return;setDateRange(current=>{const currentTo=current.to||lastSession;return {from:value,to:value>currentTo?value:currentTo};});}} className="mt-1 block rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white [color-scheme:dark]"/></label>
     <label className="text-xs text-slate-400">To<input type="date" value={chartTo} min={firstSession} max={lastSession} disabled={!daily.length} onChange={e=>{const value=e.currentTarget.value;if(!value||value<firstSession||value>lastSession)return;setDateRange(current=>{const currentFrom=current.from||firstSession;return {from:value<currentFrom?value:currentFrom,to:value};});}} className="mt-1 block rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white [color-scheme:dark]"/></label>
     <button type="button" onClick={()=>setDateRange({from:firstSession,to:lastSession})} disabled={!daily.length} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-300 hover:border-cyan-700 disabled:opacity-50">Reset range</button>
    </div>
    <div className="text-xs leading-5 text-slate-500 lg:text-right">
     <div>Showing <b className="text-slate-300">{chartDaily.length}</b> of {daily.length} sessions</div>
     <div>Selected range: <b className="text-slate-300">{formatSessionDate(chartFrom)} — {formatSessionDate(chartTo)}</b></div>
     <div>Available research: {formatSessionDate(firstSession)} — {formatSessionDate(lastSession)} · Final holdout excluded</div>
     <div>Drawdown retains the full-period high-water mark.</div>
    </div>
   </div>
   <div className="text-slate-400">{daily.length.toLocaleString("en-IN")} sessions · Validation data only · Final holdout excluded</div>
  </div>
  <div className="grid gap-5 xl:grid-cols-2">
   <Panel title="Net liquidation equity" note="Saved daily mark-to-market series">{chartDaily.length?<div className="h-72"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartDaily}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3"/><XAxis dataKey="session_date" interval="preserveStartEnd" minTickGap={20} tickFormatter={formatSessionDate} tick={{fill:"#94a3b8",fontSize:10}}/><YAxis width={65} tick={{fill:"#94a3b8",fontSize:10}} tickFormatter={v=>`₹${(v/100000).toFixed(1)}L`}/><Tooltip contentStyle={{background:"#020617",border:"1px solid #334155"}} labelFormatter={label=>formatSessionDate(String(label))} formatter={v=>[money(Number(v)),"Net liquidation equity"]}/><Line dataKey="net_liquidation_equity_inr" stroke="#38bdf8" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div>:<Empty/>}</Panel>
   <Panel title="Drawdown profile" note="Negative values indicate decline from high-water mark">{chartDaily.length?<div className="h-72"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartDaily}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3"/><XAxis dataKey="session_date" interval="preserveStartEnd" minTickGap={20} tickFormatter={formatSessionDate} tick={{fill:"#94a3b8",fontSize:10}}/><YAxis tick={{fill:"#94a3b8",fontSize:10}} tickFormatter={v=>`${v}%`}/><Tooltip contentStyle={{background:"#020617",border:"1px solid #334155"}} labelFormatter={label=>formatSessionDate(String(label))} formatter={v=>[`${f(Number(v))}%`,"Drawdown"]}/><Area dataKey="drawdown_pct" stroke="#f87171" fill="#7f1d1d" fillOpacity={0.3}/></AreaChart></ResponsiveContainer></div>:<Empty/>}</Panel>
  </div>
  <div className="grid gap-5 xl:grid-cols-2">
   <Panel title="Drawdown episodes" note="Recovery state from saved report"><div className="mb-3 grid grid-cols-3 gap-2 text-xs"><div className="rounded-lg bg-slate-950 p-3"><span className="text-slate-500">Episodes</span><b className="mt-1 block text-lg">{episodes.length}</b></div><div className="rounded-lg bg-slate-950 p-3"><span className="text-slate-500">Recovered</span><b className="mt-1 block text-lg text-emerald-300">{episodes.filter(r=>r.recovered===true).length}</b></div><div className="rounded-lg bg-slate-950 p-3"><span className="text-slate-500">Unrecovered</span><b className="mt-1 block text-lg text-amber-300">{episodes.filter(r=>r.recovered===false).length}</b></div></div><div className="max-h-64 overflow-auto"><table className="w-full text-left text-xs"><thead className="text-slate-400"><tr><th className="py-2">Peak</th><th>Trough</th><th>Drawdown</th><th>Recovery</th></tr></thead><tbody>{episodes.map((r,i)=><tr key={`${r.peak_date}-${i}`} className="border-t border-slate-800"><td className="py-2">{String(r.peak_date??"—")}</td><td>{String(r.trough_date??"—")}</td><td className="text-rose-300">{f(n(r,"maximum_drawdown_pct"))}%</td><td>{r.recovered===true?"Recovered":"Unrecovered"}</td></tr>)}</tbody></table>{!episodes.length&&<Empty/>}</div></Panel>
   <Panel title="Fold-by-fold calendar risk" note="Descriptive folds; not independent samples">{folds.length?<div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={folds}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3"/><XAxis dataKey="fold_number" tick={{fill:"#94a3b8",fontSize:10}}/><YAxis tick={{fill:"#94a3b8",fontSize:10}} tickFormatter={v=>`${v}%`}/><Tooltip contentStyle={{background:"#020617",border:"1px solid #334155"}} formatter={v=>[`${f(Number(v))}%`,"Net calendar return"]}/><Bar dataKey="net_calendar_period_return_pct_compounded" fill="#a78bfa"/></BarChart></ResponsiveContainer></div>:<Empty/>}</Panel>
  </div>
  <div className="grid gap-5 xl:grid-cols-2">
   <Panel title="Market-regime exposure" note="Daily exposure and risk by regime"><div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs"><thead className="text-slate-400"><tr><th className="py-2">Type</th><th>Regime</th><th>Days</th><th>Avg exposure</th><th>CVaR 95</th></tr></thead><tbody>{regimes.map((r,i)=><tr key={`${r.regime_type}-${r.regime}-${i}`} className="border-t border-slate-800"><td className="py-2 text-slate-400">{String(r.regime_type??"—")}</td><td>{String(r.regime??"—")}</td><td>{f(n(r,"daily_observations"),0)}</td><td>{f(n(r,"average_exposure_pct"))}%</td><td className="text-rose-300">{f(n(r,"daily_cvar_95_loss_pct"))}%</td></tr>)}</tbody></table>{!regimes.length&&<Empty/>}</div></Panel>
   <Panel title="Prediction reliability exposure" note="Retrospective ranking proxy—not model confidence or live sizing rule"><div className="space-y-3">{reliability.map((r,i)=><div key={`${r.model}-${r.prediction_reliability_state}-${i}`} className="rounded-xl border border-slate-800 bg-slate-950 p-3"><b className="text-sm">{String(r.model)} · {String(r.prediction_reliability_state)}</b><div className="mt-2 grid grid-cols-3 gap-2 text-xs"><div><span className="text-slate-500">Exposure</span><p>{f(n(r,"average_exposure_pct"))}%</p></div><div><span className="text-slate-500">Mean IC</span><p>{f(n(r,"mean_trailing_matured_ic"),4)}</p></div><div><span className="text-slate-500">CVaR 95</span><p>{f(n(r,"daily_cvar_95_loss_pct"))}%</p></div></div>{typeof r.interpretation==="string"&&<p className="mt-2 text-xs text-slate-500">{r.interpretation}</p>}</div>)}{!reliability.length&&<Empty/>}</div></Panel>
  </div>
  <Panel title="Largest contributors to negative stock-leg losses" note="Top ten by share of all negative trade loss · names resolved from the saved instrument master"><div className="overflow-auto"><table className="w-full min-w-[900px] text-left text-xs"><thead className="text-slate-400"><tr><th className="py-2">Symbol</th><th>Company name</th><th>Instrument key</th><th>Legs</th><th>Net P&amp;L</th><th>Winning legs</th><th>Share of negative loss</th></tr></thead><tbody>{concentration.map((r,i)=><tr key={`${r.instrument_key}-${i}`} className="border-t border-slate-800"><td className="py-2 font-semibold text-cyan-300">{instrumentMap[String(r.instrument_key)]?.symbol ?? "—"}</td><td>{instrumentMap[String(r.instrument_key)]?.name ?? "—"}</td><td className="font-mono text-slate-500">{String(r.instrument_key ?? "—")}</td><td>{f(n(r,"completed_stock_legs"),0)}</td><td className={(n(r,"net_pnl_inr")??0)<0?"text-rose-300":"text-emerald-300"}>{money(n(r,"net_pnl_inr"))}</td><td>{f(n(r,"winning_legs_pct"))}%</td><td>{f(n(r,"share_of_all_negative_trade_loss_pct"))}%</td></tr>)}</tbody></table>{!concentration.length&&<Empty/>}</div></Panel>
  <Panel title="Source reconciliation & research integrity"><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><div className="rounded-xl bg-slate-950 p-3"><div className="flex items-center gap-2 text-xs text-slate-400">{passed?<CheckCircle2 size={15} className="text-emerald-400"/>:<AlertTriangle size={15} className="text-amber-400"/>}Metric reconciliation</div><b className={`mt-2 block text-lg ${passed?"text-emerald-300":"text-amber-300"}`}>{passed?"Passed":"Not verified"}</b></div><div className="rounded-xl bg-slate-950 p-3"><span className="text-xs text-slate-500">Protected artifacts modified</span><b className="mt-2 block text-lg">{String(overview.integrity.protected_artifacts_modified??"—")}</b></div><div className="rounded-xl bg-slate-950 p-3"><span className="text-xs text-slate-500">Source reports modified</span><b className="mt-2 block text-lg">{String(overview.integrity.source_reports_modified??"—")}</b></div><div className="rounded-xl bg-slate-950 p-3"><span className="text-xs text-slate-500">Part 4.3 final holdout used</span><b className="mt-2 block text-lg">{String(overview.holdout.final_holdout_used??"—")}</b></div></div><div className="mt-4 grid gap-2 text-xs sm:grid-cols-2">{Object.entries(overview.source_runs).map(([k,v])=><div key={k} className="flex justify-between gap-3 border-b border-slate-800 py-2"><span className="text-slate-500">{k.replaceAll("_"," ")}</span><span className="font-mono text-slate-300">{v}</span></div>)}</div></Panel>
  <Panel title="Methodology and interpretation limits"><ul className="list-disc space-y-2 pl-5 text-xs leading-5 text-slate-400">{overview.limitations.map((x,i)=><li key={i}>{x}</li>)}</ul><div className="mt-4 rounded-xl border border-amber-900/50 bg-amber-950/20 p-4 text-xs leading-5 text-amber-100/90">Research-only evidence. Historical simulated results do not establish profitability, execution feasibility, or readiness for live trading. Reliability state is not a live risk gate. Part 4.3 states its final holdout was not used; this dashboard is not an independent final-holdout evaluation of the complete benchmark pipeline.</div></Panel>
  <footer className="flex justify-between border-t border-slate-800 pt-4 text-[11px] text-slate-600"><span>StockAI · Phase 15 · Saved reports only</span><span>Run {overview.run_id}</span></footer>
 </div></main>;
}
