import MarketScanner from "../components/MarketScanner";
import MarketRegime from "../components/MarketRegime";
import SectorStrength from "../components/SectorStrength";
import FundamentalAnalysis from "../components/FundamentalAnalysis";
import NewsEventAnalysis from "../components/NewsEventAnalysis";

export default function ScannerPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">

      <main className="mx-auto max-w-7xl px-6 py-8">
        <section className="mb-8">
          <p className="text-sm text-cyan-400">Phase 5 · Market Intelligence</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            Scanner &amp; Market Analysis
          </h1>
          <p className="mt-2 max-w-3xl text-slate-400">
            Analyze the NSE market through the Phase 5 scanner, technical setups,
            market regime, sector rotation, fundamentals, and news events.
          </p>
        </section>

        {/* Each Phase 5 layer is rendered exactly once. */}
        <div className="space-y-6">
          <MarketScanner />
          <MarketRegime />
          <SectorStrength />
          <FundamentalAnalysis />
          <NewsEventAnalysis />
        </div>
      </main>
    </div>
  );
}
