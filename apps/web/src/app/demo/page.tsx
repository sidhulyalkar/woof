import Link from 'next/link';
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  HeartHandshake,
  LockKeyhole,
  MessageCircle,
  PawPrint,
  Repeat2,
  ShieldCheck,
  Sparkles,
  Users,
} from 'lucide-react';

const matches = [
  {
    pet: 'Milo',
    owner: '@jordan_demo',
    score: 88,
    confidence: 82,
    source: 'behavior-outcome-baseline-v2',
    reason: 'Similar play intensity, sociability and successful low-pressure introductions.',
  },
  {
    pet: 'Juniper',
    owner: '@riley_demo',
    score: 79,
    confidence: 68,
    source: 'learned-shadow-v1',
    reason: 'Good behavior fit; learned score is shown in shadow mode while calibration evidence grows.',
  },
  {
    pet: 'Pepper',
    owner: '@sam_demo',
    score: 71,
    confidence: 59,
    source: 'behavior-outcome-baseline-v2',
    reason: 'Promising social fit with less evidence, so Woof lowers confidence rather than inventing certainty.',
  },
];

const funnel = [
  { label: 'Discovery', value: 100, icon: Users },
  { label: 'Conversation', value: 61, icon: MessageCircle },
  { label: 'Meetup', value: 29, icon: HeartHandshake },
  { label: 'Repeat meetup', value: 17, icon: Repeat2 },
];

export default function DemoPage() {
  return (
    <main className="min-h-screen px-4 pb-16 pt-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="glass-strong rounded-[2rem] p-6 sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Link href="/login" className="brand-mark inline-flex items-center gap-2 rounded-full px-4 py-2">
              <PawPrint className="h-5 w-5" aria-hidden="true" />
              <span className="font-semibold">Woof</span>
            </Link>
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-sm text-emerald-200">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              Synthetic data only
            </div>
          </div>

          <div className="mt-10 grid gap-8 lg:grid-cols-[1.35fr_0.65fr] lg:items-end">
            <div>
              <p className="eyebrow">Woof 0.3 beta</p>
              <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight sm:text-6xl">
                Synthetic beta demo
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
                Explore the relationship-learning loop with fictional pets and owners. This public surface
                uses no live location, no private messages, no health records and no real meetup history.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Link
                  href="#compatibility"
                  className="inline-flex min-h-11 items-center gap-2 rounded-full bg-primary px-5 py-2.5 font-medium text-primary-foreground"
                >
                  Explore compatibility <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
                <Link
                  href="/onboarding"
                  className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-background/50 px-5 py-2.5 font-medium"
                >
                  See onboarding
                </Link>
              </div>
            </div>

            <div data-demo-card className="surface-soft rounded-3xl p-5">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-primary/10 p-3 text-primary">
                  <LockKeyhole className="h-5 w-5" aria-hidden="true" />
                </div>
                <div>
                  <p className="font-medium">Privacy mode</p>
                  <p className="text-sm text-muted-foreground">No live location or external API calls</p>
                </div>
              </div>
              <ul className="mt-5 space-y-3 text-sm text-muted-foreground">
                <li className="flex gap-2"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />Coarse fictional places only</li>
                <li className="flex gap-2"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />Model source shown with every score</li>
                <li className="flex gap-2"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />Outcome metrics focus on real-world value</li>
              </ul>
            </div>
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-3" aria-label="Synthetic pet context">
          <article data-demo-card className="glass rounded-3xl p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-primary/10 p-3 text-primary"><PawPrint className="h-5 w-5" aria-hidden="true" /></div>
              <div><p className="text-sm text-muted-foreground">Demo pet</p><h2 className="text-xl font-semibold">Nova</h2></div>
            </div>
            <p className="mt-4 text-sm leading-6 text-muted-foreground">3-year-old fictional dog. Playful, moderately energetic, socially curious and initially cautious.</p>
          </article>

          <article data-demo-card className="glass rounded-3xl p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-cyan-400/10 p-3 text-cyan-300"><Activity className="h-5 w-5" aria-hidden="true" /></div>
              <div><p className="text-sm text-muted-foreground">Recent pattern</p><h2 className="text-xl font-semibold">Variety is down</h2></div>
            </div>
            <p className="mt-4 text-sm leading-6 text-muted-foreground">Five walks this week, but no play or training sessions. Woof suggests variety without prescribing medical exercise targets.</p>
          </article>

          <article data-demo-card className="glass rounded-3xl p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-violet-400/10 p-3 text-violet-300"><Sparkles className="h-5 w-5" aria-hidden="true" /></div>
              <div><p className="text-sm text-muted-foreground">Next best action</p><h2 className="text-xl font-semibold">Low-pressure play</h2></div>
            </div>
            <p className="mt-4 text-sm leading-6 text-muted-foreground">Try a short enrichment or training session today. Confidence: 76%, based on recent routine and stated preferences.</p>
          </article>
        </section>

        <section id="compatibility" className="glass-strong rounded-[2rem] p-6 sm:p-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="eyebrow">Explainable compatibility</p>
              <h2 className="mt-2 text-3xl font-semibold">Who might Nova enjoy meeting?</h2>
            </div>
            <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
              <BrainCircuit className="h-4 w-4" aria-hidden="true" />
              Baseline remains the safety fallback
            </div>
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {matches.map((match) => (
              <article key={match.pet} data-demo-card className="surface-soft rounded-3xl p-5">
                <div className="flex items-start justify-between gap-4">
                  <div><h3 className="text-xl font-semibold">{match.pet}</h3><p className="text-sm text-muted-foreground">{match.owner}</p></div>
                  <div className="rounded-2xl bg-emerald-400/10 px-3 py-2 text-right"><p className="text-lg font-semibold text-emerald-300">{match.score}%</p><p className="text-[11px] uppercase tracking-wide text-muted-foreground">compatibility</p></div>
                </div>
                <p className="mt-4 text-sm leading-6 text-muted-foreground">{match.reason}</p>
                <dl className="mt-5 grid gap-2 text-xs">
                  <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Confidence</dt><dd>{match.confidence}%</dd></div>
                  <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Score provenance</dt><dd className="max-w-[65%] truncate text-right" title={match.source}>{match.source}</dd></div>
                </dl>
              </article>
            ))}
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
          <article data-demo-card className="glass rounded-[2rem] p-6">
            <p className="eyebrow">Learning loop</p>
            <h2 className="mt-2 text-2xl font-semibold">Outcome beats attention</h2>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">A like is not evidence that a recommendation helped. The beta measures whether discovery leads to conversation, a safe meetup, and eventually a voluntary repeat interaction.</p>
            <div className="mt-6 space-y-4">
              {funnel.map(({ label, value, icon: Icon }) => (
                <div key={label}>
                  <div className="mb-2 flex items-center justify-between text-sm"><span className="flex items-center gap-2"><Icon className="h-4 w-4 text-primary" aria-hidden="true" />{label}</span><span className="font-medium">{value}%</span></div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary" style={{ width: `${value}%` }} /></div>
                </div>
              ))}
            </div>
          </article>

          <article data-demo-card className="glass rounded-[2rem] p-6">
            <p className="eyebrow">What the beta protects</p>
            <h2 className="mt-2 text-2xl font-semibold">IRL coordination with boundaries</h2>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {[
                ['Location minimization', 'Precise coordinates are never part of this demo and are not required for compatibility ranking.'],
                ['Mutual consent', 'Meetup context is revealed progressively only after both people opt into coordination.'],
                ['Block-aware ranking', 'Blocked and avoided relationships are removed before recommendations are produced.'],
                ['Uncertainty', 'Sparse evidence lowers confidence instead of being hidden behind an authoritative-looking score.'],
              ].map(([title, copy]) => (
                <div key={title} className="rounded-2xl border border-border/70 bg-background/35 p-4"><h3 className="font-medium">{title}</h3><p className="mt-2 text-sm leading-6 text-muted-foreground">{copy}</p></div>
              ))}
            </div>
          </article>
        </section>

        <footer className="flex flex-col gap-4 rounded-3xl border border-border/70 px-6 py-5 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <p>This route is intentionally synthetic. It demonstrates product contracts, not production outcomes.</p>
          <Link href="/login" className="font-medium text-foreground underline-offset-4 hover:underline">Open Woof <ArrowRight className="ml-1 inline h-4 w-4" aria-hidden="true" /></Link>
        </footer>
      </div>
    </main>
  );
}
