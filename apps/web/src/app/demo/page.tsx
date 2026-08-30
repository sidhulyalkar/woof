import type { Metadata } from 'next';
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  CircleDot,
  Clock3,
  Compass,
  DatabaseZap,
  HeartHandshake,
  LockKeyhole,
  PawPrint,
  ShieldCheck,
  Sparkles,
  UserRoundCheck,
} from 'lucide-react';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Synthetic dogOS walkthrough',
  description:
    'A synthetic walkthrough of how Woof chooses a useful next action, preserves evidence, and keeps shared-care authority explicit.',
};

const evidence = [
  {
    label: 'Owner context',
    value: 'Lower-key day requested',
    detail: 'Canonical preference for this decision, not a durable dog trait.',
    icon: UserRoundCheck,
  },
  {
    label: 'Recent rhythm',
    value: 'Mostly walks this week',
    detail: 'Activity history can support variety without becoming an exercise prescription.',
    icon: Compass,
  },
  {
    label: 'Weather',
    value: 'Not configured',
    detail: 'Missing live context stays missing. Woof does not fill the gap with invented certainty.',
    icon: CircleDot,
  },
] as const;

const outcomeSignals = [
  {
    source: 'Owner response',
    value: 'Felt easy to fit into the day',
    authority: 'Owner-reported outcome',
  },
  {
    source: 'Dog response',
    value: 'Loose body, engaged sniffing, chose to continue',
    authority: 'Observed response, kept separate from the owner response',
  },
  {
    source: 'Next-time lesson',
    value: 'Short exploratory options remain reasonable on lower-key days',
    authority: 'Bounded recommendation evidence, not a universal preference label',
  },
] as const;

const caregiverBoundaries = [
  ['Today context', 'Granted'],
  ['Context-only observation', 'Granted'],
  ['Story / household history', 'Not granted'],
  ['Medical authority', 'Not granted'],
  ['Profile correction', 'Not granted'],
  ['Bond XP / reward authority', 'Not granted'],
] as const;

const layers = [
  {
    title: 'Context + provenance',
    copy: 'Owner reports, caregiver notes, activities, connectors, and model outputs keep their source and authority instead of collapsing into one pet score.',
    icon: DatabaseZap,
  },
  {
    title: 'Policy + next action',
    copy: 'Today turns the context Woof can actually trust into one bounded suggestion, with alternatives and safe-stop semantics rather than a compulsory plan.',
    icon: Sparkles,
  },
  {
    title: 'Memory + correction',
    copy: 'Outcomes become longitudinal evidence and Story while owner corrections remain stronger than inference. Missing evidence stays missing.',
    icon: BookOpen,
  },
  {
    title: 'People + authority',
    copy: 'Owners, household members, and temporary caregivers can participate differently. Presentation never creates pet access.',
    icon: LockKeyhole,
  },
] as const;

export default function DemoPage() {
  return (
    <main id="main-content" className="min-h-screen px-4 pb-16 pt-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="glass-strong rounded-[2rem] p-6 sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <Link
              href="/login"
              className="brand-mark inline-flex items-center gap-2 rounded-full px-4 py-2"
            >
              <PawPrint className="h-5 w-5" aria-hidden="true" />
              <span className="font-semibold">Woof</span>
            </Link>
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-sm text-emerald-200">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              Synthetic data only
            </div>
          </div>

          <div className="mt-10 grid gap-8 lg:grid-cols-[1.3fr_0.7fr] lg:items-end">
            <div>
              <p className="eyebrow">A synthetic dogOS walkthrough</p>
              <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-tight sm:text-6xl">
                One useful next step, with memory and authority underneath it.
              </h1>
              <p className="mt-5 max-w-3xl text-base leading-7 text-muted-foreground sm:text-lg">
                Woof is built around a daily relationship loop: notice what matters, choose something
                reasonable to do together, read the response, and preserve only the evidence that
                should make a later decision easier.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Link
                  href="#today"
                  className="inline-flex min-h-11 items-center gap-2 rounded-full bg-primary px-5 py-2.5 font-medium text-primary-foreground"
                >
                  Walk through Today <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
                <Link
                  href="/onboarding/companion"
                  className="inline-flex min-h-11 items-center gap-2 rounded-full border border-border bg-background/50 px-5 py-2.5 font-medium"
                >
                  Choose how to start
                </Link>
              </div>
            </div>

            <aside data-demo-card className="surface-soft rounded-3xl p-5">
              <p className="eyebrow">What this demo is not</p>
              <ul className="mt-4 space-y-3 text-sm leading-6 text-muted-foreground">
                <li className="flex gap-2">
                  <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />
                  No diagnosis or exercise prescription
                </li>
                <li className="flex gap-2">
                  <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />
                  No live location, private messages, or real health records
                </li>
                <li className="flex gap-2">
                  <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />
                  No claim that a model score is product or release authority
                </li>
              </ul>
            </aside>
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-4" aria-label="dogOS layers">
          {layers.map(({ title, copy, icon: Icon }) => (
            <article key={title} data-demo-card className="glass rounded-3xl p-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <Icon className="h-5 w-5" aria-hidden="true" />
              </div>
              <h2 className="mt-4 font-semibold">{title}</h2>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">{copy}</p>
            </article>
          ))}
        </section>

        <section id="today" className="glass-strong rounded-[2rem] p-6 sm:p-8">
          <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div>
              <p className="eyebrow">1 · Notice</p>
              <h2 className="mt-2 text-3xl font-semibold">What does Woof actually know today?</h2>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
                This fictional example uses only available context. Each input keeps its source and
                meaning instead of being flattened into an all-purpose dog score.
              </p>

              <div className="mt-6 space-y-3">
                {evidence.map(({ label, value, detail, icon: Icon }) => (
                  <article key={label} data-demo-card className="surface-soft rounded-2xl p-4">
                    <div className="flex gap-3">
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                        <Icon className="h-4 w-4" aria-hidden="true" />
                      </span>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          {label}
                        </p>
                        <p className="mt-1 font-medium">{value}</p>
                        <p className="mt-1 text-xs leading-5 text-muted-foreground">{detail}</p>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <article data-demo-card className="rounded-[1.75rem] border border-primary/25 bg-primary/5 p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="eyebrow">2 · Choose</p>
                  <h2 className="mt-2 text-3xl font-semibold">A short sniff walk</h2>
                </div>
                <span className="rounded-full border border-border/70 bg-background/60 px-3 py-1 text-xs font-medium">
                  Medium confidence
                </span>
              </div>

              <p className="mt-5 text-base leading-7 text-muted-foreground">
                Try an easy exploratory walk with room to sniff. Stop early if Nova disengages or the
                outing stops feeling useful.
              </p>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl bg-background/45 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Why this surfaced
                  </p>
                  <p className="mt-2 text-sm leading-6">
                    The owner asked for a lower-key day and recent activity has been walk-heavy, so the
                    suggestion stays familiar while emphasizing exploration rather than volume.
                  </p>
                </div>
                <div className="rounded-2xl bg-background/45 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Boundaries
                  </p>
                  <p className="mt-2 text-sm leading-6">
                    No medical inference. No calorie or distance target. Missing weather is not silently
                    substituted. A safe stop still counts as a useful outcome.
                  </p>
                </div>
              </div>
            </article>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <article data-demo-card className="glass rounded-[2rem] p-6 sm:p-7">
            <p className="eyebrow">3 · Read the response</p>
            <h2 className="mt-2 text-2xl font-semibold">The outcome stays multi-part.</h2>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">
              Human convenience, observed dog response, and recommendation evidence can agree or
              disagree. Woof keeps them separate so a later recommendation can be corrected rather
              than rationalized.
            </p>

            <div className="mt-5 space-y-3">
              {outcomeSignals.map((signal) => (
                <div key={signal.source} className="rounded-2xl border border-border/70 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-medium">{signal.source}</p>
                    <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      preserved
                    </span>
                  </div>
                  <p className="mt-2 text-sm">{signal.value}</p>
                  <p className="mt-2 text-xs leading-5 text-muted-foreground">{signal.authority}</p>
                </div>
              ))}
            </div>
          </article>

          <article data-demo-card className="glass rounded-[2rem] p-6 sm:p-7">
            <p className="eyebrow">4 · Remember</p>
            <h2 className="mt-2 text-2xl font-semibold">Story remembers the relationship, not a score.</h2>
            <div className="mt-5 rounded-2xl border border-border/70 bg-background/35 p-5">
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <BookOpen className="h-5 w-5" aria-hidden="true" />
                </span>
                <div>
                  <p className="font-medium">Saturday · Easy exploration</p>
                  <p className="text-xs text-muted-foreground">Synthetic relationship memory</p>
                </div>
              </div>
              <p className="mt-4 text-sm leading-6 text-muted-foreground">
                Nova stayed engaged on a short exploratory outing. The owner said the format was easy
                to fit into a lower-key day. This can inform a later choice without becoming a permanent
                claim about Nova.
              </p>
            </div>

            <div className="mt-4 flex items-start gap-3 rounded-2xl bg-secondary/8 p-4 text-sm">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-secondary" aria-hidden="true" />
              <p className="leading-6 text-muted-foreground">
                Owner correction remains stronger than inference. Reward mechanics and social reactions
                do not become recommendation labels.
              </p>
            </div>
          </article>
        </section>

        <section className="glass-strong rounded-[2rem] p-6 sm:p-8">
          <div className="grid gap-7 lg:grid-cols-[0.78fr_1.22fr] lg:items-start">
            <div>
              <p className="eyebrow">Shared care without shared ownership</p>
              <h2 className="mt-2 text-3xl font-semibold">Temporary caregiver authority has edges.</h2>
              <p className="mt-3 text-sm leading-6 text-muted-foreground">
                In this fictional handoff, Maya can see the temporary Today context and leave a
                context-only observation until the grant expires. That does not turn Maya into a
                household member or owner.
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-full border border-border bg-background/45 px-3 py-1.5 text-sm">
                <Clock3 className="h-4 w-4 text-primary" aria-hidden="true" />
                Grant expires today at 6:00 PM
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {caregiverBoundaries.map(([label, status]) => {
                const granted = status === 'Granted';
                return (
                  <div key={label} className="surface-soft flex items-center justify-between gap-4 rounded-2xl p-4">
                    <span className="text-sm font-medium">{label}</span>
                    <span
                      className={
                        granted
                          ? 'rounded-full bg-emerald-400/10 px-2.5 py-1 text-xs text-emerald-300'
                          : 'rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground'
                      }
                    >
                      {status}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-6 flex items-start gap-3 rounded-2xl border border-border/70 bg-background/35 p-4 text-sm">
            <HeartHandshake className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
            <p className="leading-6 text-muted-foreground">
              Revocation, expiry, or a relationship block removes present-tense caregiver authority.
              Context-only caregiver observations do not automatically become Bond XP, medical truth,
              owner correction, or canonical recommendation evidence.
            </p>
          </div>
        </section>

        <section className="glass rounded-[2rem] p-6 sm:p-8">
          <p className="eyebrow">What sits above dogOS</p>
          <h2 className="mt-2 max-w-3xl text-2xl font-semibold">
            The applications can change without making every application its own source of truth.
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-muted-foreground">
            Today, Compass, Story, Coach, Community, Health Lens, Discovery, and connectors serve
            different jobs. The shared layer underneath them keeps pet identity, people authority,
            evidence provenance, longitudinal memory, and release boundaries explicit.
          </p>
        </section>

        <footer className="flex flex-col gap-4 rounded-3xl border border-border/70 px-6 py-5 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <p>
            This route is intentionally synthetic. It demonstrates qualified product contracts, not
            real-world efficacy or clinical outcomes.
          </p>
          <Link href="/login" className="font-medium text-foreground underline-offset-4 hover:underline">
            Open Woof <ArrowRight className="ml-1 inline h-4 w-4" aria-hidden="true" />
          </Link>
        </footer>
      </div>
    </main>
  );
}
