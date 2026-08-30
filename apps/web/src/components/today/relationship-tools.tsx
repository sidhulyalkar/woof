import { BellRing, Brain } from 'lucide-react';
import Link from 'next/link';

const tools = [
  {
    href: '/coach',
    icon: Brain,
    label: 'Practice with Coach',
    description: 'Work on one skill, notice comfort, and make the next repetition easier.',
  },
  {
    href: '/autopilot',
    icon: BellRing,
    label: 'Reminders & check-ins',
    description: 'Keep care reminders and conservative signals together. You stay in control.',
  },
] as const;

export function RelationshipTools() {
  return (
    <section className="mt-6" aria-labelledby="relationship-tools-heading" data-today-tools>
      <div>
        <p className="eyebrow">When you need a deeper tool</p>
        <h2 id="relationship-tools-heading" className="mt-1 text-lg font-bold tracking-tight">
          Practice or keep care on track
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          These tools support Today. They are not extra assignments and neither one acts on its own.
        </p>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {tools.map((tool) => {
          const Icon = tool.icon;
          return (
            <Link
              key={tool.href}
              href={tool.href}
              className="surface-soft group rounded-2xl p-4 transition hover:border-primary/30 focus-visible:border-primary/40"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Icon className="h-4 w-4" aria-hidden="true" />
              </span>
              <span className="mt-3 block text-sm font-semibold group-hover:text-primary">
                {tool.label}
              </span>
              <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
                {tool.description}
              </span>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
