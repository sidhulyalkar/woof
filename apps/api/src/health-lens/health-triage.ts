export type HealthTriageLevel =
  | 'emergency_now'
  | 'vet_today'
  | 'vet_soon'
  | 'monitor'
  | 'better_image'
  | 'insufficient_information';

export type DeterministicTriage = {
  level: HealthTriageLevel;
  matchedSignals: string[];
  summary: string;
  action: string;
} | null;

const EMERGENCY_PATTERNS: Array<{ id: string; patterns: RegExp[] }> = [
  {
    id: 'breathing-distress',
    patterns: [
      /struggl(?:e|ing) to breathe/i,
      /difficulty breathing/i,
      /labou?red breathing/i,
      /gasping/i,
      /blue (?:gum|gums|tongue)/i,
      /purple (?:gum|gums|tongue)/i,
      /open[- ]mouth breathing/i,
      /major change in breathing/i,
    ],
  },
  {
    id: 'collapse-consciousness',
    patterns: [/collapsed?/i, /unconscious/i, /unresponsive/i, /cannot stand/i, /can't stand/i],
  },
  {
    id: 'seizure',
    patterns: [/seiz(?:e|ing|ure).*(?:now|ongoing|won't stop|more than)/i, /multiple seizures/i],
  },
  {
    id: 'severe-bleeding',
    patterns: [
      /uncontrolled bleeding/i,
      /won't stop bleeding/i,
      /severe bleeding/i,
      /spurting blood/i,
    ],
  },
  {
    id: 'toxin',
    patterns: [
      /poison(?:ed|ing)?/i,
      /toxin/i,
      /antifreeze/i,
      /xylitol/i,
      /rodenticide/i,
      /rat poison/i,
      /human medication/i,
    ],
  },
  {
    id: 'urinary-obstruction',
    patterns: [
      /cannot urinate/i,
      /can't urinate/i,
      /unable to urinate/i,
      /straining to (?:pee|urinate).*(?:nothing|no urine)/i,
    ],
  },
  {
    id: 'gdv-bloat',
    patterns: [
      /(?:swollen|distended|bloated|hard) (?:belly|abdomen).*(?:retch|dry heav|trying to vomit)/i,
      /(?:belly|abdomen).*(?:swollen|distended|bloated|hard).*(?:retch|dry heav|trying to vomit)/i,
      /(?:retch|dry heav|trying to vomit).*(?:swollen|distended|bloated|hard) (?:belly|abdomen)/i,
      /(?:retch|dry heav|trying to vomit).*(?:belly|abdomen).*(?:swollen|distended|bloated|hard)/i,
    ],
  },
  {
    id: 'heatstroke',
    patterns: [/heat ?stroke/i, /overheated.*(?:collapse|unresponsive|vomit)/i],
  },
  {
    id: 'major-trauma',
    patterns: [/hit by (?:a )?car/i, /penetrating wound/i, /open fracture/i, /bone exposed/i],
  },
  {
    id: 'severe-allergic-reaction',
    patterns: [
      /facial swelling.*(?:breath|collapse|weak)/i,
      /hives.*(?:breath|collapse|weak)/i,
      /anaphyla/i,
    ],
  },
];

export function screenEmergencyText(text: string): DeterministicTriage {
  const normalized = text.trim();
  if (!normalized) return null;

  const matchedSignals = EMERGENCY_PATTERNS.filter((signal) =>
    signal.patterns.some((pattern) => pattern.test(normalized))
  ).map((signal) => signal.id);

  if (matchedSignals.length === 0) return null;

  return {
    level: 'emergency_now',
    matchedSignals,
    summary:
      'What you described includes a veterinary emergency warning sign that should not wait for photo or chat analysis.',
    action:
      'Contact an emergency veterinarian now and follow their transport instructions. If possible, call ahead while arranging safe transport.',
  };
}

export function buildTriageText(input: {
  concern: string;
  appetite?: string;
  energy?: string;
  breathing?: string;
  bathroom?: string;
}) {
  return [
    input.concern,
    input.breathing === 'major-change' ? 'major change in breathing' : '',
    input.energy === 'major-change' ? 'major change in energy or responsiveness' : '',
    input.bathroom === 'major-change' ? 'major change in urination or defecation' : '',
    input.appetite === 'major-change' ? 'major change in eating or drinking' : '',
  ]
    .filter(Boolean)
    .join('. ');
}
