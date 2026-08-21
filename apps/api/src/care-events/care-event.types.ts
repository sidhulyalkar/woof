export const WELLBEING_PATHWAYS = [
  'MOVE',
  'EXPLORE',
  'ENRICH',
  'LEARN',
  'CONNECT',
  'CARE',
  'RECOVER',
  'BOND',
] as const;

export type WellbeingPathway = (typeof WELLBEING_PATHWAYS)[number];

export const QUEST_EVENT_TYPES = {
  MOVE: 'QUEST_MOVE',
  EXPLORE: 'QUEST_EXPLORE',
  ENRICH: 'QUEST_ENRICH',
  LEARN: 'QUEST_LEARN',
  CONNECT: 'QUEST_CONNECT',
  CARE: 'QUEST_CARE',
  RECOVER: 'QUEST_RECOVER',
  BOND: 'QUEST_BOND',
} as const satisfies Record<WellbeingPathway, string>;

export type EvidenceType =
  | 'SELF_REPORT'
  | 'ACTIVITY'
  | 'COACH'
  | 'BEHAVIOR_VISION'
  | 'LOCATION'
  | 'MEDIA'
  | 'CLINIC';

export type CareEventInput = {
  userId: string;
  petId?: string | null;
  eventType: string;
  pathway: WellbeingPathway;
  occurredAt?: Date;
  source: string;
  evidenceType?: EvidenceType;
  evidenceConfidence?: number;
  context?: Record<string, unknown>;
  outcome?: Record<string, unknown>;
  dedupeKey: string;
  visibility?: 'PRIVATE' | 'HOUSEHOLD' | 'FRIENDS';
  safetyEligible?: boolean;
};

export type RewardReceipt = {
  careEventId: string;
  ledgerId: string | null;
  bondXp: number;
  pathway: WellbeingPathway;
  policyVersion: string;
  explanation: string;
  duplicate: boolean;
};

export type PathwayProgress = {
  pathway: WellbeingPathway;
  label: string;
  recentDays: number;
  coverage: number;
  xp: number;
  lastEventAt: string | null;
};

export type CareSummary = {
  bondXp: number;
  rhythm: {
    activeWeeks: number;
    windowWeeks: number;
    label: string;
  };
  pathways: PathwayProgress[];
  recentEvents: Array<{
    id: string;
    eventType: string;
    pathway: WellbeingPathway;
    occurredAt: string;
    outcome: Record<string, unknown> | null;
    bondXp: number;
  }>;
};
