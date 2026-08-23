export type ConciergeEvidence = {
  source: 'ADVENTURE' | 'CARE_EVENT' | 'AUTOPILOT' | 'CONNECTOR';
  label: string;
  occurredAt?: string;
  referenceId?: string;
};

export type ConciergeAction = {
  label: string;
  href: string;
};

export type ConciergeSuggestion = {
  id: string;
  kind: 'CARE_PREP' | 'CHECK_IN' | 'RECOVERY_PACE' | 'CONNECTION_ATTENTION';
  priority: 'ATTENTION' | 'GENTLE' | 'INFO';
  title: string;
  body: string;
  reason: string;
  evidence: ConciergeEvidence[];
  action?: ConciergeAction;
  suggestionOnly: true;
};

export type ConciergeContext = {
  weather: {
    status: 'NOT_CONFIGURED';
    live: false;
    detail: string;
  };
  pace: {
    mode: 'NORMAL' | 'GENTLE';
    reason: string;
    evidence: ConciergeEvidence[];
  };
};
