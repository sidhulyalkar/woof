import { Injectable } from '@nestjs/common';
import { AdventureService } from '../adventure/adventure.service';
import { AutopilotService } from '../autopilot/autopilot.service';
import { CareEventsService } from '../care-events/care-events.service';
import { ConnectorsService } from '../connectors/connectors.service';
import type { ConciergeEvidence, ConciergeSuggestion } from './concierge.types';

const HOUR_MS = 60 * 60 * 1000;
const RECENT_FEEDBACK_MS = 72 * HOUR_MS;
const CARE_LOOKAHEAD_MS = 72 * HOUR_MS;

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readableDueAt(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'soon';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'UTC',
  }).format(date);
}

@Injectable()
export class ConciergeService {
  constructor(
    private readonly adventure: AdventureService,
    private readonly careEvents: CareEventsService,
    private readonly autopilot: AutopilotService,
    private readonly connectors: ConnectorsService
  ) {}

  async getToday(userId: string, requestedPetId?: string) {
    const [adventure, autopilot, connectors] = await Promise.all([
      this.adventure.getDashboard(userId, requestedPetId),
      this.autopilot.getDashboard(userId),
      this.connectors.getDashboard(userId),
    ]);
    const care = await this.careEvents.getSummary(userId, adventure.pet.id);
    const now = new Date();
    const pace = this.buildPace(care.recentEvents, now);
    const suggestions = this.buildSuggestions({
      now,
      petId: adventure.pet.id,
      petName: adventure.pet.name,
      pace,
      reminders: autopilot.reminders,
      signals: autopilot.signals,
      connectorProviders: connectors.providers,
    });
    const topQuest = adventure.quests[0] ?? null;
    const attentionCount = suggestions.filter((item) => item.priority === 'ATTENTION').length;

    return {
      generatedAt: now.toISOString(),
      pet: adventure.pet,
      briefing: {
        title: `${adventure.pet.name}'s day at a glance`,
        summary:
          attentionCount > 0
            ? `${attentionCount} item${attentionCount === 1 ? '' : 's'} may deserve your attention. Everything here is a suggestion, not an automatic action.`
            : pace.mode === 'GENTLE'
              ? 'Recent explicit feedback points toward a lighter pace today. You still choose what fits.'
              : 'No urgent care context is surfaced right now. Use the quest deck as options, not obligations.',
        topQuest: topQuest
          ? {
              title: topQuest.title,
              reason: topQuest.why,
              action: { label: topQuest.actionLabel, href: topQuest.href },
              evidence: [
                {
                  source: 'ADVENTURE' as const,
                  label: `Adventure ranked this ${topQuest.primaryPathway.toLowerCase()} option highest for the current deck.`,
                },
              ],
            }
          : null,
      },
      context: {
        weather: {
          status: 'NOT_CONFIGURED' as const,
          live: false as const,
          detail:
            'No verified live weather provider is configured, so Concierge does not claim current conditions.',
        },
        pace,
      },
      suggestions,
      connectorSummary: {
        connected: connectors.providers.filter((provider) => provider.availability === 'CONNECTED')
          .length,
        needsReauthorization: connectors.providers.filter(
          (provider) => provider.availability === 'REAUTH_REQUIRED'
        ).length,
      },
      boundaries: {
        suggestionOnly: true as const,
        liveWeatherUsed: false as const,
        diagnosticInferenceAllowed: false as const,
        prescriptionOrDoseCalculationAllowed: false as const,
        persistentStateMutationAllowed: false as const,
        autonomousPurchaseAllowed: false as const,
      },
    };
  }

  private buildPace(
    recentEvents: Array<{
      id: string;
      occurredAt: string;
      outcome: Record<string, unknown> | null;
    }>,
    now: Date
  ) {
    for (const event of recentEvents) {
      const occurredAt = new Date(event.occurredAt);
      if (
        !Number.isFinite(occurredAt.getTime()) ||
        now.getTime() - occurredAt.getTime() > RECENT_FEEDBACK_MS
      ) {
        continue;
      }

      const outcome = objectValue(event.outcome);
      if (!outcome) continue;
      const ownerExperience = outcome.ownerExperience;
      const dogExperience = outcome.dogExperience;
      const safeOptOut = outcome.safeOptOut === true;
      if (ownerExperience !== 'a_lot_today' && dogExperience !== 'not_their_thing' && !safeOptOut) {
        continue;
      }

      const evidence: ConciergeEvidence[] = [
        {
          source: 'CARE_EVENT',
          referenceId: event.id,
          occurredAt: event.occurredAt,
          label:
            ownerExperience === 'a_lot_today'
              ? 'You recently marked an Adventure as “a lot today.”'
              : dogExperience === 'not_their_thing'
                ? 'You recently marked an Adventure as not being your dog’s thing.'
                : 'You recently chose a safe opt-out and stopped the activity.',
        },
      ];

      return {
        mode: 'GENTLE' as const,
        reason:
          'Concierge is lowering suggestion intensity from recent explicit feedback, not from a health or mood inference.',
        evidence,
      };
    }

    return {
      mode: 'NORMAL' as const,
      reason:
        'No recent explicit “a lot today,” “not their thing,” or safe-stop feedback is asking Concierge to lower the pace.',
      evidence: [] as ConciergeEvidence[],
    };
  }

  private buildSuggestions(input: {
    now: Date;
    petId: string;
    petName: string;
    pace: {
      mode: 'NORMAL' | 'GENTLE';
      reason: string;
      evidence: ConciergeEvidence[];
    };
    reminders: Array<{
      id: string;
      kind: 'VET_APPOINTMENT' | 'MEDICATION' | 'GROOMING' | 'GENERAL_CARE';
      title: string;
      petId?: string;
      dueAt: string;
      status: 'SCHEDULED' | 'COMPLETED' | 'CANCELLED';
    }>;
    signals: Array<{
      id: string;
      petId: string;
      title: string;
      body: string;
      observedAt: string;
      nonDiagnostic: true;
    }>;
    connectorProviders: Array<{
      provider: string;
      label: string;
      availability: string;
    }>;
  }) {
    const suggestions: ConciergeSuggestion[] = [];
    const careReminder = input.reminders.find((reminder) => {
      if (reminder.status !== 'SCHEDULED') return false;
      if (reminder.petId && reminder.petId !== input.petId) return false;
      const due = new Date(reminder.dueAt).getTime();
      return (
        Number.isFinite(due) &&
        due >= input.now.getTime() - 24 * HOUR_MS &&
        due <= input.now.getTime() + CARE_LOOKAHEAD_MS
      );
    });

    if (careReminder) {
      const medication = careReminder.kind === 'MEDICATION';
      suggestions.push({
        id: `care:${careReminder.id}`,
        kind: 'CARE_PREP',
        priority: 'ATTENTION',
        title: medication ? 'Medication reminder coming up' : 'Care reminder coming up',
        body: medication
          ? `“${careReminder.title}” is scheduled for ${readableDueAt(careReminder.dueAt)} UTC. Follow the veterinarian or medication-label instructions you already have; Concierge does not calculate doses.`
          : `“${careReminder.title}” is scheduled for ${readableDueAt(careReminder.dueAt)} UTC. A quick prep now may make it easier to remember later.`,
        reason: 'An existing Autopilot care reminder falls within the next 72 hours.',
        evidence: [
          {
            source: 'AUTOPILOT',
            referenceId: careReminder.id,
            occurredAt: careReminder.dueAt,
            label: `Scheduled ${careReminder.kind.toLowerCase().replaceAll('_', ' ')} reminder.`,
          },
        ],
        action: { label: 'Open Autopilot', href: '/autopilot' },
        suggestionOnly: true,
      });
    }

    const signal = input.signals.find((item) => item.petId === input.petId);
    if (signal) {
      suggestions.push({
        id: `signal:${signal.id}`,
        kind: 'CHECK_IN',
        priority: 'ATTENTION',
        title: signal.title,
        body: signal.body,
        reason:
          'Autopilot surfaced a conservative tracker check-in. Concierge passes it through as non-diagnostic context.',
        evidence: [
          {
            source: 'AUTOPILOT',
            referenceId: signal.id,
            occurredAt: signal.observedAt,
            label: 'Open non-diagnostic Autopilot check-in.',
          },
        ],
        action: { label: 'Review check-in', href: '/autopilot' },
        suggestionOnly: true,
      });
    }

    if (input.pace.mode === 'GENTLE') {
      suggestions.push({
        id: 'pace:gentle',
        kind: 'RECOVERY_PACE',
        priority: 'GENTLE',
        title: `Keep ${input.petName}'s options easy today`,
        body: 'Consider a familiar, low-pressure activity or recovery-oriented option. Stopping early is still a successful choice.',
        reason: input.pace.reason,
        evidence: input.pace.evidence,
        action: { label: 'See recovery options', href: '/compass' },
        suggestionOnly: true,
      });
    }

    const reauth = input.connectorProviders.find(
      (provider) => provider.availability === 'REAUTH_REQUIRED'
    );
    if (reauth) {
      suggestions.push({
        id: `connector:${reauth.provider}`,
        kind: 'CONNECTION_ATTENTION',
        priority: 'INFO',
        title: `${reauth.label} needs reauthorization`,
        body: 'New context from this service is paused. Existing dogOS records remain unchanged, and Concierge will not pretend stale provider access is current.',
        reason:
          'Connectors reports that the authenticated provider credential is not currently usable.',
        evidence: [
          {
            source: 'CONNECTOR',
            label: `${reauth.label} is in REAUTH_REQUIRED state.`,
          },
        ],
        action: { label: 'View connected services', href: '/connectors' },
        suggestionOnly: true,
      });
    }

    const priority = { ATTENTION: 0, GENTLE: 1, INFO: 2 } as const;
    return suggestions.sort((left, right) => priority[left.priority] - priority[right.priority]);
  }
}
