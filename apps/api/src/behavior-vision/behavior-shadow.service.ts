import { Injectable } from '@nestjs/common';
import { BehaviorVisionService } from './behavior-vision.service';
import type {
  BehaviorContext,
  BehaviorEvidenceSource,
  BehaviorPhase,
  StoredBehaviorObservation,
} from './behavior-vision.types';

export const BEHAVIOR_SHADOW_POLICY_VERSION = 'woof-behavior-shadow-v1';

const READINESS_GATES = {
  usableObservations: 20,
  ownerReviewedObservations: 10,
  confirmationRate: 0.8,
  contexts: 3,
  pairedSessions: 5,
} as const;

export type BehaviorMoment = {
  observationId: string;
  observationCreatedAt: string;
  context: BehaviorContext;
  phase: BehaviorPhase;
  startMs: number;
  endMs: number;
  confidence: number;
  labels: string[];
  sources: BehaviorEvidenceSource[];
};

@Injectable()
export class BehaviorShadowService {
  constructor(private readonly behaviorVision: BehaviorVisionService) {}

  async snapshot(userId: string, petId: string) {
    const [observations, profile] = await Promise.all([
      this.behaviorVision.timeline(userId, petId, 100),
      this.behaviorVision.profile(userId, petId),
    ]);

    const reviewed = observations.filter((entry) => entry.ownerFeedback !== undefined);
    const confirmed = reviewed.filter((entry) => entry.ownerFeedback?.accurate === true);
    const rejected = reviewed.filter((entry) => entry.ownerFeedback?.accurate === false);
    const usable = observations.filter(
      (entry) => entry.analysis.mediaQuality.usable && entry.ownerFeedback?.accurate !== false
    );
    const pairedSessions = this.countPairedSessions(usable);
    const confirmationRate = reviewed.length ? confirmed.length / reviewed.length : null;
    const evidenceReady =
      usable.length >= READINESS_GATES.usableObservations &&
      reviewed.length >= READINESS_GATES.ownerReviewedObservations &&
      confirmationRate !== null &&
      confirmationRate >= READINESS_GATES.confirmationRate &&
      profile.contextsSeen.length >= READINESS_GATES.contexts &&
      pairedSessions >= READINESS_GATES.pairedSessions;

    return {
      policy: {
        version: BEHAVIOR_SHADOW_POLICY_VERSION,
        mode: 'shadow-evidence-only' as const,
        canInfluenceCompatibility: false as const,
        canMutateCanonicalPetState: false as const,
        canMakeSafetyDecision: false as const,
        promotionEnabled: false as const,
        promotionRequiresSeparateQualifiedRelease: true as const,
      },
      evaluation: {
        observations: observations.length,
        usableObservations: usable.length,
        ownerReviewedObservations: reviewed.length,
        ownerConfirmedObservations: confirmed.length,
        ownerRejectedObservations: rejected.length,
        ownerUnreviewedObservations: observations.length - reviewed.length,
        confirmationRate,
        usableRate: observations.length ? usable.length / observations.length : 0,
        contextsSeen: profile.contextsSeen.length,
        pairedSessions,
        personalizationConfidence: profile.personalizationConfidence,
        modelVersions: [
          ...new Set(observations.map((entry) => entry.analysis.modelVersion)),
        ].sort(),
        evidenceReady,
        readinessGates: READINESS_GATES,
      },
      moments: observations.flatMap((observation) => this.deriveMoments(observation)),
    };
  }

  private countPairedSessions(observations: StoredBehaviorObservation[]) {
    const sessions = new Map<string, Set<string>>();
    for (const observation of observations) {
      const sessionKey = observation.context.sessionKey;
      if (!sessionKey) continue;
      const phases = sessions.get(sessionKey) ?? new Set<string>();
      phases.add(observation.context.phase);
      sessions.set(sessionKey, phases);
    }
    return [...sessions.values()].filter(
      (phases) =>
        phases.has('baseline') && (phases.has('during-intervention') || phases.has('recovery'))
    ).length;
  }

  private deriveMoments(observation: StoredBehaviorObservation): BehaviorMoment[] {
    const timed = observation.analysis.evidence
      .filter(
        (entry) =>
          Number.isFinite(entry.startMs) &&
          Number.isFinite(entry.endMs) &&
          (entry.endMs ?? -1) >= (entry.startMs ?? 0)
      )
      .map((entry) => ({
        startMs: Math.max(0, entry.startMs ?? 0),
        endMs: Math.max(0, entry.endMs ?? entry.startMs ?? 0),
        confidence: Math.max(0, Math.min(1, entry.confidence)),
        label: entry.label,
        source: entry.source,
      }))
      .sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);

    const groups: Array<{
      startMs: number;
      endMs: number;
      confidence: number;
      labels: Set<string>;
      sources: Set<BehaviorEvidenceSource>;
    }> = [];

    for (const entry of timed) {
      const current = groups.at(-1);
      if (
        current &&
        entry.startMs <= current.endMs + 1000 &&
        entry.endMs - current.startMs <= 12_000
      ) {
        current.endMs = Math.max(current.endMs, entry.endMs);
        current.confidence = Math.max(current.confidence, entry.confidence);
        current.labels.add(entry.label);
        current.sources.add(entry.source);
        continue;
      }
      groups.push({
        startMs: entry.startMs,
        endMs: entry.endMs,
        confidence: entry.confidence,
        labels: new Set([entry.label]),
        sources: new Set([entry.source]),
      });
    }

    return groups.slice(0, 12).map((group) => ({
      observationId: observation.id,
      observationCreatedAt: observation.createdAt,
      context: observation.context.context,
      phase: observation.context.phase,
      startMs: group.startMs,
      endMs: group.endMs,
      confidence: group.confidence,
      labels: [...group.labels].slice(0, 8),
      sources: [...group.sources].slice(0, 6),
    }));
  }
}
