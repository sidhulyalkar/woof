import type { EvidenceType } from '../care-events/care-event.types';
import type {
  ProjectionAuthority,
  QualifiedProjectionSourceType,
} from './evidence-projection-v1.types';

export type QualifiedEvidenceMapping = {
  sourceType: QualifiedProjectionSourceType;
  authority: ProjectionAuthority;
};

export const EVIDENCE_NORMALIZATION_V1 = Object.freeze({
  version: 'evidence-normalization-v1' as const,
  ownerChoiceBuckets: {
    LESS: -1,
    USUAL: 0,
    MORE: 1,
  } as const,
  evidenceMappings: {
    SELF_REPORT: { sourceType: 'OWNER_CHECKIN', authority: 'BASELINE_ELIGIBLE' },
    ACTIVITY: { sourceType: 'ACTIVITY', authority: 'CONTEXT_ONLY' },
    COACH: { sourceType: 'COACHING', authority: 'CONTEXT_ONLY' },
  } as const satisfies Partial<Record<EvidenceType, QualifiedEvidenceMapping>>,
  maxContextBytes: 4096,
  maxNormalizationReasonLength: 512,
  maxSourceIdentityLength: 256,
  baselineDimensions: [
    'APPETITE',
    'ENERGY',
    'BATHROOM_ROUTINE',
    'MOBILITY_COMFORT',
    'ENGAGEMENT_SOCIAL_COMFORT',
    'SLEEP_REST',
  ] as const,
  contextOnlyDimensions: [
    'ACTIVITY_LOAD',
    'RECOVERY_REST_PROXY',
    'TRAINING_COMFORT_SUCCESS',
  ] as const,
});

export function qualifiedEvidenceMapping(
  evidenceType: EvidenceType
): QualifiedEvidenceMapping | null {
  const mapping =
    EVIDENCE_NORMALIZATION_V1.evidenceMappings[
      evidenceType as keyof typeof EVIDENCE_NORMALIZATION_V1.evidenceMappings
    ];
  return mapping ?? null;
}
