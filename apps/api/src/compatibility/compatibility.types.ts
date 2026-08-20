export type CompatibilityFactors = Record<string, number>;

export type CompatibilityProvenance = {
  scorer: 'deterministic' | 'learned';
  modelVersion: string;
  featureVersion: string;
  calibrationVersion?: string;
  generatedAt: string;
  fallback: boolean;
  fallbackReason?: string;
};

export type CompatibilityScore = {
  compatibilityScore: number;
  confidence: number;
  source: string;
  factors: CompatibilityFactors;
  explanation: string[];
  provenance: CompatibilityProvenance;
};

export type CanonicalBehaviorFeatures = {
  energy?: number;
  sociability?: number;
  caution?: number;
  excitability?: number;
  trainability?: number;
  socialRisk?: number;
  coverage: number;
};

export type CanonicalPetCompatibilityFeatures = {
  species: string;
  breed?: string;
  ageYears?: number;
  behavior: CanonicalBehaviorFeatures;
};

export type CompatibilityOutcomeFeatures = {
  sampleCount: number;
  meanRating?: number;
  positiveRate?: number;
  repeatMeetupCount: number;
  lastOutcomeDaysAgo?: number;
};

export type LearnedCompatibilityRequest = {
  featureVersion: 'compatibility-features-v1';
  petA: CanonicalPetCompatibilityFeatures;
  petB: CanonicalPetCompatibilityFeatures;
  outcomes: CompatibilityOutcomeFeatures;
};

export function isCompatibilityScore(value: unknown): value is CompatibilityScore {
  if (!value || typeof value !== 'object') return false;
  const score = value as Partial<CompatibilityScore>;
  return (
    typeof score.compatibilityScore === 'number' &&
    Number.isFinite(score.compatibilityScore) &&
    score.compatibilityScore >= 0 &&
    score.compatibilityScore <= 1 &&
    typeof score.confidence === 'number' &&
    Number.isFinite(score.confidence) &&
    score.confidence >= 0 &&
    score.confidence <= 1 &&
    typeof score.source === 'string' &&
    Array.isArray(score.explanation) &&
    !!score.provenance &&
    typeof score.provenance.modelVersion === 'string' &&
    typeof score.provenance.featureVersion === 'string'
  );
}
