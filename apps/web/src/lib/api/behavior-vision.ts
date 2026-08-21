import { apiClient } from './client';

export type BehaviorDimension =
  | 'arousal'
  | 'body-tension'
  | 'social-orientation'
  | 'approach-tendency'
  | 'avoidance-tendency'
  | 'handler-engagement'
  | 'environment-engagement'
  | 'recovery';

export type BehaviorContext =
  | 'home'
  | 'street'
  | 'park'
  | 'dog-park'
  | 'trail'
  | 'daycare'
  | 'vet'
  | 'training-class'
  | 'vehicle'
  | 'other';

export type HandlerAction =
  | 'none'
  | 'increase-distance'
  | 'decrease-distance'
  | 'loosen-leash'
  | 'tighten-leash'
  | 'single-cue'
  | 'repeated-cues'
  | 'find-it'
  | 'parallel-walk'
  | 'u-turn'
  | 'pause-and-observe'
  | 'allow-greeting'
  | 'end-interaction'
  | 'other';

export type BehaviorVisionAnalysis = {
  schemaVersion: string;
  modelVersion: string;
  featureVersion: string;
  mediaQuality: {
    usable: boolean;
    confidence: number;
    issues: string[];
    recaptureInstructions: string[];
  };
  evidence: Array<{
    label: string;
    source: string;
    confidence: number;
    startMs?: number;
    endMs?: number;
  }>;
  dimensions: Array<{
    dimension: BehaviorDimension;
    value: number;
    confidence: number;
    basis: string[];
  }>;
  hypotheses: Array<{
    id: string;
    confidence: number;
    statement: string;
    supportingEvidence: string[];
    contradictoryEvidence: string[];
  }>;
  observableSummary: string;
  uncertainty: string;
};

export type IndividualBehaviorProfile = {
  schemaVersion: string;
  petId: string;
  sampleCount: number;
  contextsSeen: BehaviorContext[];
  baselines: Array<{
    dimension: BehaviorDimension;
    mean: number;
    confidence: number;
    sampleCount: number;
  }>;
  interventionEffects: Array<{
    action: HandlerAction;
    pairedSessions: number;
    arousalDelta: number | null;
    tensionDelta: number | null;
    engagementDelta: number | null;
    confidence: number;
  }>;
  personalizationConfidence: number;
  recommendation: {
    headline: string;
    explanation: string;
    nextSafeExperiment: string[];
    neverAutoRecommendGreeting: true;
  };
};

export type BehaviorVisionResult = {
  observationId: string | null;
  generatedAt: string;
  pet: { id: string; name: string; species: string; breed?: string | null };
  context: {
    context: BehaviorContext;
    sessionKey?: string;
    phase: 'baseline' | 'during-intervention' | 'recovery';
    handlerAction: HandlerAction;
    leashState: 'off-leash' | 'loose' | 'tight' | 'unknown';
    otherDogsPresent: boolean;
    otherDogDistanceMeters?: number;
    familiarDog?: boolean;
    audioAnalysisAllowed: boolean;
    ownerNote?: string;
  };
  analysis: BehaviorVisionAnalysis;
  coach: {
    headline: string;
    explanation: string;
    observableSummary?: string;
    hypothesis?: { statement: string; confidence: number; caveat: string } | null;
    nextSteps: string[];
    socialSafety?: string;
  };
  profile: IndividualBehaviorProfile;
  provenance: {
    pathway: string;
    schemaVersion: string;
    modelConfigured: boolean;
    savedToTimeline: boolean;
  };
  privacy: {
    mediaStoredByWoof: boolean;
    audioAnalysisAllowed: boolean;
    mediaPolicy: string;
  };
  safety: string;
};

export type BehaviorVisionInput = {
  petId: string;
  media: File | Blob;
  context: BehaviorContext;
  sessionKey?: string;
  phase?: 'baseline' | 'during-intervention' | 'recovery';
  handlerAction?: HandlerAction;
  leashState?: 'off-leash' | 'loose' | 'tight' | 'unknown';
  otherDogsPresent: boolean;
  otherDogDistanceMeters?: number;
  familiarDog?: boolean;
  includeAudio?: boolean;
  ownerNote?: string;
  question?: string;
  saveToTimeline?: boolean;
};

export const behaviorVisionApi = {
  analyze: async (input: BehaviorVisionInput) => {
    const form = new FormData();
    form.append('petId', input.petId);
    form.append('context', input.context);
    form.append('otherDogsPresent', String(input.otherDogsPresent));
    form.append('includeAudio', String(input.includeAudio === true));
    form.append('saveToTimeline', String(input.saveToTimeline !== false));
    if (input.sessionKey) form.append('sessionKey', input.sessionKey);
    if (input.phase) form.append('phase', input.phase);
    if (input.handlerAction) form.append('handlerAction', input.handlerAction);
    if (input.leashState) form.append('leashState', input.leashState);
    if (input.otherDogDistanceMeters !== undefined) {
      form.append('otherDogDistanceMeters', String(input.otherDogDistanceMeters));
    }
    if (input.familiarDog !== undefined) form.append('familiarDog', String(input.familiarDog));
    if (input.ownerNote) form.append('ownerNote', input.ownerNote);
    if (input.question) form.append('question', input.question);
    const filename = input.media instanceof File ? input.media.name : `behavior-${Date.now()}.webm`;
    form.append('media', input.media, filename);

    return apiClient.post('/behavior-vision/analyze', form) as unknown as Promise<BehaviorVisionResult>;
  },

  profile: async (petId: string) =>
    apiClient.get('/behavior-vision/profile', { params: { petId } }) as unknown as Promise<
      IndividualBehaviorProfile
    >,

  timeline: async (petId: string, limit = 30) =>
    apiClient.get('/behavior-vision/timeline', { params: { petId, limit } }) as unknown as Promise<
      Array<{
        id: string;
        petId: string;
        createdAt: string;
        mediaType: 'image' | 'video';
        context: BehaviorVisionResult['context'];
        analysis: BehaviorVisionAnalysis;
        ownerFeedback?: { accurate: boolean; note?: string };
      }>
    >,

  feedback: async (observationId: string, accurate: boolean, note?: string) =>
    apiClient.post('/behavior-vision/feedback', {
      observationId,
      accurate,
      note,
    }) as unknown as Promise<{
      feedbackId: string;
      observationId: string;
      accurate: boolean;
      createdAt: string;
      learning: string;
    }>,

  deleteObservation: async (observationId: string) =>
    apiClient.delete(`/behavior-vision/observations/${observationId}`) as unknown as Promise<{
      deleted: boolean;
    }>,
};