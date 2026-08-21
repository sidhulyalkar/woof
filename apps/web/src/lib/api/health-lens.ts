import { apiClient } from './client';

export type HealthTriageLevel =
  | 'emergency_now'
  | 'vet_today'
  | 'vet_soon'
  | 'monitor'
  | 'better_image'
  | 'insufficient_information';

export type HealthAssessment = {
  triage: HealthTriageLevel;
  confidence: number;
  summary: string;
  visibleFindings: string[];
  possibleCategories: string[];
  photoFeedback: {
    usable: boolean;
    reason: string;
    betterPhotoInstructions: string[];
  };
  questions: string[];
  ownerActions: string[];
  avoid: string[];
  vetHandoff: {
    recommended: boolean;
    timing: 'now' | 'today' | 'within-2-days' | 'routine' | 'not-yet';
    summary: string;
    bring: string[];
  };
};

export type HealthLensResult = {
  assessmentId: string | null;
  generatedAt: string;
  pet: { id: string; name: string; species: string; breed?: string | null };
  assessment: HealthAssessment;
  provenance: {
    version: string;
    pathway: string;
    imageAnalyzed: boolean;
    modelConfigured: boolean;
    savedToTimeline: boolean;
  };
  privacy: {
    imageStoredByWoof: boolean;
    imagePolicy: string;
  };
  safety: string;
};

export type HealthTimelineEntry = {
  id: string;
  kind: 'assessment' | 'follow-up';
  createdAt: string;
  triage: HealthTriageLevel;
  summary: string;
  bodyArea: string | null;
  concern: string | null;
  hadImage: boolean;
};

export type HealthLensInput = {
  petId: string;
  concern: string;
  bodyArea?: string;
  onset?: string;
  appetite?: 'normal' | 'mild-change' | 'major-change' | 'unknown';
  energy?: 'normal' | 'mild-change' | 'major-change' | 'unknown';
  breathing?: 'normal' | 'mild-change' | 'major-change' | 'unknown';
  bathroom?: 'normal' | 'mild-change' | 'major-change' | 'unknown';
  saveToTimeline?: boolean;
  image?: File | Blob | null;
};

export const healthLensApi = {
  analyze: async (input: HealthLensInput) => {
    const form = new FormData();
    form.append('petId', input.petId);
    form.append('concern', input.concern);
    if (input.bodyArea) form.append('bodyArea', input.bodyArea);
    if (input.onset) form.append('onset', input.onset);
    if (input.appetite) form.append('appetite', input.appetite);
    if (input.energy) form.append('energy', input.energy);
    if (input.breathing) form.append('breathing', input.breathing);
    if (input.bathroom) form.append('bathroom', input.bathroom);
    form.append('saveToTimeline', String(input.saveToTimeline !== false));
    if (input.image) {
      const filename =
        input.image instanceof File ? input.image.name : `health-capture-${Date.now()}.jpg`;
      form.append('image', input.image, filename);
    }

    // Do not set Content-Type manually. The browser/Axios adapter must attach the multipart boundary.
    return apiClient.post('/health-lens/analyze', form) as unknown as Promise<HealthLensResult>;
  },

  followUp: async (assessmentId: string, message: string) =>
    apiClient.post('/health-lens/follow-up', { assessmentId, message }) as unknown as Promise<{
      followUpId: string;
      assessmentId: string;
      generatedAt: string;
      assessment: HealthAssessment;
      provenance: { version: string; pathway: string; imageReused: boolean };
      safety: string;
    }>,

  timeline: async (petId: string) =>
    apiClient.get('/health-lens/timeline', { params: { petId } }) as unknown as Promise<
      HealthTimelineEntry[]
    >,

  deleteTimelineEntry: async (entryId: string) =>
    apiClient.delete(`/health-lens/timeline/${entryId}`) as unknown as Promise<{ deleted: boolean }>,
};
