export type DogOsActivityType =
  | 'WALK'
  | 'RUN'
  | 'PLAY'
  | 'HIKE'
  | 'TRAINING'
  | 'GROOMING'
  | 'VET_VISIT'
  | 'ENRICHMENT'
  | 'SCENT'
  | 'PUZZLE'
  | 'SOCIAL'
  | 'MEETUP'
  | 'PARALLEL_WALK'
  | 'RECOVERY'
  | 'REST'
  | 'DECOMPRESSION'
  | 'OTHER';

export type DogOsActivityTypeInput = DogOsActivityType | Lowercase<DogOsActivityType>;

/**
 * Canonical dogOS activity write contract.
 *
 * `petIds` is the preferred multi-pet field. `petId` remains accepted so old
 * clients can roll forward without a flag-day release. If both are supplied,
 * the server de-duplicates them into one household activity.
 */
export interface CreateActivityRequest {
  petIds?: string[];

  /** @deprecated Prefer `petIds`, even for a one-dog activity. */
  petId?: string;

  householdId?: string;
  startedAt?: string;
  endedAt?: string;
  type: DogOsActivityTypeInput;
  route?: Record<string, unknown>;
  humanMetrics?: Record<string, unknown>;
  petMetrics?: Record<string, unknown>;
  jointMetrics?: Record<string, unknown>;
}

export type UpdateActivityRequest = Partial<CreateActivityRequest>;
