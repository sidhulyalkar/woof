import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { HealthTriageLevel } from './health-triage';

export type PetHealthModelResult = {
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

export type PetHealthModelInput = {
  pet: {
    name: string;
    species: string;
    breed: string | null;
    ageYears: number | null;
    temperament: unknown;
  };
  concern: string;
  bodyArea?: string;
  onset?: string;
  appetite?: string;
  energy?: string;
  breathing?: string;
  bathroom?: string;
  recentContext: Array<{ type: string; startedAt: string }>;
  priorHealthObservations: Array<{ createdAt: string; triage: string; summary: string }>;
  image?: {
    mimeType: string;
    bytes: Buffer;
  };
};

const HEALTH_RESULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: [
    'triage',
    'confidence',
    'summary',
    'visibleFindings',
    'possibleCategories',
    'photoFeedback',
    'questions',
    'ownerActions',
    'avoid',
    'vetHandoff',
  ],
  properties: {
    triage: {
      type: 'string',
      enum: [
        'emergency_now',
        'vet_today',
        'vet_soon',
        'monitor',
        'better_image',
        'insufficient_information',
      ],
    },
    confidence: { type: 'number', minimum: 0, maximum: 1 },
    summary: { type: 'string' },
    visibleFindings: { type: 'array', items: { type: 'string' }, maxItems: 8 },
    possibleCategories: { type: 'array', items: { type: 'string' }, maxItems: 6 },
    photoFeedback: {
      type: 'object',
      additionalProperties: false,
      required: ['usable', 'reason', 'betterPhotoInstructions'],
      properties: {
        usable: { type: 'boolean' },
        reason: { type: 'string' },
        betterPhotoInstructions: { type: 'array', items: { type: 'string' }, maxItems: 5 },
      },
    },
    questions: { type: 'array', items: { type: 'string' }, maxItems: 6 },
    ownerActions: { type: 'array', items: { type: 'string' }, maxItems: 6 },
    avoid: { type: 'array', items: { type: 'string' }, maxItems: 6 },
    vetHandoff: {
      type: 'object',
      additionalProperties: false,
      required: ['recommended', 'timing', 'summary', 'bring'],
      properties: {
        recommended: { type: 'boolean' },
        timing: {
          type: 'string',
          enum: ['now', 'today', 'within-2-days', 'routine', 'not-yet'],
        },
        summary: { type: 'string' },
        bring: { type: 'array', items: { type: 'string' }, maxItems: 6 },
      },
    },
  },
} as const;

const SYSTEM_PROMPT = `You are Woof Health Lens, a conservative veterinary health screening assistant for pet owners.

Your role is NOT to diagnose disease from a photo or chat. Your job is to:
1. describe only what is actually visible or reported,
2. recognize when information or photo quality is inadequate,
3. ask the smallest number of useful follow-up questions,
4. provide safe observation and documentation steps,
5. recommend veterinary evaluation at an appropriate urgency when warranted.

Safety rules:
- Never state that a pet definitely has a disease based on a photo or chat.
- Use broad possibility categories such as dermatologic irritation, trauma, ocular concern, dental/oral concern, gastrointestinal concern, mobility concern, or infection/inflammation concern rather than definitive diagnoses.
- Never provide medication doses, recommend human medications, tell an owner to start/stop prescriptions, induce vomiting, drain lesions, lance masses, or perform invasive treatment.
- Do not reassure away serious symptoms because a photo looks mild.
- Err toward veterinary evaluation when there is breathing difficulty, collapse, altered consciousness, ongoing/repeated seizure, uncontrolled bleeding, poisoning/toxin exposure, urinary obstruction, heat stroke, major trauma, severe pain, rapidly worsening signs, eye injury, or a distended abdomen with unproductive retching.
- If the photo is blurry, too dark, too distant, occluded by fur, poorly framed, or lacks scale/context, set triage to better_image unless the reported symptoms independently justify more urgent care.
- A better photo request should explain exactly what view, lighting, distance, or scale would help.
- Separate visible findings from possible categories. Do not imply that a visible finding confirms a category.
- If serious/persistent symptoms are reported, recommend a veterinarian even if the image is inconclusive.
- Keep owner actions limited to safe observation, documentation, preventing licking/trauma when practical, rest/normal hydration access, and arranging veterinary care. Do not prescribe treatment.

Return only the requested structured JSON.`;

@Injectable()
export class HealthAiService {
  private readonly logger = new Logger(HealthAiService.name);
  private readonly apiKey: string | null;
  private readonly model: string;
  private readonly timeoutMs: number;

  constructor(private readonly config: ConfigService) {
    this.apiKey = this.config.get<string>('OPENAI_API_KEY') || null;
    this.model = this.config.get<string>('OPENAI_HEALTH_MODEL') || 'gpt-5.6-luna';
    this.timeoutMs = Math.max(3000, Math.min(30000, Number(this.config.get('OPENAI_HEALTH_TIMEOUT_MS') || 12000)));
  }

  isConfigured() {
    return Boolean(this.apiKey);
  }

  async analyze(input: PetHealthModelInput): Promise<PetHealthModelResult> {
    if (!this.apiKey) {
      throw new ServiceUnavailableException('Multimodal health screening is not configured in this environment');
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const userContent: Array<Record<string, unknown>> = [
        {
          type: 'input_text',
          text: JSON.stringify({
            pet: input.pet,
            concern: input.concern,
            bodyArea: input.bodyArea ?? null,
            onset: input.onset ?? null,
            appetite: input.appetite ?? null,
            energy: input.energy ?? null,
            breathing: input.breathing ?? null,
            bathroom: input.bathroom ?? null,
            recentContext: input.recentContext,
            priorHealthObservations: input.priorHealthObservations,
          }),
        },
      ];

      if (input.image) {
        userContent.push({
          type: 'input_image',
          detail: 'high',
          image_url: `data:${input.image.mimeType};base64,${input.image.bytes.toString('base64')}`,
        });
      }

      const response = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: this.model,
          store: false,
          input: [
            { role: 'system', content: [{ type: 'input_text', text: SYSTEM_PROMPT }] },
            { role: 'user', content: userContent },
          ],
          text: {
            format: {
              type: 'json_schema',
              name: 'woof_pet_health_lens',
              strict: true,
              schema: HEALTH_RESULT_SCHEMA,
            },
          },
        }),
      });

      if (!response.ok) {
        const body = await response.text();
        this.logger.warn(`Health model request failed with ${response.status}: ${body.slice(0, 500)}`);
        throw new ServiceUnavailableException('Health screening model is temporarily unavailable');
      }

      const payload = (await response.json()) as {
        output?: Array<{ content?: Array<{ type?: string; text?: string }> }>;
      };
      const text = payload.output
        ?.flatMap((item) => item.content ?? [])
        .find((item) => item.type === 'output_text' && typeof item.text === 'string')?.text;

      if (!text) {
        throw new ServiceUnavailableException('Health screening model returned no usable assessment');
      }

      return this.validateResult(JSON.parse(text) as PetHealthModelResult);
    } catch (error) {
      if (error instanceof ServiceUnavailableException) throw error;
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ServiceUnavailableException('Health screening model timed out');
      }
      this.logger.error(`Health model analysis failed: ${error instanceof Error ? error.message : 'unknown error'}`);
      throw new ServiceUnavailableException('Health screening model is temporarily unavailable');
    } finally {
      clearTimeout(timeout);
    }
  }

  async followUp(input: {
    prior: PetHealthModelResult;
    petName: string;
    message: string;
  }): Promise<PetHealthModelResult> {
    return this.analyze({
      pet: {
        name: input.petName,
        species: 'UNKNOWN',
        breed: null,
        ageYears: null,
        temperament: null,
      },
      concern: `Previous screening: ${JSON.stringify(input.prior)}\nOwner follow-up: ${input.message}`,
      recentContext: [],
      priorHealthObservations: [],
    });
  }

  private validateResult(result: PetHealthModelResult): PetHealthModelResult {
    const allowed = new Set<HealthTriageLevel>([
      'emergency_now',
      'vet_today',
      'vet_soon',
      'monitor',
      'better_image',
      'insufficient_information',
    ]);
    if (!result || !allowed.has(result.triage) || typeof result.summary !== 'string') {
      throw new ServiceUnavailableException('Health screening model returned an invalid assessment');
    }

    return {
      ...result,
      confidence: Math.max(0, Math.min(1, Number(result.confidence) || 0)),
      visibleFindings: Array.isArray(result.visibleFindings) ? result.visibleFindings.slice(0, 8) : [],
      possibleCategories: Array.isArray(result.possibleCategories)
        ? result.possibleCategories.slice(0, 6)
        : [],
      questions: Array.isArray(result.questions) ? result.questions.slice(0, 6) : [],
      ownerActions: Array.isArray(result.ownerActions) ? result.ownerActions.slice(0, 6) : [],
      avoid: Array.isArray(result.avoid) ? result.avoid.slice(0, 6) : [],
    };
  }
}
