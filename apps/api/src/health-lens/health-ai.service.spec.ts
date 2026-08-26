import { ServiceUnavailableException } from '@nestjs/common';
import {
  normalizeHealthModelResult,
  type PetHealthModelResult,
} from './health-ai.service';

function assessment(
  overrides: Partial<PetHealthModelResult> = {}
): PetHealthModelResult {
  return {
    triage: 'monitor',
    confidence: 0.72,
    summary: 'A localized visible change is present, but the cause cannot be determined here.',
    visibleFindings: ['small area of redness'],
    possibleCategories: ['dermatologic irritation'],
    photoFeedback: {
      usable: true,
      reason: 'The area is visible.',
      betterPhotoInstructions: [],
    },
    questions: ['Is it changing quickly?'],
    ownerActions: ['Document whether the area changes over the next several hours.'],
    avoid: ['Do not use human medication based on an automated screening result.'],
    vetHandoff: {
      recommended: false,
      timing: 'not-yet',
      summary: 'Monitor and contact your veterinarian if the concern persists or worsens.',
      bring: [],
    },
    ...overrides,
  };
}

describe('Health Lens model output authority', () => {
  it('raises an emergency model result to an authoritative immediate handoff', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        triage: 'emergency_now',
        vetHandoff: {
          recommended: false,
          timing: 'not-yet',
          summary: 'Model handoff was internally inconsistent.',
          bring: [],
        },
      })
    );

    expect(normalized.vetHandoff.recommended).toBe(true);
    expect(normalized.vetHandoff.timing).toBe('now');
    expect(normalized.ownerActions[0]).toMatch(/emergency veterinarian now/i);
  });

  it('does not let vet-today or vet-soon triage carry a lower-urgency handoff', () => {
    const today = normalizeHealthModelResult(
      assessment({
        triage: 'vet_today',
        vetHandoff: {
          recommended: false,
          timing: 'routine',
          summary: 'Needs assessment.',
          bring: [],
        },
      })
    );
    const soon = normalizeHealthModelResult(
      assessment({
        triage: 'vet_soon',
        vetHandoff: {
          recommended: false,
          timing: 'not-yet',
          summary: 'Needs assessment.',
          bring: [],
        },
      })
    );

    expect(today.vetHandoff).toMatchObject({ recommended: true, timing: 'today' });
    expect(soon.vetHandoff).toMatchObject({
      recommended: true,
      timing: 'within-2-days',
    });
  });

  it('makes better-image triage authoritative over a contradictory usable-photo flag', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        triage: 'better_image',
        photoFeedback: {
          usable: true,
          reason: 'The model nevertheless requested a better image.',
          betterPhotoInstructions: ['Use even lighting and include the surrounding area.'],
        },
      })
    );

    expect(normalized.photoFeedback.usable).toBe(false);
  });

  it('fails closed on positive medication, dosing, prescription, vomiting, or invasive directives', () => {
    const unsafe = [
      'Give 25 mg of a painkiller now.',
      'Administer the medication twice today.',
      'Stop the prescription until the swelling improves.',
      'Induce vomiting at home.',
      'Drain the lesion and keep it clean.',
    ];

    for (const directive of unsafe) {
      expect(() =>
        normalizeHealthModelResult(assessment({ ownerActions: [directive] }))
      ).toThrow(ServiceUnavailableException);
    }
  });

  it('allows explicit avoid/negated safety language while bounding generated arrays', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        ownerActions: ['Do not give human medication without veterinary guidance.'],
        visibleFindings: Array.from({ length: 20 }, (_, index) => `finding-${index}`),
      })
    );

    expect(normalized.ownerActions).toEqual([
      'Do not give human medication without veterinary guidance.',
    ]);
    expect(normalized.visibleFindings).toHaveLength(8);
  });

  it('rejects malformed nested contracts even when top-level triage is valid', () => {
    expect(() =>
      normalizeHealthModelResult({
        ...assessment(),
        vetHandoff: null,
      })
    ).toThrow(new ServiceUnavailableException('Health screening model returned an invalid assessment'));
  });
});
