import 'reflect-metadata';
import { validate } from 'class-validator';
import { CreateDailySignalsDto, DailySignalsAnswersDto } from './daily-signals.dto';

function validPayload(householdId: string) {
  const signals = new DailySignalsAnswersDto();
  signals.appetite = 'USUAL';

  const dto = new CreateDailySignalsDto();
  dto.householdId = householdId;
  dto.petId = '7dd7de8d-42af-4a70-8398-bd572c3bea34';
  dto.observedAt = '2026-08-25T18:00:00.000Z';
  dto.signals = signals;
  return dto;
}

describe('Daily Signals DTO household identity', () => {
  it('accepts legacy deterministic UUID-shaped household identifiers', async () => {
    const errors = await validate(validPayload('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'));

    expect(errors).toHaveLength(0);
  });

  it('rejects malformed household identifiers', async () => {
    const errors = await validate(validPayload('not-a-household-id'));

    expect(errors.some((error) => error.property === 'householdId')).toBe(true);
  });
});
