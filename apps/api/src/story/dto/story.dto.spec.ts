import { validate } from 'class-validator';
import { UpdateStoryCurationDto } from './story.dto';

const sourceId = '22222222-2222-4222-8222-222222222222';

function curationDto(action: string) {
  const dto = new UpdateStoryCurationDto();
  dto.sourceType = 'ACTIVITY';
  dto.sourceId = sourceId;
  (dto as unknown as { action: string }).action = action;
  return dto;
}

describe('UpdateStoryCurationDto', () => {
  it('accepts reversible SAVE and CLEAR actions', async () => {
    for (const action of ['SAVE', 'CLEAR']) {
      const errors = await validate(curationDto(action));
      expect(errors.find((error) => error.property === 'action')).toBeUndefined();
    }
  });

  it('rejects HIDE until Story has an explicit recovery surface', async () => {
    const errors = await validate(curationDto('HIDE'));

    expect(errors).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          property: 'action',
          constraints: expect.objectContaining({ isIn: expect.any(String) }),
        }),
      ])
    );
  });
});
