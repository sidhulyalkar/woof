import { buildTriageText, screenEmergencyText } from './health-triage';

describe('pet health emergency screening', () => {
  it('escalates breathing distress before model analysis', () => {
    const result = screenEmergencyText('My dog is struggling to breathe and has blue gums');
    expect(result?.level).toBe('emergency_now');
    expect(result?.matchedSignals).toContain('breathing-distress');
  });

  it('escalates possible gastric dilation warning signs', () => {
    const result = screenEmergencyText('His belly is swollen and hard and he keeps trying to vomit but nothing comes up');
    expect(result?.level).toBe('emergency_now');
    expect(result?.matchedSignals).toContain('gdv-bloat');
  });

  it('escalates inability to urinate', () => {
    const result = screenEmergencyText('She is straining to urinate and no urine is coming out');
    expect(result?.level).toBe('emergency_now');
    expect(result?.matchedSignals).toContain('urinary-obstruction');
  });

  it('does not invent an emergency from a localized mild concern', () => {
    expect(screenEmergencyText('Small red patch on paw noticed this morning, otherwise acting normal')).toBeNull();
  });

  it('includes structured major changes in the triage text', () => {
    const text = buildTriageText({
      concern: 'Something seems off',
      breathing: 'major-change',
      energy: 'normal',
      bathroom: 'normal',
      appetite: 'normal',
    });
    expect(text).toContain('major change in breathing');
  });
});
