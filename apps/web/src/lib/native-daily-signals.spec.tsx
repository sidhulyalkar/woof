import React, { useEffect } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
// DOM primitives exercise state and requests, not native layout or VoiceOver.
vi.mock('../../../mobile/node_modules/react-native', () => {
  const box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return {
    View: box,
    ScrollView: box,
    Text: box,
    ActivityIndicator: () => <span>Loading</span>,
    Pressable: ({
      children,
      onPress,
      disabled,
      accessibilityLabel,
    }: {
      children?: React.ReactNode;
      onPress?: () => void;
      disabled?: boolean;
      accessibilityLabel?: string;
    }) => (
      <button disabled={disabled} onClick={onPress} aria-label={accessibilityLabel}>
        {children}
      </button>
    ),
    TextInput: ({
      value,
      onChangeText,
      editable,
      accessibilityLabel,
    }: {
      value: string;
      onChangeText: (v: string) => void;
      editable?: boolean;
      accessibilityLabel: string;
    }) => (
      <textarea
        value={value}
        onChange={(e) => onChangeText(e.target.value)}
        disabled={editable === false}
        aria-label={accessibilityLabel}
      />
    ),
    StyleSheet: { create: (styles: unknown) => styles },
  };
});
vi.mock('../../../mobile/node_modules/@expo/vector-icons', () => ({ Ionicons: () => null }));
vi.mock('../../../mobile/node_modules/@react-navigation/native', () => ({
  useFocusEffect: (callback: () => void) => useEffect(callback, [callback]),
}));
vi.mock('../../../mobile/src/api/households', () => ({ householdsApi: { getMine: vi.fn() } }));
vi.mock('../../../mobile/src/api/intelligence', () => ({
  intelligenceApi: {
    currentDailySignals: vi.fn(),
    captureDailySignals: vi.fn(),
    correctDailySignals: vi.fn(),
  },
}));
import DailySignalsScreen from '../../../mobile/src/screens/DailySignalsScreen';
import { householdsApi } from '../../../mobile/src/api/households';
import { intelligenceApi } from '../../../mobile/src/api/intelligence';
const state = {
  rootCareEventId: 'root',
  currentCareEventId: 'tip',
  correctionSequence: 0,
  householdId: 'house',
  petId: 'dog',
  localDate: '2026-11-22',
  timezone: 'America/Los_Angeles',
  status: 'ORIGINAL' as const,
  signals: { energy: 'MORE' as const },
};
const receipt = {
  correctionCareEventId: 'new',
  duplicate: false,
  currentAdvanced: false,
  state,
  projectedDimensions: [],
  retractedDimensions: [],
};
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(householdsApi.getMine).mockResolvedValue([
    {
      id: 'house',
      name: 'Home',
      timezone: 'America/Los_Angeles',
      viewerRole: 'OWNER',
      pets: [
        {
          id: 'membership',
          status: 'ACTIVE',
          joinedAt: '2026-01-01',
          pet: { id: 'dog', name: 'Shasta', species: 'DOG' },
        },
      ],
    },
  ] as Awaited<ReturnType<typeof householdsApi.getMine>>);
  vi.mocked(intelligenceApi.currentDailySignals).mockResolvedValue(state);
});
async function openEditor() {
  render(<DailySignalsScreen />);
  fireEvent.click(await screen.findByText('Edit observations'));
}
it('sends a predecessor-bound correction without private notes', async () => {
  vi.mocked(intelligenceApi.correctDailySignals).mockResolvedValue(receipt);
  await openEditor();
  expect(screen.queryByText('Add a private note')).not.toBeInTheDocument();
  fireEvent.click(screen.getByText(/Save correction:/));
  await waitFor(() =>
    expect(intelligenceApi.correctDailySignals).toHaveBeenCalledWith({
      householdId: 'house',
      petId: 'dog',
      expectedCurrentCareEventId: 'tip',
      signals: { energy: 'MORE' },
    })
  );
  expect(intelligenceApi.captureDailySignals).not.toHaveBeenCalled();
});
it('retries exactly the same correction after a lost response', async () => {
  vi.mocked(intelligenceApi.correctDailySignals)
    .mockRejectedValueOnce(new Error('lost'))
    .mockResolvedValueOnce({ ...receipt, duplicate: true });
  await openEditor();
  fireEvent.click(screen.getByText(/Save correction:/));
  fireEvent.click(await screen.findByText(/Retry saving/));
  await waitFor(() => expect(intelligenceApi.correctDailySignals).toHaveBeenCalledTimes(2));
  expect(vi.mocked(intelligenceApi.correctDailySignals).mock.calls[0]).toEqual(
    vi.mocked(intelligenceApi.correctDailySignals).mock.calls[1]
  );
});
it('requires reload after a competing edit', async () => {
  vi.mocked(intelligenceApi.correctDailySignals).mockRejectedValue({
    isAxiosError: true,
    response: { status: 409 },
  });
  await openEditor();
  fireEvent.click(screen.getByText(/Save correction:/));
  await screen.findByText(/Today’s observations changed/);
  expect(screen.getByText(/Retry saving/).closest('button')).toBeDisabled();
  fireEvent.click(screen.getByText('Reload saved observations'));
  await screen.findByText('Edit observations');
});
it('allows explicitly retracting all structured observations', async () => {
  vi.mocked(intelligenceApi.correctDailySignals).mockResolvedValue({
    ...receipt,
    state: { ...state, signals: {} },
  });
  await openEditor();
  fireEvent.click(screen.getByRole('button', { name: /Energy/ }));
  fireEvent.click(screen.getByText('Leave unreported'));
  fireEvent.click(screen.getByText(/Save correction:/));
  await waitFor(() =>
    expect(intelligenceApi.correctDailySignals).toHaveBeenCalledWith(
      expect.objectContaining({ signals: {} })
    )
  );
});
it('does not let a delayed response for another dog replace the current editor', async () => {
  vi.mocked(householdsApi.getMine).mockResolvedValue([
    {
      id: 'house',
      name: 'Home',
      timezone: 'America/Los_Angeles',
      viewerRole: 'OWNER',
      pets: ['dog', 'other'].map((id) => ({
        id,
        status: 'ACTIVE',
        joinedAt: '2026-01-01',
        pet: { id, name: id, species: 'DOG' },
      })),
    },
  ]);
  let finish!: (value: typeof state) => void;
  vi.mocked(intelligenceApi.currentDailySignals).mockImplementation((_household, pet) =>
    pet === 'dog'
      ? new Promise((resolve) => {
          finish = resolve;
        })
      : Promise.resolve({ ...state, petId: 'other', currentCareEventId: 'other-tip' })
  );
  render(<DailySignalsScreen preferredPetId="dog" />);
  fireEvent.click(await screen.findByRole('button', { name: 'other, Home' }));
  fireEvent.click(await screen.findByText('Edit observations'));
  finish(state);
  vi.mocked(intelligenceApi.correctDailySignals).mockResolvedValue(receipt);
  fireEvent.click(screen.getByText(/Save correction:/));
  await waitFor(() =>
    expect(intelligenceApi.correctDailySignals).toHaveBeenCalledWith(
      expect.objectContaining({ petId: 'other', expectedCurrentCareEventId: 'other-tip' })
    )
  );
});
it('keeps a first capture sparse and freezes its timestamp across retries', async () => {
  vi.mocked(intelligenceApi.currentDailySignals).mockResolvedValue(null);
  vi.mocked(intelligenceApi.captureDailySignals).mockRejectedValue(new Error('lost response'));
  render(<DailySignalsScreen />);
  await waitFor(() => expect(intelligenceApi.currentDailySignals).toHaveBeenCalled());
  fireEvent.click(await screen.findByRole('button', { name: /Energy/ }));
  fireEvent.click(screen.getByText('More'));
  fireEvent.click(screen.getByText('Save 1 observation'));
  fireEvent.click(await screen.findByText(/Retry saving/));
  await waitFor(() => expect(intelligenceApi.captureDailySignals).toHaveBeenCalledTimes(2));
  const calls = vi.mocked(intelligenceApi.captureDailySignals).mock.calls;
  expect(calls[0]).toEqual(calls[1]);
  expect(calls[0]![0].signals).toEqual({ energy: 'MORE' });
});
