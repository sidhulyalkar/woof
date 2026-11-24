import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { isAxiosError } from 'axios';
import { householdsApi, type HouseholdSnapshot } from '../api/households';
import {
  intelligenceApi,
  type DailySignalChoice,
  type DailySignalsAnswers,
  type DailySignalsEffectiveState,
  type CreateDailySignalsInput,
  type CorrectDailySignalsInput,
} from '../api/intelligence';
import { colors } from '../theme/tokens';

const dimensions: {
  key: keyof DailySignalsAnswers;
  label: string;
  prompt: string;
}[] = [
  { key: 'appetite', label: 'Appetite', prompt: 'How was eating compared with usual?' },
  { key: 'energy', label: 'Energy', prompt: 'How was energy compared with usual?' },
  {
    key: 'bathroomRoutine',
    label: 'Bathroom / routine',
    prompt: 'Did bathroom habits or routine feel different?',
  },
  {
    key: 'mobilityComfort',
    label: 'Mobility / comfort',
    prompt: 'How comfortable did movement seem?',
  },
  {
    key: 'engagementSocialComfort',
    label: 'Engagement',
    prompt: 'How engaged or socially comfortable did they seem?',
  },
  { key: 'sleepRest', label: 'Sleep / rest', prompt: 'How restful did rest seem?' },
];

const choices: { value: DailySignalChoice; label: string }[] = [
  { value: 'LESS', label: 'Less' },
  { value: 'USUAL', label: 'Usual' },
  { value: 'MORE', label: 'More' },
  { value: 'UNSURE', label: 'Not sure' },
];

const choiceLabel = new Map<DailySignalChoice, string>(
  choices.map((choice) => [choice.value, choice.label])
);

type HouseholdPetContext = {
  householdId: string;
  householdName: string;
  timezone?: string | null;
  petId: string;
  petName: string;
};

type DailySignalsSuccess = {
  petName: string;
  savedCount: number;
  unsureCount: number;
  receipt: { duplicate: boolean };
};

type Props = {
  preferredPetId?: string;
  onDone?: () => void;
};

function contextsFromHouseholds(households: HouseholdSnapshot[]): HouseholdPetContext[] {
  return households.flatMap((household) =>
    household.pets
      .filter((membership) => membership.status === 'ACTIVE')
      .map((membership) => ({
        householdId: household.id,
        householdName: household.name,
        timezone: household.timezone,
        petId: membership.pet.id,
        petName: membership.pet.name,
      }))
  );
}

function contextKey(context: Pick<HouseholdPetContext, 'householdId' | 'petId'>) {
  return `${context.householdId}:${context.petId}`;
}

function resolveSelectedContextKey(
  contexts: HouseholdPetContext[],
  currentKey: string | null,
  preferredPetId?: string
) {
  if (currentKey && contexts.some((context) => contextKey(context) === currentKey)) {
    return currentKey;
  }

  if (preferredPetId) {
    const preferred = contexts.filter((context) => context.petId === preferredPetId);
    if (preferred.length === 1) return contextKey(preferred[0]!);
  }

  if (contexts.length === 1) return contextKey(contexts[0]!);
  return null;
}

export default function DailySignalsScreen({ preferredPetId, onDone }: Props) {
  const [contexts, setContexts] = useState<HouseholdPetContext[]>([]);
  const [selectedContextKey, setSelectedContextKey] = useState<string | null>(null);
  const [answers, setAnswers] = useState<DailySignalsAnswers>({});
  const [expandedDimension, setExpandedDimension] = useState<keyof DailySignalsAnswers | null>(
    null
  );
  const [noteOpen, setNoteOpen] = useState(false);
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<DailySignalsSuccess | null>(null);

  const [current, setCurrent] = useState<DailySignalsEffectiveState | null>(null);
  const [stateReady, setStateReady] = useState(false);
  const [editing, setEditing] = useState(false);
  const [submissionLocked, setSubmissionLocked] = useState(false);
  const stateGeneration = useRef(0);
  const savingRef = useRef(false);
  const pending = useRef<
    | { kind: 'capture'; input: CreateDailySignalsInput }
    | { kind: 'correction'; input: CorrectDailySignalsInput }
    | null
  >(null);

  const resetDraft = useCallback(() => {
    setAnswers({});
    setExpandedDimension(null);
    setNote('');
    setNoteOpen(false);
    setSuccess(null);
    setError(null);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const households = await householdsApi.getMine();
      const nextContexts = contextsFromHouseholds(households);
      setContexts(nextContexts);
      setSelectedContextKey((current) =>
        resolveSelectedContextKey(nextContexts, current, preferredPetId)
      );
      setError(null);
    } catch {
      setError('Woof could not load your household context. No check-in was recorded.');
    } finally {
      setLoading(false);
    }
  }, [preferredPetId]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const selected = contexts.find((context) => contextKey(context) === selectedContextKey) ?? null;
  const loadCurrent = useCallback(async () => {
    const generation = ++stateGeneration.current;
    setStateReady(false);
    setSaving(false);
    setCurrent(null);
    setEditing(false);
    setSubmissionLocked(false);
    pending.current = null;
    resetDraft();
    if (!selectedContextKey) return;
    const [householdId, petId] = selectedContextKey.split(':');
    if (!householdId || !petId) return;
    try {
      const state = await intelligenceApi.currentDailySignals(householdId, petId);
      if (generation !== stateGeneration.current) return;
      setCurrent(state);
      setAnswers(state?.signals ?? {});
      setEditing(!state);
      setStateReady(true);
    } catch {
      if (generation === stateGeneration.current)
        setError('Could not load today’s observations. Reload before editing.');
    }
  }, [selectedContextKey, resetDraft]);

  useEffect(() => {
    void loadCurrent();
    return () => {
      stateGeneration.current += 1;
    };
  }, [loadCurrent]);

  const answeredCount = useMemo(
    () => Object.values(answers).filter((value) => value !== undefined).length,
    [answers]
  );
  const unsureCount = useMemo(
    () => Object.values(answers).filter((value) => value === 'UNSURE').length,
    [answers]
  );

  const chooseContext = (key: string) => {
    if (key === selectedContextKey || savingRef.current) return;
    stateGeneration.current += 1;
    setStateReady(false);
    setCurrent(null);
    setEditing(false);
    setSelectedContextKey(key);
    resetDraft();
  };

  const setChoice = (key: keyof DailySignalsAnswers, value: DailySignalChoice) => {
    if (!stateReady || !editing || submissionLocked || savingRef.current) return;
    setAnswers((current) => ({ ...current, [key]: value }));
    setExpandedDimension(null);
    setSuccess(null);
    setError(null);
  };

  const clearChoice = (key: keyof DailySignalsAnswers) => {
    if (!stateReady || !editing || submissionLocked || savingRef.current) return;
    setAnswers((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    setExpandedDimension(null);
    setSuccess(null);
    setError(null);
  };

  const save = async () => {
    if (
      !selected ||
      !stateReady ||
      !editing ||
      savingRef.current ||
      (!current && answeredCount === 0)
    )
      return;
    const generation = stateGeneration.current;
    const savedCount = answeredCount;
    const savedUnsureCount = unsureCount;
    pending.current ??= current
      ? {
          kind: 'correction',
          input: {
            householdId: selected.householdId,
            petId: selected.petId,
            expectedCurrentCareEventId: current.currentCareEventId,
            signals: { ...answers },
          },
        }
      : {
          kind: 'capture',
          input: {
            householdId: selected.householdId,
            petId: selected.petId,
            observedAt: new Date().toISOString(),
            signals: { ...answers },
            ...(note.trim() ? { note: note.trim() } : {}),
          },
        };
    const submission = pending.current;
    savingRef.current = true;
    setSaving(true);
    setSubmissionLocked(true);
    setError(null);
    try {
      const receipt =
        submission.kind === 'correction'
          ? await intelligenceApi.correctDailySignals(submission.input)
          : await intelligenceApi.captureDailySignals(submission.input);
      if (generation !== stateGeneration.current) return;
      setSuccess({ petName: selected.petName, savedCount, unsureCount: savedUnsureCount, receipt });
      setEditing(false);
      setNote('');
      setNoteOpen(false);
      pending.current = null;
      try {
        const state = await intelligenceApi.currentDailySignals(
          selected.householdId,
          selected.petId
        );
        if (generation !== stateGeneration.current) return;
        setCurrent(state);
        setAnswers(state?.signals ?? {});
        setEditing(!state);
        setStateReady(true);
        setSubmissionLocked(false);
      } catch {
        if (generation === stateGeneration.current) {
          setStateReady(false);
          setError('Saved successfully. Reload today’s observations before making another edit.');
        }
      }
    } catch (caught) {
      if (generation !== stateGeneration.current) return;
      const status = isAxiosError(caught) ? caught.response?.status : undefined;
      if (status === 409 || status === 404) {
        setStateReady(false);
        setError(
          'Today’s observations changed. Reload the saved version before making your correction. Reloading replaces this draft.'
        );
      } else {
        setError(
          'We could not confirm this save. Retry sends the same observations; it will not count them twice.'
        );
      }
    } finally {
      savingRef.current = false;
      if (generation === stateGeneration.current) setSaving(false);
    }
  };

  if (loading && contexts.length === 0) {
    return (
      <View style={styles.centered} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Loading household context…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.eyebrow}>PRIVATE CHECK-IN</Text>
      <Text style={styles.title}>Anything different today?</Text>
      <Text style={styles.subtitle}>
        Only mark what you actually noticed. Skipping the rest leaves it unknown.
      </Text>
      <Text style={styles.boundaryText}>This is not a diagnosis or health score.</Text>

      {selected && (
        <View style={styles.noticeCard}>
          <Text style={styles.noticeTitle}>
            {current
              ? editing
                ? 'Edit today’s observations'
                : 'Today’s saved observations'
              : 'Today’s observations'}
          </Text>
          {current && (
            <Text style={styles.noticeText}>
              Changes replace the reported observations for today. Private notes are unchanged.
              Leaving every item unreported clears these observations.
            </Text>
          )}
          {current && stateReady && !editing && (
            <Pressable
              accessibilityRole="button"
              style={styles.clearButton}
              onPress={() => {
                setEditing(true);
                setSuccess(null);
              }}
            >
              <Text style={styles.clearButtonText}>Edit observations</Text>
            </Pressable>
          )}
          {!stateReady && (
            <Pressable
              accessibilityRole="button"
              style={styles.clearButton}
              accessibilityState={{ disabled: saving }}
              disabled={saving}
              onPress={() => void loadCurrent()}
            >
              <Text style={styles.clearButtonText}>Reload saved observations</Text>
            </Pressable>
          )}
        </View>
      )}

      {contexts.length === 0 ? (
        <View style={styles.noticeCard}>
          <Ionicons name="paw-outline" size={22} color={colors.primary[700]} />
          <Text style={styles.noticeTitle}>No active household pet context yet</Text>
          <Text style={styles.noticeText}>
            Create or join an authorized pet household before recording Daily Signals.
          </Text>
        </View>
      ) : (
        <>
          <View style={styles.contextSection}>
            <Text style={styles.sectionEyebrow}>FOR</Text>
            {selected ? (
              <Text style={styles.contextSummary}>
                {selected.petName}
                {contexts.filter((context) => context.petId === selected.petId).length > 1
                  ? ` · ${selected.householdName}`
                  : ''}
              </Text>
            ) : (
              <Text style={styles.contextSummary}>Choose dog + household</Text>
            )}

            {(contexts.length > 1 || !selected) && (
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.contextRow}
              >
                {contexts.map((context) => {
                  const key = contextKey(context);
                  const selectedContext = key === selectedContextKey;
                  return (
                    <Pressable
                      key={key}
                      accessibilityRole="button"
                      accessibilityState={{ selected: selectedContext, disabled: saving }}
                      disabled={saving}
                      accessibilityLabel={`${context.petName}, ${context.householdName}`}
                      style={[styles.contextChip, selectedContext && styles.contextChipSelected]}
                      onPress={() => chooseContext(key)}
                    >
                      <Ionicons
                        name="paw-outline"
                        size={18}
                        color={selectedContext ? colors.primary[800] : colors.gray[700]}
                      />
                      <View style={styles.contextCopy}>
                        <Text style={styles.contextPet}>{context.petName}</Text>
                        <Text style={styles.contextHousehold}>{context.householdName}</Text>
                      </View>
                    </Pressable>
                  );
                })}
              </ScrollView>
            )}

            {!selected && contexts.length > 1 && (
              <Text style={styles.contextHint}>
                Woof found more than one authorized context, so it will not guess which household
                you mean.
              </Text>
            )}

            {selected && !selected.timezone && (
              <Text style={styles.timezoneWarning}>
                This household needs a timezone before Daily Signals can be recorded.
              </Text>
            )}
          </View>

          {selected && (
            <>
              <View style={styles.signalsSection}>
                <View style={styles.signalHeadingRow}>
                  <View style={styles.flex}>
                    <Text style={styles.sectionTitle}>What did you notice?</Text>
                    <Text style={styles.sectionSubtitle}>
                      Tap only the dimensions that mattered. Untouched means not reported.
                    </Text>
                  </View>
                  {answeredCount > 0 && (
                    <Text style={styles.observationCount}>
                      {answeredCount} {answeredCount === 1 ? 'observation' : 'observations'}
                    </Text>
                  )}
                </View>

                <View style={styles.signalList}>
                  {dimensions.map((dimension) => {
                    const selectedValue = answers[dimension.key];
                    const expanded = expandedDimension === dimension.key;
                    const stateLabel = selectedValue
                      ? (choiceLabel.get(selectedValue) ?? selectedValue)
                      : 'Not reported';

                    return (
                      <View
                        key={dimension.key}
                        style={[styles.signalRow, expanded && styles.signalRowExpanded]}
                      >
                        <Pressable
                          accessibilityRole="button"
                          accessibilityState={{ expanded }}
                          accessibilityLabel={`${dimension.label}. ${stateLabel}.`}
                          accessibilityHint={
                            expanded ? 'Closes choices' : 'Opens relative observation choices'
                          }
                          style={styles.signalRowHeader}
                          onPress={() =>
                            setExpandedDimension((current) =>
                              current === dimension.key ? null : dimension.key
                            )
                          }
                        >
                          <View style={styles.flex}>
                            <Text style={styles.signalLabel}>{dimension.label}</Text>
                            <Text
                              style={[
                                styles.signalState,
                                selectedValue && styles.signalStateSelected,
                              ]}
                            >
                              {stateLabel}
                            </Text>
                          </View>
                          <Ionicons
                            name={expanded ? 'chevron-up' : 'chevron-down'}
                            size={20}
                            color={colors.gray[500]}
                          />
                        </Pressable>

                        {expanded && (
                          <View style={styles.signalChoices}>
                            <Text style={styles.signalPrompt}>{dimension.prompt}</Text>
                            <View style={styles.choiceRow}>
                              {choices.map((choice) => {
                                const isSelected = selectedValue === choice.value;
                                return (
                                  <Pressable
                                    key={choice.value}
                                    accessibilityRole="button"
                                    accessibilityState={{
                                      selected: isSelected,
                                      disabled: !stateReady || !editing || submissionLocked,
                                    }}
                                    style={[styles.choice, isSelected && styles.choiceSelected]}
                                    disabled={!stateReady || !editing || submissionLocked}
                                    onPress={() => setChoice(dimension.key, choice.value)}
                                  >
                                    <Text
                                      style={[
                                        styles.choiceText,
                                        isSelected && styles.choiceTextSelected,
                                      ]}
                                    >
                                      {choice.label}
                                    </Text>
                                  </Pressable>
                                );
                              })}
                            </View>
                            {selectedValue && (
                              <Pressable
                                accessibilityRole="button"
                                style={styles.clearButton}
                                disabled={!stateReady || !editing || submissionLocked}
                                onPress={() => clearChoice(dimension.key)}
                              >
                                <Text style={styles.clearButtonText}>Leave unreported</Text>
                              </Pressable>
                            )}
                          </View>
                        )}
                      </View>
                    );
                  })}
                </View>
              </View>

              {!current && (
                <View style={styles.noteSection}>
                  <Pressable
                    accessibilityRole="button"
                    accessibilityState={{ expanded: noteOpen }}
                    style={styles.noteToggle}
                    onPress={() => setNoteOpen((current) => !current)}
                  >
                    <View style={styles.flex}>
                      <Text style={styles.noteToggleTitle}>Add a private note</Text>
                      <Text style={styles.noteToggleSubtitle}>Optional · up to 500 characters</Text>
                    </View>
                    <Ionicons
                      name={noteOpen ? 'chevron-up' : 'chevron-down'}
                      size={20}
                      color={colors.gray[500]}
                    />
                  </Pressable>

                  {noteOpen && (
                    <>
                      <TextInput
                        editable={stateReady && editing && !submissionLocked}
                        value={note}
                        onChangeText={(value) => setNote(value.slice(0, 500))}
                        placeholder="A short private note"
                        placeholderTextColor={colors.gray[400]}
                        multiline
                        textAlignVertical="top"
                        style={styles.noteInput}
                        accessibilityLabel="Optional private Daily Signals note"
                      />
                      <Text style={styles.characterCount}>{note.length}/500</Text>
                    </>
                  )}
                </View>
              )}

              {error && (
                <View style={styles.errorCard} accessibilityRole="alert">
                  <Ionicons name="alert-circle-outline" size={20} color={colors.error.dark} />
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              {success && (
                <View style={styles.successCard} accessibilityRole="summary">
                  <Ionicons name="checkmark-circle-outline" size={22} color={colors.success.dark} />
                  <View style={styles.successCopy}>
                    <Text style={styles.successTitle}>
                      Saved {success.savedCount}{' '}
                      {success.savedCount === 1 ? 'observation' : 'observations'} for{' '}
                      {success.petName}.
                    </Text>
                    {success.receipt.duplicate && (
                      <Text style={styles.successText}>
                        This matched today&apos;s saved check-in, so Woof did not count it twice.
                      </Text>
                    )}
                    {success.unsureCount > 0 && (
                      <Text style={styles.successText}>
                        “Not sure” stays uncertainty and does not become a baseline value.
                      </Text>
                    )}
                    <Text style={styles.successText}>
                      These observations stay private context, not a health score.
                    </Text>
                  </View>
                </View>
              )}

              <Pressable
                accessibilityRole="button"
                accessibilityState={{
                  disabled:
                    saving ||
                    !stateReady ||
                    !editing ||
                    (!current && answeredCount === 0) ||
                    !selected.timezone,
                }}
                disabled={
                  saving ||
                  !stateReady ||
                  !editing ||
                  (!current && answeredCount === 0) ||
                  !selected.timezone
                }
                style={[
                  styles.saveButton,
                  (saving ||
                    !stateReady ||
                    !editing ||
                    (!current && answeredCount === 0) ||
                    !selected.timezone) &&
                    styles.disabledButton,
                ]}
                onPress={() => void save()}
              >
                {saving ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.saveButtonText}>
                    {submissionLocked ? 'Retry saving' : current ? 'Save correction:' : 'Save'}{' '}
                    {answeredCount || ''} {answeredCount === 1 ? 'observation' : 'observations'}
                  </Text>
                )}
              </Pressable>

              {onDone && (
                <Pressable accessibilityRole="button" style={styles.nothingButton} onPress={onDone}>
                  <Text style={styles.nothingButtonText}>
                    {answeredCount === 0 ? 'Nothing to add today' : 'Leave without saving'}
                  </Text>
                </Pressable>
              )}

              <Text style={styles.privacyNote}>
                Unreported stays unknown. “Not sure” stays uncertainty. A saved check-in is
                canonical for that household-local day and is not silently overwritten by a later
                different answer.
              </Text>
            </>
          )}
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 90 },
  flex: { flex: 1 },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.5 },
  title: {
    marginTop: 3,
    color: colors.text.primary,
    fontSize: 31,
    lineHeight: 38,
    fontWeight: '800',
  },
  subtitle: { marginTop: 8, color: colors.text.secondary, fontSize: 14, lineHeight: 21 },
  boundaryText: { marginTop: 5, color: colors.text.secondary, fontSize: 12, lineHeight: 18 },
  contextSection: {
    marginTop: 20,
    paddingVertical: 14,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: colors.gray[200],
  },
  sectionEyebrow: {
    color: colors.text.secondary,
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  contextSummary: { marginTop: 3, color: colors.text.primary, fontSize: 16, fontWeight: '800' },
  contextRow: { gap: 8, paddingTop: 10, paddingRight: 18 },
  contextChip: {
    minWidth: 148,
    minHeight: 60,
    paddingHorizontal: 13,
    paddingVertical: 10,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.gray[300],
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#ffffff',
  },
  contextChipSelected: { borderColor: colors.primary[500], backgroundColor: colors.primary[50] },
  contextCopy: { flexShrink: 1 },
  contextPet: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  contextHousehold: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  contextHint: { marginTop: 9, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
  timezoneWarning: { marginTop: 10, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  signalsSection: { marginTop: 24 },
  signalHeadingRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  sectionTitle: { color: colors.text.primary, fontSize: 18, fontWeight: '800' },
  sectionSubtitle: { marginTop: 4, color: colors.text.secondary, fontSize: 12, lineHeight: 17 },
  observationCount: {
    maxWidth: 120,
    color: colors.primary[700],
    fontSize: 11,
    lineHeight: 16,
    fontWeight: '800',
    textAlign: 'right',
  },
  signalList: {
    marginTop: 12,
    borderWidth: 1,
    borderColor: colors.gray[200],
    borderRadius: 18,
    overflow: 'hidden',
    backgroundColor: '#ffffff',
  },
  signalRow: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.gray[200],
  },
  signalRowExpanded: { backgroundColor: colors.gray[50] },
  signalRowHeader: {
    minHeight: 58,
    paddingHorizontal: 14,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  signalLabel: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  signalState: { marginTop: 2, color: colors.text.secondary, fontSize: 11, lineHeight: 16 },
  signalStateSelected: { color: colors.primary[700], fontWeight: '700' },
  signalChoices: {
    paddingHorizontal: 14,
    paddingBottom: 14,
  },
  signalPrompt: { color: colors.text.secondary, fontSize: 12, lineHeight: 18 },
  choiceRow: { marginTop: 10, flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  choice: {
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: 13,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
  },
  choiceSelected: { borderColor: colors.primary[500], backgroundColor: colors.primary[50] },
  choiceText: { color: colors.gray[700], fontSize: 12, fontWeight: '700' },
  choiceTextSelected: { color: colors.primary[800] },
  clearButton: {
    minHeight: 44,
    alignSelf: 'flex-start',
    justifyContent: 'center',
    marginTop: 5,
    paddingHorizontal: 4,
  },
  clearButtonText: { color: colors.text.secondary, fontSize: 12, fontWeight: '700' },
  noteSection: {
    marginTop: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: colors.gray[200],
    borderRadius: 16,
    backgroundColor: '#ffffff',
  },
  noteToggle: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  noteToggleTitle: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  noteToggleSubtitle: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  noteInput: {
    marginTop: 10,
    minHeight: 100,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: colors.gray[50],
    color: colors.text.primary,
    fontSize: 14,
  },
  characterCount: { marginTop: 5, textAlign: 'right', color: colors.text.secondary, fontSize: 11 },
  noticeCard: {
    marginTop: 20,
    padding: 18,
    borderRadius: 20,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
    gap: 7,
  },
  noticeTitle: { color: colors.text.primary, fontSize: 16, fontWeight: '800' },
  noticeText: { color: colors.text.secondary, fontSize: 13, lineHeight: 19 },
  errorCard: {
    marginTop: 18,
    padding: 14,
    borderRadius: 14,
    flexDirection: 'row',
    gap: 8,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 18 },
  successCard: {
    marginTop: 18,
    padding: 15,
    borderRadius: 16,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: colors.success.light,
  },
  successCopy: { flex: 1, gap: 3 },
  successTitle: { color: colors.success.dark, fontSize: 14, fontWeight: '800', lineHeight: 20 },
  successText: { color: colors.success.dark, fontSize: 11, lineHeight: 17 },
  saveButton: {
    marginTop: 18,
    minHeight: 50,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    paddingHorizontal: 16,
  },
  disabledButton: { opacity: 0.45 },
  saveButtonText: { color: '#ffffff', fontSize: 15, fontWeight: '800', textAlign: 'center' },
  nothingButton: {
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 5,
    paddingHorizontal: 12,
  },
  nothingButtonText: { color: colors.text.secondary, fontSize: 13, fontWeight: '700' },
  privacyNote: {
    marginTop: 6,
    color: colors.text.secondary,
    fontSize: 11,
    lineHeight: 17,
    textAlign: 'center',
  },
});
