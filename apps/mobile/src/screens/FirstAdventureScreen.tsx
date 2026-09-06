import React, { useEffect, useState } from 'react';
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
import { isAxiosError } from 'axios';
import { adaptiveProfileApi } from '../api/adaptive-profile';
import { companionApi, type CompanionMode, type CompanionState } from '../api/companion';
import { petsApi, type CreatedOwnedPet } from '../api/pets';
import { useAuth } from '../contexts/AuthContext';
import {
  buildFirstAdventureResponses,
  emptyFirstAdventureSelections,
  type FirstAdventureEffort,
  type FirstAdventureGoal,
  type FirstAdventureSelections,
  type FirstAdventureSocialComfort,
  type FirstAdventureTimeBudget,
} from '../onboarding/first-adventure';
import {
  clearPetCreationRecovery,
  getOrCreatePetCreationRecovery,
  markPetCreationAmbiguous,
  readPetCreationRecovery,
} from '../onboarding/recovery';
import { colors } from '../theme/tokens';

type Phase = 'pet' | 'goals' | 'capacity' | 'social';

const goalChoices: { value: FirstAdventureGoal; label: string }[] = [
  { value: 'MORE_ADVENTURES', label: 'More adventures' },
  { value: 'TRAINING', label: 'Training together' },
  { value: 'CALMER_ROUTINES', label: 'Calmer routines' },
  { value: 'SOCIAL_CONFIDENCE', label: 'Social confidence' },
  { value: 'CARE_ROUTINES', label: 'Care routines' },
  { value: 'JUST_HAVE_FUN', label: 'Just have fun' },
];

const timeChoices: { value: FirstAdventureTimeBudget; label: string }[] = [
  { value: 'FIVE_MIN', label: 'About 5 min' },
  { value: 'TEN_TO_FIFTEEN', label: '10–15 min' },
  { value: 'TWENTY_TO_THIRTY', label: '20–30 min' },
  { value: 'FORTY_PLUS', label: '40+ min' },
  { value: 'VARIES', label: 'It varies' },
];

const effortChoices: { value: FirstAdventureEffort; label: string }[] = [
  { value: 'KEEP_IT_EASY', label: 'Keep it easy' },
  { value: 'MODERATE', label: 'Moderate' },
  { value: 'UP_FOR_A_CHALLENGE', label: 'Up for a challenge' },
  { value: 'VARIES', label: 'It varies' },
];

const socialChoices: { value: FirstAdventureSocialComfort; label: string }[] = [
  { value: 'PREFERS_SPACE', label: 'Usually prefers space' },
  { value: 'CALM_AT_DISTANCE', label: 'Comfortable at a distance' },
  { value: 'SELECTIVELY_SOCIAL', label: 'Selectively social' },
  { value: 'OFTEN_SOCIAL', label: 'Often social' },
  { value: 'NOT_SURE', label: 'Not sure yet' },
];

export default function FirstAdventureScreen({
  onComplete,
  onModeResolved,
  onRecheck,
}: {
  onComplete: () => void;
  onModeResolved: (state: CompanionState) => void;
  onRecheck: () => void;
}) {
  const { user } = useAuth();
  const [phase, setPhase] = useState<Phase>('pet');
  const [name, setName] = useState('');
  const [breed, setBreed] = useState('');
  const [pet, setPet] = useState<CreatedOwnedPet | null>(null);
  const [selections, setSelections] = useState<FirstAdventureSelections>(
    emptyFirstAdventureSelections
  );
  const [creating, setCreating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [switchingMode, setSwitchingMode] = useState<CompanionMode | null>(null);
  const [ambiguousCreate, setAmbiguousCreate] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user?.id) return;
    void readPetCreationRecovery(user.id).then((recovery) => {
      if (!recovery) return;
      setName(recovery.name);
      setBreed(recovery.breed ?? '');
      setAmbiguousCreate(recovery.ambiguous === true);
      if (recovery.ambiguous) {
        setError(
          'Woof still has an unresolved pet-creation attempt. Retry the exact create or check server state before changing anything.'
        );
      }
    });
  }, [user?.id]);

  const createDog = async () => {
    if (!user?.id || !name.trim()) {
      setError('Add your dog’s name first. Breed is optional.');
      return;
    }

    setCreating(true);
    setError(null);
    try {
      const recovery = await getOrCreatePetCreationRecovery(user.id, name, breed);
      const created = await petsApi.createDog({
        name: recovery.name,
        species: 'DOG',
        breed: recovery.breed,
        creationKey: recovery.creationKey,
      });
      setPet(created);
      setAmbiguousCreate(false);
      await clearPetCreationRecovery();
      setPhase('goals');
    } catch (caught) {
      const status = isAxiosError(caught) ? caught.response?.status : undefined;
      const ambiguous = status === undefined || status >= 500;
      setAmbiguousCreate(ambiguous);
      await markPetCreationAmbiguous(ambiguous);
      setError(
        ambiguous
          ? 'Woof could not confirm whether that create reached the server. These exact details are now frozen: retry the same create or check server state before doing anything else.'
          : 'Woof rejected those pet details. Nothing new was created. Check the fields and try again.'
      );
    } finally {
      setCreating(false);
    }
  };

  const changeMode = async (mode: CompanionMode) => {
    if (ambiguousCreate) {
      setError(
        'Resolve the uncertain pet-creation attempt first. Woof will not change account mode while a durable write may already exist.'
      );
      return;
    }

    setSwitchingMode(mode);
    setError(null);
    try {
      const state = await companionApi.updateMode(mode);
      onModeResolved(state);
    } catch {
      setError(
        'Woof could not change your starting role. Pet access and relationship state did not change.'
      );
    } finally {
      setSwitchingMode(null);
    }
  };

  const toggleGoal = (goal: FirstAdventureGoal) => {
    setSelections((current) => {
      const selected = current.goals.includes(goal);
      if (selected) return { ...current, goals: current.goals.filter((value) => value !== goal) };
      if (current.goals.length >= 3) return current;
      return { ...current, goals: [...current.goals, goal] };
    });
  };

  const finish = async (skipAll = false) => {
    if (!pet) return;
    setSaving(true);
    setError(null);
    const householdId = pet.householdMemberships[0]?.householdId;

    if (householdId) {
      const responses = buildFirstAdventureResponses(pet.id, selections, skipAll);
      // These answers are optional evidence, never a gate. A partial/network
      // failure cannot undo the durable pet relationship or keep the pair out
      // of Today. Deterministic response IDs make safe retry possible later.
      await Promise.allSettled(
        responses.map((response) =>
          adaptiveProfileApi.recordQuestionResponse(householdId, pet.id, response)
        )
      );
    }

    setSaving(false);
    onComplete();
  };

  const header = (
    <>
      <View style={styles.brandMark}>
        <Ionicons name="paw" size={23} color="#ffffff" />
      </View>
      <Text style={styles.eyebrow}>FIRST ADVENTURE</Text>
    </>
  );

  if (phase === 'pet') {
    const modeSwitchDisabled = switchingMode !== null || ambiguousCreate;

    return (
      <ScrollView
        style={styles.screen}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        {header}
        <Text style={styles.title}>Start with the dog you actually care for.</Text>
        <Text style={styles.intro}>
          A name is enough to create the relationship. Breed is optional. Photos, temperament, and
          extra profile work can wait until they are useful.
        </Text>

        <View style={styles.formCard}>
          <Text style={styles.label}>Dog’s name</Text>
          <TextInput
            accessibilityLabel="Dog name"
            style={styles.input}
            value={name}
            onChangeText={setName}
            editable={!creating && !ambiguousCreate}
            placeholder="Shasta"
            placeholderTextColor={colors.gray[400]}
            autoCapitalize="words"
            maxLength={80}
          />
          <Text style={styles.label}>Breed, if useful</Text>
          <TextInput
            accessibilityLabel="Dog breed optional"
            style={styles.input}
            value={breed}
            onChangeText={setBreed}
            editable={!creating && !ambiguousCreate}
            placeholder="Optional"
            placeholderTextColor={colors.gray[400]}
            autoCapitalize="words"
            maxLength={120}
          />

          <Pressable
            accessibilityRole="button"
            disabled={creating || !name.trim()}
            style={[styles.primaryButton, (creating || !name.trim()) && styles.disabled]}
            onPress={() => void createDog()}
          >
            {creating ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.primaryButtonText}>
                {ambiguousCreate ? 'Retry exact create' : 'Create our pair'}
              </Text>
            )}
          </Pressable>

          {ambiguousCreate && (
            <View style={styles.airlockCard}>
              <Ionicons name="lock-closed-outline" size={19} color={colors.warning.dark} />
              <View style={styles.airlockCopy}>
                <Text style={styles.airlockTitle}>Uncertain write in progress</Text>
                <Text style={styles.airlockText}>
                  Name, breed, and replay identity stay frozen until Woof resolves whether the
                  server created this dog.
                </Text>
              </View>
            </View>
          )}

          {ambiguousCreate && (
            <Pressable accessibilityRole="button" style={styles.outlineButton} onPress={onRecheck}>
              <Text style={styles.outlineButtonText}>Check server state first</Text>
            </Pressable>
          )}
        </View>

        <View style={styles.truthCard}>
          <Ionicons name="shield-checkmark-outline" size={20} color={colors.success.dark} />
          <Text style={styles.truthText}>
            Choosing Pet Guardian does not create pet access. This server-authorized pet creation is
            the relationship that opens pet-specific dogOS.
          </Text>
        </View>

        <Text style={styles.altTitle}>No dog to add right now?</Text>
        <View style={styles.altRow}>
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ disabled: modeSwitchDisabled }}
            disabled={modeSwitchDisabled}
            style={[styles.altButton, modeSwitchDisabled && styles.disabled]}
            onPress={() => void changeMode('ANIMAL_ALLY')}
          >
            <Text style={styles.altButtonText}>I’m here to learn</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ disabled: modeSwitchDisabled }}
            disabled={modeSwitchDisabled}
            style={[styles.altButton, modeSwitchDisabled && styles.disabled]}
            onPress={() => void changeMode('FOSTER_CAREGIVER')}
          >
            <Text style={styles.altButtonText}>I foster / support</Text>
          </Pressable>
        </View>

        {error && (
          <Text style={styles.errorText} accessibilityRole="alert">
            {error}
          </Text>
        )}
      </ScrollView>
    );
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      {header}
      <Text style={styles.progress}>
        OPTIONAL CONTEXT · {phase === 'goals' ? '1' : phase === 'capacity' ? '2' : '3'} OF 3
      </Text>

      {phase === 'goals' && (
        <>
          <Text style={styles.title}>What would feel useful together?</Text>
          <Text style={styles.intro}>
            Choose up to three, or skip. This helps break ties between otherwise safe suggestions.
          </Text>
          <View style={styles.chipWrap}>
            {goalChoices.map((choice) => {
              const selected = selections.goals.includes(choice.value);
              return (
                <Pressable
                  key={choice.value}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  style={[styles.chip, selected && styles.chipSelected]}
                  onPress={() => toggleGoal(choice.value)}
                >
                  <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
                    {choice.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
          <Pressable
            style={styles.primaryButton}
            onPress={() => setPhase('capacity')}
            accessibilityRole="button"
          >
            <Text style={styles.primaryButtonText}>Continue</Text>
          </Pressable>
        </>
      )}

      {phase === 'capacity' && (
        <>
          <Text style={styles.title}>What fits a real day?</Text>
          <Text style={styles.intro}>
            No aspirational homework. Tell Woof what is realistically easy to fit, or leave it
            unknown.
          </Text>
          <Text style={styles.question}>Time that often fits</Text>
          <View style={styles.chipWrap}>
            {timeChoices.map((choice) => (
              <ChoiceChip
                key={choice.value}
                label={choice.label}
                selected={selections.timeBudget === choice.value}
                onPress={() =>
                  setSelections((current) => ({ ...current, timeBudget: choice.value }))
                }
              />
            ))}
          </View>
          <Text style={styles.question}>Effort that usually feels fair</Text>
          <View style={styles.chipWrap}>
            {effortChoices.map((choice) => (
              <ChoiceChip
                key={choice.value}
                label={choice.label}
                selected={selections.effort === choice.value}
                onPress={() => setSelections((current) => ({ ...current, effort: choice.value }))}
              />
            ))}
          </View>
          <Pressable
            style={styles.primaryButton}
            onPress={() => setPhase('social')}
            accessibilityRole="button"
          >
            <Text style={styles.primaryButtonText}>Continue</Text>
          </Pressable>
        </>
      )}

      {phase === 'social' && (
        <>
          <Text style={styles.title}>
            How does {pet?.name ?? 'your dog'} usually feel around unfamiliar dogs?
          </Text>
          <Text style={styles.intro}>
            This is a starting observation, not a personality label. “Not sure” is useful
            information too.
          </Text>
          <View style={styles.choiceColumn}>
            {socialChoices.map((choice) => (
              <ChoiceChip
                key={choice.value}
                label={choice.label}
                selected={selections.socialComfort === choice.value}
                onPress={() =>
                  setSelections((current) => ({ ...current, socialComfort: choice.value }))
                }
                wide
              />
            ))}
          </View>
          <Pressable
            accessibilityRole="button"
            disabled={saving}
            style={[styles.primaryButton, saving && styles.disabled]}
            onPress={() => void finish(false)}
          >
            {saving ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.primaryButtonText}>Open Today</Text>
            )}
          </Pressable>
        </>
      )}

      <Pressable
        accessibilityRole="button"
        disabled={saving}
        style={styles.skipButton}
        onPress={() => void finish(true)}
      >
        <Text style={styles.skipText}>Skip personalization and open Today</Text>
      </Pressable>
      <Text style={styles.permissionText}>
        Skipping never reduces access, rewards, or relationship status.
      </Text>
      {error && (
        <Text style={styles.errorText} accessibilityRole="alert">
          {error}
        </Text>
      )}
    </ScrollView>
  );
}

function ChoiceChip({
  label,
  selected,
  onPress,
  wide = false,
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
  wide?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected }}
      style={[styles.chip, wide && styles.chipWide, selected && styles.chipSelected]}
      onPress={onPress}
    >
      <Text style={[styles.chipText, selected && styles.chipTextSelected]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.default },
  content: { paddingHorizontal: 24, paddingTop: 64, paddingBottom: 42 },
  brandMark: {
    width: 46,
    height: 46,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginBottom: 24,
  },
  eyebrow: {
    color: colors.primary[700],
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.4,
    marginBottom: 8,
  },
  progress: {
    color: colors.gray[500],
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.1,
    marginBottom: 14,
  },
  title: { color: colors.gray[900], fontSize: 31, lineHeight: 38, fontWeight: '800' },
  intro: { color: colors.gray[600], fontSize: 15, lineHeight: 23, marginTop: 12 },
  formCard: {
    marginTop: 26,
    borderWidth: 1,
    borderColor: colors.gray[200],
    borderRadius: 20,
    padding: 18,
    backgroundColor: '#ffffff',
  },
  label: { color: colors.gray[800], fontSize: 13, fontWeight: '700', marginBottom: 8 },
  input: {
    minHeight: 52,
    borderWidth: 1,
    borderColor: colors.gray[300],
    borderRadius: 14,
    paddingHorizontal: 15,
    color: colors.gray[900],
    fontSize: 16,
    marginBottom: 17,
    backgroundColor: colors.gray[50],
  },
  primaryButton: {
    minHeight: 52,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginTop: 24,
    paddingHorizontal: 18,
  },
  primaryButtonText: { color: '#ffffff', fontSize: 15, fontWeight: '800' },
  disabled: { opacity: 0.5 },
  airlockCard: {
    flexDirection: 'row',
    gap: 10,
    borderRadius: 14,
    padding: 13,
    marginTop: 12,
    backgroundColor: colors.warning.light,
  },
  airlockCopy: { flex: 1 },
  airlockTitle: { color: colors.warning.dark, fontSize: 13, fontWeight: '800' },
  airlockText: { color: colors.warning.dark, fontSize: 12, lineHeight: 18, marginTop: 3 },
  outlineButton: {
    minHeight: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.primary[300],
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  outlineButtonText: { color: colors.primary[700], fontSize: 14, fontWeight: '700' },
  truthCard: {
    flexDirection: 'row',
    gap: 10,
    padding: 15,
    borderRadius: 16,
    backgroundColor: colors.success.light,
    marginTop: 16,
  },
  truthText: { flex: 1, color: colors.success.dark, fontSize: 12, lineHeight: 18 },
  altTitle: {
    color: colors.gray[800],
    fontSize: 14,
    fontWeight: '800',
    marginTop: 28,
    marginBottom: 10,
  },
  altRow: { flexDirection: 'row', gap: 10 },
  altButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.gray[200],
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  altButtonText: {
    color: colors.gray[700],
    fontSize: 12,
    fontWeight: '700',
    textAlign: 'center',
  },
  errorText: {
    color: colors.error.dark,
    backgroundColor: colors.error.light,
    borderRadius: 12,
    padding: 12,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 16,
  },
  chipWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 9, marginTop: 22 },
  chip: {
    minHeight: 44,
    justifyContent: 'center',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
    paddingHorizontal: 15,
    paddingVertical: 10,
  },
  chipWide: { width: '100%', borderRadius: 14 },
  chipSelected: { borderColor: colors.primary[500], backgroundColor: colors.primary[50] },
  chipText: { color: colors.gray[700], fontSize: 13, fontWeight: '600' },
  chipTextSelected: { color: colors.primary[800], fontWeight: '800' },
  question: { color: colors.gray[800], fontSize: 14, fontWeight: '800', marginTop: 26 },
  choiceColumn: { gap: 9, marginTop: 22 },
  skipButton: {
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
    paddingHorizontal: 12,
  },
  skipText: { color: colors.gray[600], fontSize: 13, fontWeight: '700', textAlign: 'center' },
  permissionText: {
    color: colors.gray[500],
    fontSize: 11,
    lineHeight: 17,
    textAlign: 'center',
    marginTop: 4,
  },
});
