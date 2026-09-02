import React, { useCallback, useMemo, useState } from 'react';
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
import { householdsApi, type HouseholdSnapshot } from '../api/households';
import {
  intelligenceApi,
  type DailySignalChoice,
  type DailySignalsAnswers,
} from '../api/intelligence';
import { colors } from '../theme/tokens';

const dimensions: Array<{
  key: keyof DailySignalsAnswers;
  label: string;
  prompt: string;
}> = [
  { key: 'appetite', label: 'Appetite', prompt: 'How was eating compared with usual?' },
  { key: 'energy', label: 'Energy', prompt: 'How was energy compared with usual?' },
  { key: 'bathroomRoutine', label: 'Bathroom / routine', prompt: 'Did bathroom habits or routine feel different?' },
  { key: 'mobilityComfort', label: 'Mobility / comfort', prompt: 'How comfortable did movement seem?' },
  { key: 'engagementSocialComfort', label: 'Engagement', prompt: 'How engaged or socially comfortable did they seem?' },
  { key: 'sleepRest', label: 'Sleep / rest', prompt: 'How restful did rest seem?' },
];

const choices: Array<{ value: DailySignalChoice; label: string }> = [
  { value: 'LESS', label: 'Less' },
  { value: 'USUAL', label: 'Usual' },
  { value: 'MORE', label: 'More' },
  { value: 'UNSURE', label: 'Not sure' },
];

type HouseholdPetContext = {
  householdId: string;
  householdName: string;
  timezone?: string | null;
  petId: string;
  petName: string;
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
      })),
  );
}

export default function DailySignalsScreen() {
  const [contexts, setContexts] = useState<HouseholdPetContext[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [answers, setAnswers] = useState<DailySignalsAnswers>({});
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const households = await householdsApi.getMine();
      const nextContexts = contextsFromHouseholds(households);
      setContexts(nextContexts);
      setSelectedIndex((current) => Math.min(current, Math.max(0, nextContexts.length - 1)));
      setError(null);
    } catch {
      setError('Woof could not load your household context. No check-in was recorded.');
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const selected = contexts[selectedIndex] ?? null;
  const answeredCount = useMemo(
    () => Object.values(answers).filter((value) => value !== undefined).length,
    [answers],
  );

  const setChoice = (key: keyof DailySignalsAnswers, value: DailySignalChoice) => {
    setAnswers((current) => ({ ...current, [key]: current[key] === value ? undefined : value }));
    setSuccess(null);
  };

  const save = async () => {
    if (!selected || answeredCount === 0) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const receipt = await intelligenceApi.captureDailySignals({
        householdId: selected.householdId,
        petId: selected.petId,
        signals: answers,
        ...(note.trim() ? { note: note.trim() } : {}),
      });
      const duplicateCopy = receipt.duplicate ? ' This was the same check-in Woof already had for today.' : '';
      setSuccess(`Saved for ${selected.petName} on ${receipt.localDate}.${duplicateCopy}`);
      setAnswers({});
      setNote('');
    } catch (caught: any) {
      const status = caught?.response?.status;
      if (status === 409) {
        setError('A different Daily Signals check-in is already recorded for this dog and household day. Woof will not silently overwrite it.');
      } else {
        setError('Woof could not save this check-in. Nothing was partially replaced.');
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading && contexts.length === 0) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Loading household context…</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <Text style={styles.eyebrow}>PRIVATE CHECK-IN</Text>
      <Text style={styles.title}>Daily Signals</Text>
      <Text style={styles.subtitle}>
        A few owner-observable signals can help Woof learn what is normal for this individual dog. This is not a diagnosis or health score.
      </Text>

      {contexts.length === 0 ? (
        <View style={styles.noticeCard}>
          <Ionicons name="paw-outline" size={22} color={colors.primary[700]} />
          <Text style={styles.noticeTitle}>No active household pet context yet</Text>
          <Text style={styles.noticeText}>Create or join an authorized pet household before recording Daily Signals.</Text>
        </View>
      ) : (
        <>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Who is this for?</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.contextRow}>
              {contexts.map((context, index) => {
                const selectedContext = index === selectedIndex;
                return (
                  <Pressable
                    key={`${context.householdId}:${context.petId}`}
                    accessibilityRole="button"
                    accessibilityState={{ selected: selectedContext }}
                    style={[styles.contextChip, selectedContext && styles.contextChipSelected]}
                    onPress={() => {
                      setSelectedIndex(index);
                      setAnswers({});
                      setNote('');
                      setSuccess(null);
                    }}
                  >
                    <Ionicons name="paw-outline" size={18} color={selectedContext ? colors.primary[800] : colors.gray[700]} />
                    <View>
                      <Text style={styles.contextPet}>{context.petName}</Text>
                      <Text style={styles.contextHousehold}>{context.householdName}</Text>
                    </View>
                  </Pressable>
                );
              })}
            </ScrollView>
            {selected && !selected.timezone && (
              <Text style={styles.timezoneWarning}>This household needs a timezone before Daily Signals can be recorded.</Text>
            )}
          </View>

          {dimensions.map((dimension) => (
            <View key={dimension.key} style={styles.dimensionCard}>
              <Text style={styles.dimensionLabel}>{dimension.label}</Text>
              <Text style={styles.dimensionPrompt}>{dimension.prompt}</Text>
              <View style={styles.choiceRow}>
                {choices.map((choice) => {
                  const isSelected = answers[dimension.key] === choice.value;
                  return (
                    <Pressable
                      key={choice.value}
                      accessibilityRole="button"
                      accessibilityState={{ selected: isSelected }}
                      style={[styles.choice, isSelected && styles.choiceSelected]}
                      onPress={() => setChoice(dimension.key, choice.value)}
                    >
                      <Text style={[styles.choiceText, isSelected && styles.choiceTextSelected]}>{choice.label}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>
          ))}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Something else you noticed?</Text>
            <Text style={styles.sectionSubtitle}>Optional, private, and limited to 500 characters.</Text>
            <TextInput
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
          </View>

          {error && (
            <View style={styles.errorCard} accessibilityRole="alert">
              <Ionicons name="alert-circle-outline" size={20} color={colors.error.dark} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {success && (
            <View style={styles.successCard} accessibilityRole="summary">
              <Ionicons name="checkmark-circle-outline" size={20} color={colors.success.dark} />
              <Text style={styles.successText}>{success}</Text>
            </View>
          )}

          <Pressable
            accessibilityRole="button"
            accessibilityState={{ disabled: saving || answeredCount === 0 || !selected?.timezone }}
            disabled={saving || answeredCount === 0 || !selected?.timezone}
            style={[
              styles.saveButton,
              (saving || answeredCount === 0 || !selected?.timezone) && styles.disabledButton,
            ]}
            onPress={() => void save()}
          >
            {saving ? <ActivityIndicator color="#ffffff" /> : <Text style={styles.saveButtonText}>Save check-in</Text>}
          </Pressable>

          <Text style={styles.privacyNote}>
            Skipping is allowed. “Not sure” remains missing evidence. A saved check-in is canonical for that household-local day and is not silently overwritten by a later different answer.
          </Text>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 80 },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.5 },
  title: { marginTop: 3, color: colors.text.primary, fontSize: 32, fontWeight: '800' },
  subtitle: { marginTop: 8, color: colors.text.secondary, fontSize: 14, lineHeight: 21 },
  section: { marginTop: 24 },
  sectionTitle: { color: colors.text.primary, fontSize: 17, fontWeight: '800' },
  sectionSubtitle: { marginTop: 4, color: colors.text.secondary, fontSize: 12, lineHeight: 17 },
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
  contextPet: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  contextHousehold: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  timezoneWarning: { marginTop: 10, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  dimensionCard: {
    marginTop: 12,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  dimensionLabel: { color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  dimensionPrompt: { marginTop: 3, color: colors.text.secondary, fontSize: 12, lineHeight: 17 },
  choiceRow: { marginTop: 12, flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  choice: {
    minHeight: 42,
    justifyContent: 'center',
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
  },
  choiceSelected: { borderColor: colors.primary[500], backgroundColor: colors.primary[50] },
  choiceText: { color: colors.gray[700], fontSize: 12, fontWeight: '600' },
  choiceTextSelected: { color: colors.primary[800] },
  noteInput: {
    marginTop: 10,
    minHeight: 110,
    padding: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
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
    padding: 14,
    borderRadius: 14,
    flexDirection: 'row',
    gap: 8,
    backgroundColor: colors.success.light,
  },
  successText: { flex: 1, color: colors.success.dark, fontSize: 12, lineHeight: 18 },
  saveButton: {
    marginTop: 18,
    minHeight: 50,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
  },
  disabledButton: { opacity: 0.45 },
  saveButtonText: { color: '#ffffff', fontSize: 15, fontWeight: '800' },
  privacyNote: { marginTop: 12, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
});
