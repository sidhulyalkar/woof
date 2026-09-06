import React, { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { companionApi, type CompanionMode, type CompanionState } from '../api/companion';
import { colors } from '../theme/tokens';

const choices: Array<{
  mode: CompanionMode;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
}> = [
  {
    mode: 'PET_GUARDIAN',
    title: 'I care for a dog',
    description: 'Start dogOS around a real dog you own or are authorized to care for.',
    icon: 'paw-outline',
  },
  {
    mode: 'ANIMAL_ALLY',
    title: 'I’m here to learn and help',
    description: 'Explore Woof without inventing a pet profile or claiming pet access.',
    icon: 'heart-outline',
  },
  {
    mode: 'FOSTER_CAREGIVER',
    title: 'I foster or support dogs',
    description: 'Use a caregiver-first view while pet access stays tied to real relationships.',
    icon: 'home-outline',
  },
];

export default function CompanionModeScreen({
  onResolved,
}: {
  onResolved: (state: CompanionState) => void;
}) {
  const [saving, setSaving] = useState<CompanionMode | null>(null);
  const [error, setError] = useState<string | null>(null);

  const choose = async (mode: CompanionMode) => {
    setSaving(mode);
    setError(null);
    try {
      const state = await companionApi.updateMode(mode);
      onResolved(state);
    } catch {
      setError('Woof could not save that starting role. No pet or relationship access changed.');
    } finally {
      setSaving(null);
    }
  };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.brandMark}>
        <Ionicons name="paw" size={24} color="#ffffff" />
      </View>
      <Text style={styles.eyebrow}>WELCOME TO WOOF</Text>
      <Text style={styles.title}>How do you want to start?</Text>
      <Text style={styles.intro}>
        This choice changes the experience you see. It never creates access to a dog. Pet-specific
        dogOS opens only from a real owned or authorized relationship.
      </Text>

      <View style={styles.choiceList}>
        {choices.map((choice) => (
          <Pressable
            key={choice.mode}
            accessibilityRole="button"
            accessibilityState={{ disabled: saving !== null }}
            disabled={saving !== null}
            style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}
            onPress={() => void choose(choice.mode)}
          >
            <View style={styles.iconWrap}>
              <Ionicons name={choice.icon} size={23} color={colors.primary[700]} />
            </View>
            <View style={styles.cardCopy}>
              <Text style={styles.cardTitle}>{choice.title}</Text>
              <Text style={styles.cardText}>{choice.description}</Text>
            </View>
            {saving === choice.mode ? (
              <ActivityIndicator color={colors.primary[600]} />
            ) : (
              <Ionicons name="chevron-forward" size={20} color={colors.gray[400]} />
            )}
          </Pressable>
        ))}
      </View>

      {error && (
        <View style={styles.errorCard} accessibilityRole="alert">
          <Ionicons name="alert-circle-outline" size={19} color={colors.error.dark} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <Text style={styles.permissionText}>
        You can change presentation later. Woof never treats a mode choice as evidence that you own
        or can access a pet.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.default },
  content: { paddingHorizontal: 24, paddingTop: 72, paddingBottom: 40 },
  brandMark: {
    width: 48,
    height: 48,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginBottom: 28,
  },
  eyebrow: {
    color: colors.primary[700],
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.4,
    marginBottom: 9,
  },
  title: { color: colors.gray[900], fontSize: 34, lineHeight: 40, fontWeight: '800' },
  intro: { color: colors.gray[600], fontSize: 15, lineHeight: 23, marginTop: 12 },
  choiceList: { marginTop: 28, gap: 12 },
  card: {
    minHeight: 104,
    borderWidth: 1,
    borderColor: colors.gray[200],
    borderRadius: 20,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
  },
  cardPressed: { backgroundColor: colors.primary[50] },
  iconWrap: {
    width: 46,
    height: 46,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
    marginRight: 14,
  },
  cardCopy: { flex: 1, paddingRight: 8 },
  cardTitle: { color: colors.gray[900], fontSize: 16, fontWeight: '800', marginBottom: 5 },
  cardText: { color: colors.gray[600], fontSize: 13, lineHeight: 19 },
  errorCard: {
    marginTop: 18,
    borderRadius: 14,
    padding: 14,
    flexDirection: 'row',
    gap: 9,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 13, lineHeight: 19 },
  permissionText: {
    color: colors.gray[500],
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
    marginTop: 24,
  },
});
